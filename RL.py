import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
from collections import deque

# --- 1. Configuration ---
GRID_SIZE = 8
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 2_000_000
NUM_ENVS = 64
STEPS_PER_UPDATE = 64
GAMMA = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

VOCAB = [' ', '#', 'S', 'E', 'A']
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# --- 2. Vectorized Grid World Environment ---
class VectorizedGridWorldEnv:
    def __init__(self, num_envs, grid_size, device):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.device = device
        self.action_space = 4

        self.grids = torch.full((num_envs, grid_size, grid_size), stoi[' '], dtype=torch.long, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.end_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        
    def reset(self):
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_state()
        
    def get_state(self):
        state = self.grids.clone()
        state[torch.arange(self.num_envs, device=self.device), self.agent_pos[:, 0], self.agent_pos[:, 1]] = stoi['A']
        return state

    def step(self, actions):
        next_pos = self.agent_pos.clone()
        next_pos[actions == 0, 0] -= 1; next_pos[actions == 1, 0] += 1
        next_pos[actions == 2, 1] -= 1; next_pos[actions == 3, 1] += 1

        out_of_bounds = (next_pos < 0) | (next_pos >= self.grid_size)
        out_of_bounds = out_of_bounds.any(dim=1)

        wall_collisions = torch.zeros_like(out_of_bounds)
        in_bounds_indices = torch.where(~out_of_bounds)[0]
        
        if len(in_bounds_indices) > 0:
            in_bounds_pos = next_pos[in_bounds_indices]
            wall_collisions[in_bounds_indices] = self.grids[in_bounds_indices, in_bounds_pos[:, 0], in_bounds_pos[:, 1]] == stoi['#']
        
        invalid_moves = out_of_bounds | wall_collisions
        self.agent_pos[~invalid_moves] = next_pos[~invalid_moves]
        
        rewards = torch.full((self.num_envs,), -0.1, device=self.device, dtype=torch.float)
        rewards[invalid_moves] = -1.0
        
        dones = (self.agent_pos[:, 0] == self.end_pos[:, 0]) & (self.agent_pos[:, 1] == self.end_pos[:, 1])
        rewards[dones] = 10.0
        
        if dones.any():
            self._reset_envs(dones)

        return self.get_state(), rewards, dones

    def _reset_envs(self, done_mask):
        num_to_reset = done_mask.sum().item()
        if num_to_reset == 0: return
        
        self.grids[done_mask] = torch.full((self.grid_size, self.grid_size), stoi[' '], device=self.device)
        for _ in range(self.grid_size * 2):
            coords = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
            self.grids[done_mask, coords[:, 0], coords[:, 1]] = stoi['#']
        
        self.agent_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        self.end_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        
        # Get the global indices for environments that are done
        # CORRECTED: Create the arange tensor on the correct device
        env_indices = torch.arange(self.num_envs, device=self.device)[done_mask]

        self.grids[env_indices, self.agent_pos[done_mask, 0], self.agent_pos[done_mask, 1]] = stoi['S']
        self.grids[env_indices, self.end_pos[done_mask, 0], self.end_pos[done_mask, 1]] = stoi['E']

# --- 3. Model Architecture ---
class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
    def _sa_block(self, x, attn_mask, key_padding_mask, pos_ids=None, rope=None):
        if rope is not None and pos_ids is not None:
            q = k = rope.rotate_queries_and_keys(x, pos_ids)
            return self.self_attn(q, k, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else: return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_ids=None, rope=None):
        x = src; x = x + self.dropout1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, pos_ids, rope)); x = x + self.dropout2(self._ff_block(self.norm2(x))); return x

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, d_model, grid_size):
        super().__init__(); self.d_row, self.d_col = d_model // 2, d_model - d_model // 2; self.max_h, self.max_w = grid_size
        freqs_row = self._precompute_freqs(self.d_row, self.max_h); freqs_col = self._precompute_freqs(self.d_col, self.max_w)
        self.register_buffer('freqs_row_cos', freqs_row['cos']); self.register_buffer('freqs_row_sin', freqs_row['sin'])
        self.register_buffer('freqs_col_cos', freqs_col['cos']); self.register_buffer('freqs_col_sin', freqs_col['sin'])
    def _precompute_freqs(self, dim, max_pos, theta=10000.0):
        inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2).float() / dim)); t = torch.arange(max_pos, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq); emb = torch.cat((freqs, freqs), dim=-1); return {'cos': emb.cos(), 'sin': emb.sin()}
    def _apply_rotary_emb(self, x, cos, sin):
        x2 = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1); return x * cos + x2 * sin
    def rotate_queries_and_keys(self, x, pos_ids):
        x_row, x_col = x[..., :self.d_row], x[..., self.d_row:]; row_ids, col_ids = pos_ids[..., 0], pos_ids[..., 1]
        cos_row, sin_row = self.freqs_row_cos[row_ids], self.freqs_row_sin[row_ids]; cos_col, sin_col = self.freqs_col_cos[col_ids], self.freqs_col_sin[col_ids]
        return torch.cat([self._apply_rotary_emb(x_row, cos_row, sin_row), self._apply_rotary_emb(x_col, cos_col, sin_col)], dim=-1)

class RLPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, num_actions):
        super().__init__(); self.grid_h, self.grid_w = grid_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
        encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.action_head = nn.Linear(d_model, num_actions)
    def forward(self, board):
        B, H, W = board.shape; flat_board = board.view(B, H * W)
        token_emb = self.token_embedding(flat_board)
        cls_tokens = self.cls_token.repeat(B, 1, 1); x = torch.cat([cls_tokens, token_emb], dim=1)
        rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W)
        cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
        pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2)
        cls_pos_ids = torch.zeros(B, 1, 2, dtype=torch.long, device=board.device)
        pos_ids = torch.cat([cls_pos_ids, pos_ids], dim=1)
        for layer in self.transformer_encoder.layers:
            x = layer(x, pos_ids=pos_ids, rope=self.rope)
        cls_output = x[:, 0]
        action_logits = self.action_head(cls_output)
        return nn.functional.softmax(action_logits, dim=-1)

# --- 4. Vectorized Training Loop ---
def run_rl_training(policy_model, env, num_updates, steps_per_update, lr, gamma):
    optimizer = optim.AdamW(policy_model.parameters(), lr=lr)
    print("Starting Vectorized RL Training...")
    recent_rewards = deque(maxlen=100)
    state = env.reset()
    for update in range(num_updates):
        log_probs = []; rewards = []
        for _ in range(steps_per_update):
            action_probs = policy_model(state)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            next_state, reward_batch, done_batch = env.step(actions)
            log_probs.append(dist.log_prob(actions)); rewards.append(reward_batch)
            state = next_state
            for i, done in enumerate(done_batch):
                if done: recent_rewards.append(sum(r[i].item() for r in rewards)) # A rough estimate
        returns = torch.zeros(steps_per_update, env.num_envs, device=device)
        discounted_return = 0
        for i in reversed(range(steps_per_update)):
            discounted_return = rewards[i] + gamma * discounted_return
            returns[i] = discounted_return
        returns = returns.view(-1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        log_probs = torch.cat(log_probs)
        policy_loss = -(log_probs * returns).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if update % 50 == 0 and len(recent_rewards) > 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Update {update}, Avg Reward (last 100 episodes): {avg_reward:.2f}")
    print("Training finished.")


def infer(policy_model, env, stoi, itos):
    """Runs inference on a single environment instance using the VectorizedEnv."""
    print("\n--- Running Inference on a single environment ---")
    
    single_env = VectorizedGridWorldEnv(num_envs=1, grid_size=env.grid_size, device=env.device)
    state = single_env.reset()
    done = torch.tensor([False], device=env.device)
    step = 0
    max_steps = env.grid_size * 4
    
    while not done.any() and step < max_steps:
        # CORRECTED: Call .cpu() before converting to numpy for printing
        state_np = state.cpu().numpy()[0]
        print(f"\nStep {step+1}")
        print("\n".join("".join([itos[c] for c in row]) for row in state_np))

        with torch.no_grad():
            action_probs = policy_model(state)
        
        action = torch.argmax(action_probs, dim=-1)
        
        state, _, done = single_env.step(action)
        step += 1
    
    # CORRECTED: Call .cpu() before converting the final state to numpy
    state_np = state.cpu().numpy()[0]
    print("\nFinal Board State:")
    print("\n".join("".join([itos[c] for c in row]) for row in state_np))
    
    if done.any():
        print("\nSuccess! Agent reached the exit.")
    else:
        print("\nFailure. Agent did not reach the exit in time.")

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_UPDATE)
    env = VectorizedGridWorldEnv(num_envs=NUM_ENVS, grid_size=GRID_SIZE, device=device)
    policy_model = RLPolicyNetwork(
        vocab_size=len(VOCAB), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        grid_size=(GRID_SIZE, GRID_SIZE), num_actions=env.action_space
    ).to(device)
    run_rl_training(policy_model, env, num_updates=num_updates, steps_per_update=STEPS_PER_UPDATE, lr=LEARNING_RATE, gamma=GAMMA)
    infer(policy_model, env, stoi, itos)