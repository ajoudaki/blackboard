import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
from collections import deque

# --- 1. Configuration ---
GRID_SIZE = 12
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 500_000
NUM_ENVS = 64              # Number of parallel environments
STEPS_PER_UPDATE = 64     # Number of parallel steps to collect before an update

# PPO-specific hyperparameters
GAMMA = 0.99               # Discount factor for future rewards
GAE_LAMBDA = 0.95          # Lambda for Generalized Advantage Estimation
CLIP_EPS = 0.2             # PPO clipping parameter
EPOCHS_PER_UPDATE = 10     # How many times to loop over the collected data
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5           # Value function loss coefficient
ENTROPY_COEF = 0.01        # Entropy bonus coefficient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

VOCAB = [' ', '#', 'S', 'E', 'A']
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# --- 2. Vectorized Grid World Environment (Unchanged) ---
class VectorizedGridWorldEnv:
    def __init__(self, num_envs, grid_size, device):
        self.num_envs, self.grid_size, self.device = num_envs, grid_size, device
        self.action_space = 4
        self.grids = torch.full((num_envs, grid_size, grid_size), stoi[' '], dtype=torch.long, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.end_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
    def reset(self):
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device)); return self.get_state()
    def get_state(self):
        state = self.grids.clone(); state[torch.arange(self.num_envs, device=self.device), self.agent_pos[:, 0], self.agent_pos[:, 1]] = stoi['A']; return state
    def step(self, actions):
        next_pos = self.agent_pos.clone()
        next_pos[actions == 0, 0] -= 1; next_pos[actions == 1, 0] += 1
        next_pos[actions == 2, 1] -= 1; next_pos[actions == 3, 1] += 1
        out_of_bounds = (next_pos < 0) | (next_pos >= self.grid_size); out_of_bounds = out_of_bounds.any(dim=1)
        wall_collisions = torch.zeros_like(out_of_bounds); in_bounds_indices = torch.where(~out_of_bounds)[0]
        if len(in_bounds_indices) > 0:
            in_bounds_pos = next_pos[in_bounds_indices]
            wall_collisions[in_bounds_indices] = self.grids[in_bounds_indices, in_bounds_pos[:, 0], in_bounds_pos[:, 1]] == stoi['#']
        invalid_moves = out_of_bounds | wall_collisions
        self.agent_pos[~invalid_moves] = next_pos[~invalid_moves]
        rewards = torch.full((self.num_envs,), -0.01, device=self.device, dtype=torch.float) # Smaller step penalty
        rewards[invalid_moves] = -0.2
        dones = (self.agent_pos[:, 0] == self.end_pos[:, 0]) & (self.agent_pos[:, 1] == self.end_pos[:, 1])
        rewards[dones] = 1.0 # Normalized reward
        if dones.any(): self._reset_envs(dones)
        return self.get_state(), rewards, dones
    def _reset_envs(self, done_mask):
        num_to_reset = done_mask.sum().item()
        if num_to_reset == 0: return
        self.grids[done_mask] = torch.full((self.grid_size, self.grid_size), stoi[' '], device=self.device)
        for _ in range(int(self.grid_size * 1.5)): # Fewer walls for easier navigation
            coords = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
            self.grids[done_mask, coords[:, 0], coords[:, 1]] = stoi['#']
        self.agent_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        self.end_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        env_indices = torch.arange(self.num_envs, device=self.device)[done_mask]
        self.grids[env_indices, self.agent_pos[done_mask, 0], self.agent_pos[done_mask, 1]] = stoi['S']
        self.grids[env_indices, self.end_pos[done_mask, 0], self.end_pos[done_mask, 1]] = stoi['E']

# --- 3. Model Architecture (Actor-Critic) ---
class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    # (Unchanged)
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
    # (Unchanged)
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

class ActorCriticNetwork(nn.Module):
    """A single Transformer model with two heads: one for policy (actor) and one for value (critic)."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, num_actions):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
        encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.actor_head = nn.Linear(d_model, num_actions) # Policy head
        self.critic_head = nn.Linear(d_model, 1)        # Value head

    def get_action_and_value(self, board, action=None):
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
        
        # Get value and action logits from the shared body
        value = self.critic_head(cls_output)
        logits = self.actor_head(cls_output)
        
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), value

# --- 4. PPO Training Loop ---
def run_ppo_training(agent, env):
    optimizer = optim.AdamW(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_UPDATE)
    
    # Storage for the rollout
    states = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS, GRID_SIZE, GRID_SIZE), dtype=torch.long, device=device)
    actions = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), dtype=torch.long, device=device)
    log_probs = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    rewards = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    dones = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    values = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    
    print("Starting PPO Training...")
    global_step = 0
    next_state = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    
    for update in range(1, num_updates + 1):
        # --- 1. Collect a Rollout of Experience ---
        for step in range(STEPS_PER_UPDATE):
            global_step += NUM_ENVS
            states[step], dones[step] = next_state, next_done
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_state)
                values[step] = value.flatten()
            actions[step], log_probs[step] = action, log_prob
            next_state, reward, next_done = env.step(action)
            rewards[step] = reward
        
        # --- 2. Calculate Advantages using GAE ---
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_state)[3].reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(STEPS_PER_UPDATE)):
                if t == STEPS_PER_UPDATE - 1:
                    next_non_terminal = 1.0 - next_done.float()
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t+1].float()
                    next_values = values[t+1]
                delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            returns = advantages + values

        # --- 3. Perform PPO Update ---
        b_states = states.reshape((-1, GRID_SIZE, GRID_SIZE))
        b_actions, b_log_probs = actions.reshape(-1), log_probs.reshape(-1)
        b_advantages, b_returns = advantages.reshape(-1), returns.reshape(-1)
        
        b_inds = np.arange(NUM_ENVS * STEPS_PER_UPDATE)
        
        # Buffers to store loss values for logging
        pg_losses, v_losses, entropy_losses = [], [], []

        for epoch in range(EPOCHS_PER_UPDATE):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * STEPS_PER_UPDATE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_states[mb_inds], b_actions[mb_inds]
                )
                
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - ENTROPY_COEF * entropy_loss + VALUE_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                # Append individual losses
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        if update % 10 == 0:
            print(
                f"Update {update}, Global Step: {global_step}\n"
                f"  Policy Loss: {np.mean(pg_losses):.4f}, "
                f"Value Loss: {np.mean(v_losses):.4f}, "
                f"Entropy: {np.mean(entropy_losses):.4f}\n"
            )

    print("Training finished.")

# --- 5. Inference Function (Unchanged) ---
def infer(policy_model, env, stoi, itos):
    print("\n--- Running Inference on a single environment ---")
    single_env = VectorizedGridWorldEnv(num_envs=1, grid_size=env.grid_size, device=env.device)
    state = single_env.reset()
    done = torch.tensor([False], device=env.device)
    step = 0; max_steps = env.grid_size * 4
    while not done.any() and step < max_steps:
        state_np = state.cpu().numpy()[0]
        print(f"\nStep {step+1}"); print("\n".join("".join([itos[c] for c in row]) for row in state_np))
        with torch.no_grad():
            action, _, _, _ = policy_model.get_action_and_value(state)
        state, _, done = single_env.step(action)
        step += 1
    state_np = state.cpu().numpy()[0]
    print("\nFinal Board State:"); print("\n".join("".join([itos[c] for c in row]) for row in state_np))
    if done.any(): print("\nSuccess! Agent reached the exit.")
    else: print("\nFailure. Agent did not reach the exit in time.")

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    env = VectorizedGridWorldEnv(num_envs=NUM_ENVS, grid_size=GRID_SIZE, device=device)
    agent = ActorCriticNetwork(
        vocab_size=len(VOCAB), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        grid_size=(GRID_SIZE, GRID_SIZE), num_actions=env.action_space
    ).to(device)
    
    run_ppo_training(agent, env)
    infer(agent, env, stoi, itos)