import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math

# --- 1. Configuration ---
GRID_SIZE = 8
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
MAX_STEPS_PER_EPISODE = 50
GAMMA = 0.99 # Discount factor for future rewards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Grid World Environment ---
class GridWorldEnv:
    """A simple grid world environment for the RL agent."""
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.action_space = 4 # 0:Up, 1:Down, 2:Left, 3:Right
        self.agent_pos = None
        self.start_pos = None
        self.end_pos = None
        self.grid = None
        self.reset()

    def reset(self):
        """Resets the environment to a new random maze."""
        self.grid = np.full((self.grid_size, self.grid_size), ' ')
        
        for _ in range(self.grid_size * 2):
            r, c = np.random.randint(0, self.grid_size, 2)
            self.grid[r, c] = '#'
            
        self.start_pos = tuple(np.random.randint(0, self.grid_size, 2))
        self.grid[self.start_pos] = ' '
        self.end_pos = tuple(np.random.randint(0, self.grid_size, 2))
        while np.all(self.start_pos == self.end_pos) or self.grid[self.end_pos] == '#':
            self.end_pos = tuple(np.random.randint(0, self.grid_size, 2))
        self.grid[self.start_pos] = 'S'
        self.grid[self.end_pos] = 'E'
        
        self.agent_pos = self.start_pos
        return self.get_state()

    def get_state(self):
        """Returns the current grid state with the agent marked as 'A'."""
        state = self.grid.copy()
        state[self.agent_pos] = 'A'
        return state

    def step(self, action):
        """Performs an action and returns (next_state, reward, done)."""
        r, c = self.agent_pos
        if action == 0: r -= 1 # Up
        elif action == 1: r += 1 # Down
        elif action == 2: c -= 1 # Left
        elif action == 3: c += 1 # Right

        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size or self.grid[r, c] == '#':
            return self.get_state(), -1.0, False 
        
        self.agent_pos = (r, c)
        if self.agent_pos == self.end_pos:
            return self.get_state(), 10.0, True
        else:
            return self.get_state(), -0.1, False

# --- 3. Model Architecture ---
# We reuse the powerful RoPE Transformer components from before
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
    """The Transformer model adapted to be an RL policy network."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, num_actions):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
        encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, board):
        B, H, W = board.shape
        flat_board = board.view(B, H * W)
        token_emb = self.token_embedding(flat_board)
        
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_tokens, token_emb], dim=1)

        rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W)
        cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
        pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2)
        cls_pos_ids = torch.zeros(B, 1, 2, dtype=torch.long, device=board.device)
        pos_ids = torch.cat([cls_pos_ids, pos_ids], dim=1)
        
        for layer in self.transformer_encoder.layers:
            x = layer(x, pos_ids=pos_ids, rope=self.rope)
        
        cls_output = x[:, 0] # Use the [CLS] token's output
        action_logits = self.action_head(cls_output)
        return nn.functional.softmax(action_logits, dim=-1)

# --- 4. Training Loop ---
def run_rl_training(policy_model, env, stoi, epochs, lr, max_steps_per_episode, gamma):
    optimizer = optim.AdamW(policy_model.parameters(), lr=lr)
    
    print("Starting RL Training...")
    for epoch in range(epochs):
        log_probs = []; rewards = []
        state = env.reset()
        
        for t in range(max_steps_per_episode):
            state_tensor = torch.tensor([[stoi.get(c, 0) for c in row] for row in state], dtype=torch.long).unsqueeze(0).to(device)
            action_probs = policy_model(state_tensor)
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            state, reward, done = env.step(action.item())
            rewards.append(reward)
            if done: break

        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.tensor(returns, device=device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = [-log_prob * R for log_prob, R in zip(log_probs, returns)]
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Episode Length: {t+1}, Total Reward: {sum(rewards):.2f}")

    print("Training finished.")

# --- 5. Inference Function ---
def infer(policy_model, env, stoi, itos):
    print("\n--- Running Inference ---")
    state = env.reset()
    done = False
    step = 0
    max_steps = env.grid_size * 2
    
    while not done and step < max_steps:
        print(f"\nStep {step+1}")
        print("\n".join("".join(row) for row in state))
        
        state_tensor = torch.tensor([[stoi.get(c, 0) for c in row] for row in state], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = policy_model(state_tensor)
        
        # In inference, we take the best action instead of sampling
        action = torch.argmax(action_probs).item()
        
        state, reward, done = env.step(action)
        step += 1
    
    print("\nFinal Board State:")
    print("\n".join("".join(row) for row in state))
    if done:
        print("\nSuccess! Agent reached the exit.")
    else:
        print("\nFailure. Agent did not reach the exit.")

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    # Vocabulary for the grid world
    VOCAB = [' ', '#', 'S', 'E', 'A']
    stoi = {c: i for i, c in enumerate(VOCAB)}
    itos = {i: c for i, c in enumerate(VOCAB)}
    
    # Initialize environment and policy model
    env = GridWorldEnv(grid_size=GRID_SIZE)
    policy_model = RLPolicyNetwork(
        vocab_size=len(VOCAB),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        grid_size=(GRID_SIZE, GRID_SIZE),
        num_actions=env.action_space
    ).to(device)
    
    # Train the model
    run_rl_training(policy_model, env, stoi, NUM_EPOCHS, LEARNING_RATE, MAX_STEPS_PER_EPISODE, GAMMA)
    
    # See how it performs
    infer(policy_model, env, stoi, itos)