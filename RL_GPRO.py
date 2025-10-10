import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import deque

# --- 1. Configuration ---
GRID_SIZE = 12
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
LEARNING_RATE = 1e-4
TOTAL_TIMESTEPS = 400_000
NUM_ENVS = 64          # For training
STEPS_PER_UPDATE = 64  # For training
MAX_EPISODE_STEPS = GRID_SIZE * 4 # Max steps per episode

# GRPO-specific hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPOCHS_PER_UPDATE = 10
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
GRPO_BETA = 0.1  # Temperature parameter for GRPO objective

# Evaluation settings
LOG_INTERVAL = 10    # Print logs every 10 updates
EVAL_INTERVAL = 20   # Run evaluation every 20 updates
EVAL_EPISODES = 256  # Number of parallel episodes for a stable evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

VOCAB = [' ', '#', 'S', 'E', 'A']
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# --- 2. Vectorized Grid World Environment ---
class VectorizedGridWorldEnv:
    def __init__(self, num_envs, grid_size, device):
        self.num_envs, self.grid_size, self.device = num_envs, grid_size, device
        self.action_space = 4
        self.grids = torch.full((num_envs, grid_size, grid_size), stoi[' '], dtype=torch.long, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.start_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.end_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.max_steps = MAX_EPISODE_STEPS
    
    def reset(self):
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_state()
    
    def get_state(self):
        state = self.grids.clone()
        state[(state == stoi['S']) | (state == stoi['E']) | (state == stoi['A'])] = stoi[' ']
        
        agent_not_at_start = (self.agent_pos[:, 0] != self.start_pos[:, 0]) | (self.agent_pos[:, 1] != self.start_pos[:, 1])
        if agent_not_at_start.any():
            valid_envs = torch.where(agent_not_at_start)[0]
            state[valid_envs, self.start_pos[valid_envs, 0], self.start_pos[valid_envs, 1]] = stoi['S']
        
        state[torch.arange(self.num_envs, device=self.device), self.end_pos[:, 0], self.end_pos[:, 1]] = stoi['E']
        state[torch.arange(self.num_envs, device=self.device), self.agent_pos[:, 0], self.agent_pos[:, 1]] = stoi['A']
        
        return state
    
    def step(self, actions):
        self.episode_steps += 1
        next_pos = self.agent_pos.clone()
        next_pos[actions == 0, 0] -= 1  # up
        next_pos[actions == 1, 0] += 1  # down
        next_pos[actions == 2, 1] -= 1  # left
        next_pos[actions == 3, 1] += 1  # right
        
        out_of_bounds = ((next_pos < 0) | (next_pos >= self.grid_size)).any(dim=1)
        wall_collisions = torch.zeros_like(out_of_bounds)
        in_bounds_indices = torch.where(~out_of_bounds)[0]
        if len(in_bounds_indices) > 0:
            in_bounds_pos = next_pos[in_bounds_indices]
            wall_collisions[in_bounds_indices] = self.grids[in_bounds_indices, in_bounds_pos[:, 0], in_bounds_pos[:, 1]] == stoi['#']
        
        invalid_moves = out_of_bounds | wall_collisions
        self.agent_pos[~invalid_moves] = next_pos[~invalid_moves]
        
        rewards = torch.full((self.num_envs,), -0.01, device=self.device, dtype=torch.float)
        rewards[invalid_moves] = -0.1
        
        goal_reached = (self.agent_pos[:, 0] == self.end_pos[:, 0]) & (self.agent_pos[:, 1] == self.end_pos[:, 1])
        rewards[goal_reached] = 1.0
        
        timeout = self.episode_steps >= self.max_steps
        dones = goal_reached | timeout
        
        if dones.any():
            self._reset_envs(dones)
        
        return self.get_state(), rewards, dones, goal_reached
    
    def _reset_envs(self, done_mask):
        num_to_reset = done_mask.sum().item()
        if num_to_reset == 0:
            return
        
        self.episode_steps[done_mask] = 0
        self.grids[done_mask] = torch.full((self.grid_size, self.grid_size), stoi[' '], device=self.device)
        
        for _ in range(int(self.grid_size * 1.5)):
            coords = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
            self.grids[done_mask, coords[:, 0], coords[:, 1]] = stoi['#']
        
        self.agent_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        self.start_pos[done_mask] = self.agent_pos[done_mask].clone()
        self.end_pos[done_mask] = torch.randint(0, self.grid_size, (num_to_reset, 2), device=self.device)
        
        env_indices = torch.arange(self.num_envs, device=self.device)[done_mask]
        self.grids[env_indices, self.agent_pos[done_mask, 0], self.agent_pos[done_mask, 1]] = stoi['S']
        self.grids[env_indices, self.end_pos[done_mask, 0], self.end_pos[done_mask, 1]] = stoi['E']

# --- 3. Model Architecture (Actor-Critic) ---
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

class ActorCriticNetwork(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, num_actions):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
        encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.actor_head = nn.Linear(d_model, num_actions)
        self.critic_head = nn.Linear(d_model, 1)

    def get_action_and_value(self, board, action=None, deterministic=False):
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
        value = self.critic_head(cls_output)
        logits = self.actor_head(cls_output)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

# --- 4. Rigorous Evaluation Function ---
def evaluate_agent(agent, grid_size, device, num_episodes=256):
    """Runs the agent in a deterministic mode on a fixed number of episodes for stable evaluation."""
    print("\n--- Running Validation ---")
    eval_env = VectorizedGridWorldEnv(num_envs=num_episodes, grid_size=grid_size, device=device)
    states = eval_env.reset()
    
    episode_lengths = torch.zeros(num_episodes, device=device)
    active_episodes = torch.ones(num_episodes, device=device, dtype=torch.bool)
    
    successes = 0
    total_path_length = 0
    total_shortest_path = 0

    start_positions = eval_env.agent_pos.clone()
    end_positions = eval_env.end_pos.clone()

    for _ in range(MAX_EPISODE_STEPS):
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(states, deterministic=True)
        
        states, _, dones, terminals = eval_env.step(actions)
        
        episode_lengths[active_episodes] += 1
        
        newly_finished = torch.where(terminals & active_episodes)[0]

        if len(newly_finished) > 0:
            successes += len(newly_finished)
            
            for i in newly_finished:
                path_len = episode_lengths[i].item()
                start_pos = start_positions[i]
                end_pos = end_positions[i]
                shortest_path = (torch.abs(start_pos - end_pos)).sum().item()
                
                total_path_length += path_len
                total_shortest_path += shortest_path if shortest_path > 0 else 1
            
            active_episodes[newly_finished] = False

        if not active_episodes.any():
            break

    avg_success_rate = (successes / num_episodes) * 100
    avg_success_len = total_path_length / successes if successes > 0 else float('nan')
    avg_efficiency = total_path_length / total_shortest_path if successes > 0 else float('nan')
    
    return {
        "success_rate": avg_success_rate,
        "avg_len": avg_success_len,
        "avg_efficiency": avg_efficiency
    }

# --- 5. GRPO Training Loop ---
def run_grpo_training(agent, env):
    """
    GRPO (Group Relative Policy Optimization) training.
    
    Key differences from PPO:
    1. Uses group-relative advantage comparisons within minibatches
    2. No clipping - uses a softer objective based on relative performance
    3. Explicitly encourages actions with higher relative advantages
    """
    optimizer = optim.AdamW(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * STEPS_PER_UPDATE)
    
    states = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS, GRID_SIZE, GRID_SIZE), dtype=torch.long, device=device)
    actions = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), dtype=torch.long, device=device)
    log_probs = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    rewards = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    dones = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    values = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)
    terminals = torch.zeros((STEPS_PER_UPDATE, NUM_ENVS), device=device)

    print("Starting GRPO Training...")
    global_step = 0
    next_state = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    next_terminal = torch.zeros(NUM_ENVS, device=device)

    for update in range(1, num_updates + 1):
        # --- Collect Rollout (Same as PPO) ---
        for step in range(STEPS_PER_UPDATE):
            global_step += NUM_ENVS
            states[step], dones[step] = next_state, next_done
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_state)
                values[step] = value.flatten()
            actions[step], log_probs[step] = action, log_prob
            next_state, reward, next_done, next_terminal = env.step(action)
            rewards[step], terminals[step] = reward, next_terminal

        # --- GAE Calculation (Same as PPO) ---
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_state)[3].reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(STEPS_PER_UPDATE)):
                if t == STEPS_PER_UPDATE - 1:
                    next_non_terminal = 1.0 - next_terminal.float()
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - terminals[t + 1].float()
                    next_values = values[t + 1]
                delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            returns = advantages + values

        # --- GRPO Update (Different from PPO) ---
        b_states = states.reshape((-1, GRID_SIZE, GRID_SIZE))
        b_actions, b_log_probs = actions.reshape(-1), log_probs.reshape(-1)
        b_advantages, b_returns = advantages.reshape(-1), returns.reshape(-1)
        b_inds = np.arange(NUM_ENVS * STEPS_PER_UPDATE)
        
        pg_losses, v_losses, entropy_losses = [], [], []

        for epoch in range(EPOCHS_PER_UPDATE):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_ENVS * STEPS_PER_UPDATE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]
                
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                
                # --- GRPO OBJECTIVE ---
                # 1. Compute group-relative advantages
                mb_advantages = b_advantages[mb_inds]
                group_mean = mb_advantages.mean()
                group_std = mb_advantages.std() + 1e-8
                relative_advantages = (mb_advantages - group_mean) / group_std
                
                # 2. Compute importance ratio
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)
                
                # 3. GRPO loss: weighted by relative advantages with a soft constraint
                # Instead of clipping, we use a KL-regularized objective
                # This encourages the policy to improve actions with high relative advantages
                # while staying close to the old policy
                pg_loss = -(relative_advantages * ratio - GRPO_BETA * (ratio - 1)**2).mean()
                
                # Alternative GRPO formulation (comment out the above and use this if you prefer):
                # This version uses a margin-based approach
                # pg_loss = -(relative_advantages * ratio).mean() + GRPO_BETA * (ratio.log()**2).mean()
                
                # Value and entropy losses (same as PPO)
                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                
                # Combined loss
                loss = pg_loss - ENTROPY_COEF * entropy_loss + VALUE_COEF * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

        # --- Logging ---
        if update % LOG_INTERVAL == 0:
            print(f"--- Update {update}, Global Step: {global_step} ---")
            print(f"  Training Diagnostics:")
            print(f"    Policy Loss: {np.mean(pg_losses):.4f}, Value Loss: {np.mean(v_losses):.4f}, Entropy: {np.mean(entropy_losses):.4f}")
            
            if update % EVAL_INTERVAL == 0:
                eval_metrics = evaluate_agent(agent, GRID_SIZE, device, num_episodes=EVAL_EPISODES)
                print(f"  Validation Performance:")
                print(f"    Success Rate:        {eval_metrics['success_rate']:.1f}%")
                print(f"    Avg Success Length:  {eval_metrics['avg_len']:.1f} steps")
                print(f"    Avg Efficiency:      {eval_metrics['avg_efficiency']:.2f} (actual/shortest)")
            print()

    print("Training finished.")

# --- 6. Inference Function ---
def infer(policy_model, env, stoi, itos):
    print("\n--- Running Inference on a single environment ---")
    single_env = VectorizedGridWorldEnv(num_envs=1, grid_size=env.grid_size, device=env.device)
    state = single_env.reset()
    max_steps = MAX_EPISODE_STEPS 
    
    for step in range(max_steps):
        state_np = state.cpu().numpy()[0]
        print(f"\nStep {step+1}")
        print("\n".join("".join([itos[c] for c in row]) for row in state_np))
        
        with torch.no_grad():
            action, _, _, _ = policy_model.get_action_and_value(state, deterministic=True)
        
        state, reward, done, goal_reached = single_env.step(action)
        
        if goal_reached.any():
            print(f"\n✓ Success! Agent reached the exit in {step+1} steps (reward: {reward.item():.2f}).")
            return
        
        if done.any():
            print(f"\n✗ Failure. Agent timed out after {step+1} steps.")
            return
    
    print(f"\n✗ Failure. Agent did not reach the exit in {max_steps} steps.")

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    env = VectorizedGridWorldEnv(num_envs=NUM_ENVS, grid_size=GRID_SIZE, device=device)
    agent = ActorCriticNetwork(
        vocab_size=len(VOCAB), d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        grid_size=(GRID_SIZE, GRID_SIZE), num_actions=env.action_space
    ).to(device)
    run_grpo_training(agent, env)
    infer(agent, env, stoi, itos)