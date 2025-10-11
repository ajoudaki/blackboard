import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================================
# Hierarchical Configuration
# ============================================================================
CONFIG = {
    # Shared parameters (used by both env and model)
    'shared': {
        'grid_size': 12,
        'vocab': [' ', '#', 'S', 'E', 'A'],
        'num_actions': 4,
    },
    # Environment-only parameters
    'env': {
        'wall_density': 1.5,
        'max_episode_steps_multiplier': 4,
        'reward_step': -0.01,
        'reward_collision': -0.2,
        'reward_goal': 1.0,
    },
    # Model-only parameters
    'model': {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward_multiplier': 2,
        'dropout': 0.1,
        'rope_theta': 10000.0,
    },
    # Training parameters
    'training': {
        'total_timesteps': 1_000_000,
        'num_envs': 64,
        'steps_per_update': 64,
        'learning_rate': 1e-4,
        'optimizer_eps': 1e-8,
        'grad_clip_norm': 0.5,
        'epochs_per_update': 10,
        'minibatch_size': 256,
    },
    # GRPO algorithm parameters
    'grpo': {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'grpo_beta': 0.1,
        'value_coef': 0.5,
        'entropy_coef': 0.1,
    },
    # Logging parameters
    'logging': {
        'log_interval': 10,
        'eval_interval': 20,
        'eval_episodes': 256,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ============================================================================
# Helper Functions
# ============================================================================
def get_char_mappings(vocab):
    """Create character to index and index to character mappings."""
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}

stoi, itos = get_char_mappings(CONFIG['shared']['vocab'])


# ============================================================================
# Environment
# ============================================================================
class VectorizedGridWorldEnv:
    def __init__(self, num_envs, device, grid_size, vocab, num_actions, 
                 wall_density, max_episode_steps_multiplier, reward_step, 
                 reward_collision, reward_goal):
        self.num_envs = num_envs
        self.device = device
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.max_steps = grid_size * max_episode_steps_multiplier
        
        # Reward parameters
        self.reward_step = reward_step
        self.reward_collision = reward_collision
        self.reward_goal = reward_goal
        
        # State tensors
        self.grids = torch.full((num_envs, grid_size, grid_size), 
                                stoi[' '], dtype=torch.long, device=device)
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.start_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.end_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=device)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    def reset(self):
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_state()
    
    def get_state(self):
        """Render current state with S (start), E (end), A (agent) markers."""
        state = self.grids.clone()
        state[(state == stoi['S']) | (state == stoi['E']) | (state == stoi['A'])] = stoi[' ']
        
        # Add start marker (only if agent moved away)
        agent_moved = (self.agent_pos != self.start_pos).any(dim=1)
        if agent_moved.any():
            envs = torch.where(agent_moved)[0]
            state[envs, self.start_pos[envs, 0], self.start_pos[envs, 1]] = stoi['S']
        
        # Add end and agent markers
        env_idx = torch.arange(self.num_envs, device=self.device)
        state[env_idx, self.end_pos[:, 0], self.end_pos[:, 1]] = stoi['E']
        state[env_idx, self.agent_pos[:, 0], self.agent_pos[:, 1]] = stoi['A']
        
        return state
    
    def step(self, actions):
        """Execute actions and return next state, rewards, dones, goal_reached."""
        self.episode_steps += 1
        
        # Calculate next positions
        next_pos = self.agent_pos.clone()
        next_pos[actions == 0, 0] -= 1  # up
        next_pos[actions == 1, 0] += 1  # down
        next_pos[actions == 2, 1] -= 1  # left
        next_pos[actions == 3, 1] += 1  # right
        
        # Check collisions
        out_of_bounds = ((next_pos < 0) | (next_pos >= self.grid_size)).any(dim=1)
        wall_collision = torch.zeros_like(out_of_bounds)
        
        valid_idx = torch.where(~out_of_bounds)[0]
        if len(valid_idx) > 0:
            valid_pos = next_pos[valid_idx]
            wall_collision[valid_idx] = self.grids[valid_idx, valid_pos[:, 0], valid_pos[:, 1]] == stoi['#']
        
        # Update positions
        valid_moves = ~(out_of_bounds | wall_collision)
        self.agent_pos[valid_moves] = next_pos[valid_moves]
        
        # Calculate rewards (using config parameters)
        rewards = torch.full((self.num_envs,), self.reward_step, device=self.device, dtype=torch.float)
        rewards[~valid_moves] = self.reward_collision
        
        goal_reached = (self.agent_pos == self.end_pos).all(dim=1)
        rewards[goal_reached] = self.reward_goal
        
        # Check done conditions
        timeout = self.episode_steps >= self.max_steps
        dones = goal_reached | timeout
        
        if dones.any():
            self._reset_envs(dones)
        
        return self.get_state(), rewards, dones, goal_reached
    
    def _reset_envs(self, mask):
        """Reset environments specified by mask."""
        num_reset = mask.sum().item()
        if num_reset == 0:
            return
        
        self.episode_steps[mask] = 0
        self.grids[mask] = stoi[' ']
        
        # Add random walls
        num_walls = int(self.grid_size * self.wall_density)
        for _ in range(num_walls):
            coords = torch.randint(0, self.grid_size, (num_reset, 2), device=self.device)
            self.grids[mask, coords[:, 0], coords[:, 1]] = stoi['#']
        
        # Set random start and end positions
        self.agent_pos[mask] = torch.randint(0, self.grid_size, (num_reset, 2), device=self.device)
        self.start_pos[mask] = self.agent_pos[mask].clone()
        self.end_pos[mask] = torch.randint(0, self.grid_size, (num_reset, 2), device=self.device)
        
        # Mark positions in grid
        env_idx = torch.arange(self.num_envs, device=self.device)[mask]
        self.grids[env_idx, self.start_pos[mask, 0], self.start_pos[mask, 1]] = stoi['S']
        self.grids[env_idx, self.end_pos[mask, 0], self.end_pos[mask, 1]] = stoi['E']


# ============================================================================
# Model Architecture
# ============================================================================
class RotaryPositionalEmbedding2D(nn.Module):
    """2D Rotary Position Embeddings for grid-based inputs."""
    def __init__(self, d_model, grid_size, rope_theta):
        super().__init__()
        self.d_row = d_model // 2
        self.d_col = d_model - self.d_row
        
        # Precompute frequencies
        for dim, max_pos, name in [(self.d_row, grid_size, 'row'), 
                                    (self.d_col, grid_size, 'col')]:
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_pos, device=inv_freq.device).type_as(inv_freq)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer(f'freqs_{name}_cos', emb.cos())
            self.register_buffer(f'freqs_{name}_sin', emb.sin())
    
    def _apply_rotary(self, x, cos, sin):
        x2 = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1)
        return x * cos + x2 * sin
    
    def rotate_queries_and_keys(self, x, pos_ids):
        """Apply 2D rotary embeddings to input tensor."""
        x_row, x_col = x[..., :self.d_row], x[..., self.d_row:]
        row_ids, col_ids = pos_ids[..., 0], pos_ids[..., 1]
        
        x_row = self._apply_rotary(x_row, self.freqs_row_cos[row_ids], self.freqs_row_sin[row_ids])
        x_col = self._apply_rotary(x_col, self.freqs_col_cos[col_ids], self.freqs_col_sin[col_ids])
        
        return torch.cat([x_row, x_col], dim=-1)


class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer with RoPE support."""
    def _sa_block(self, x, attn_mask, key_padding_mask, pos_ids=None, rope=None):
        if rope is not None and pos_ids is not None:
            q = k = rope.rotate_queries_and_keys(x, pos_ids)
            return self.self_attn(q, k, x, attn_mask=attn_mask, 
                                key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.self_attn(x, x, x, attn_mask=attn_mask, 
                            key_padding_mask=key_padding_mask, need_weights=False)[0]
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_ids=None, rope=None):
        x = src
        x = x + self.dropout1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, pos_ids, rope))
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x


class ActorCriticNetwork(nn.Module):
    """Transformer-based actor-critic network with 2D RoPE."""
    def __init__(self, vocab, grid_size, num_actions, d_model, nhead, num_layers,
                 dim_feedforward_multiplier, dropout, rope_theta):
        super().__init__()
        vocab_size = len(vocab)
        dim_feedforward = d_model * dim_feedforward_multiplier
        
        self.grid_size = grid_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding2D(d_model, grid_size, rope_theta)
        
        encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                     dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.actor_head = nn.Linear(d_model, num_actions)
        self.critic_head = nn.Linear(d_model, 1)
    
    def forward(self, board, action=None, deterministic=False):
        """Forward pass returning action, log_prob, entropy, value."""
        B, H, W = board.shape
        
        # Embed tokens and add CLS token
        token_emb = self.token_embedding(board.view(B, -1))
        x = torch.cat([self.cls_token.repeat(B, 1, 1), token_emb], dim=1)
        
        # Create 2D position IDs
        rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W)
        cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
        pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2)
        pos_ids = torch.cat([torch.zeros(B, 1, 2, dtype=torch.long, device=board.device), pos_ids], dim=1)
        
        # Apply transformer with RoPE
        for layer in self.transformer.layers:
            x = layer(x, pos_ids=pos_ids, rope=self.rope)
        
        # Get outputs
        cls_output = x[:, 0]
        value = self.critic_head(cls_output)
        logits = self.actor_head(cls_output)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample or select action
        if action is None:
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


# ============================================================================
# Training
# ============================================================================
def compute_gae(rewards, values, terminals, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(steps)):
        if t == steps - 1:
            next_non_terminal = 1.0 - terminals[-1].float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - terminals[t + 1].float()
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    return advantages, advantages + values


def train_grpo(agent, env, device, total_timesteps, num_envs, steps_per_update,
               learning_rate, optimizer_eps, grad_clip_norm, epochs_per_update, 
               minibatch_size, gamma, gae_lambda, grpo_beta, value_coef, 
               entropy_coef, log_interval, eval_interval, eval_episodes, **kwargs):
    """GRPO training loop with group-relative policy optimization."""
    optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, eps=optimizer_eps)
    num_updates = total_timesteps // (num_envs * steps_per_update)
    grid_size = env.grid_size
    
    # Rollout buffers
    states = torch.zeros((steps_per_update, num_envs, grid_size, grid_size), 
                         dtype=torch.long, device=device)
    actions = torch.zeros((steps_per_update, num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((steps_per_update, num_envs), device=device)
    rewards = torch.zeros((steps_per_update, num_envs), device=device)
    values = torch.zeros((steps_per_update, num_envs), device=device)
    terminals = torch.zeros((steps_per_update, num_envs), device=device)
    
    print("Starting GRPO Training...\n")
    next_state = env.reset()
    next_terminal = torch.zeros(num_envs, device=device)
    
    for update in range(1, num_updates + 1):
        # Collect rollout
        for step in range(steps_per_update):
            states[step] = next_state
            with torch.no_grad():
                action, log_prob, _, value = agent(next_state)
                values[step] = value.flatten()
            
            actions[step] = action
            log_probs[step] = log_prob
            next_state, reward, _, next_terminal = env.step(action)
            rewards[step] = reward
            terminals[step] = next_terminal
        
        # Compute advantages
        with torch.no_grad():
            next_value = agent(next_state)[3].reshape(1, -1)
            advantages, returns = compute_gae(rewards, values, terminals, next_value, 
                                             gamma, gae_lambda)
        
        # Prepare batch
        b_states = states.reshape(-1, grid_size, grid_size)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_inds = np.arange(num_envs * steps_per_update)
        
        # Training metrics
        pg_losses, v_losses, ent_losses = [], [], []
        
        # Optimization epochs
        for _ in range(epochs_per_update):
            np.random.shuffle(b_inds)
            
            for start in range(0, len(b_inds), minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                
                _, new_log_prob, entropy, new_value = agent(b_states[mb_inds], b_actions[mb_inds])
                
                # GRPO objective: group-relative advantages with soft KL constraint
                mb_adv = b_advantages[mb_inds]
                rel_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                ratio = torch.exp(new_log_prob - b_log_probs[mb_inds])
                pg_loss = -(rel_adv * ratio - grpo_beta * (ratio - 1)**2).mean()
                
                v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds])**2).mean()
                ent_loss = entropy.mean()
                
                loss = pg_loss - entropy_coef * ent_loss + value_coef * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), grad_clip_norm)
                optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
        
        # Logging
        if update % log_interval == 0:
            print(f"Update {update}/{num_updates}")
            print(f"  Policy: {np.mean(pg_losses):.4f} | "
                  f"Value: {np.mean(v_losses):.4f} | "
                  f"Entropy: {np.mean(ent_losses):.4f}")
            
            if update % eval_interval == 0:
                metrics = evaluate(agent, env, device, eval_episodes)
                print(f"  Success: {metrics['success_rate']:.1f}% | "
                      f"Length: {metrics['avg_len']:.1f} | "
                      f"Efficiency: {metrics['efficiency']:.2f}")
            print()
    
    print("Training complete!")


def evaluate(agent, env, device, num_episodes):
    """Evaluate agent performance."""
    eval_env = VectorizedGridWorldEnv(num_episodes, device, **CONFIG['shared'], **CONFIG['env'])
    states = eval_env.reset()
    
    active = torch.ones(num_episodes, dtype=torch.bool, device=device)
    ep_lengths = torch.zeros(num_episodes, device=device)
    
    successes = 0
    total_len = 0
    total_shortest = 0
    
    start_pos = eval_env.agent_pos.clone()
    end_pos = eval_env.end_pos.clone()
    
    for _ in range(eval_env.max_steps):
        with torch.no_grad():
            actions = agent(states, deterministic=True)[0]
        
        states, _, _, goal_reached = eval_env.step(actions)
        ep_lengths[active] += 1
        
        finished = torch.where(goal_reached & active)[0]
        if len(finished) > 0:
            successes += len(finished)
            for i in finished:
                total_len += ep_lengths[i].item()
                total_shortest += torch.abs(start_pos[i] - end_pos[i]).sum().item()
            active[finished] = False
        
        if not active.any():
            break
    
    return {
        'success_rate': (successes / num_episodes) * 100,
        'avg_len': total_len / successes if successes > 0 else float('nan'),
        'efficiency': total_len / max(total_shortest, 1) if successes > 0 else float('nan')
    }


def infer(agent, env, device):
    """Run inference on a single environment."""
    print("\n=== Inference ===")
    infer_env = VectorizedGridWorldEnv(1, device, **CONFIG['shared'], **CONFIG['env'])
    state = infer_env.reset()
    
    for step in range(infer_env.max_steps):
        # Print state
        grid = state.cpu().numpy()[0]
        print(f"\nStep {step + 1}:")
        print("\n".join("".join(itos[c] for c in row) for row in grid))
        
        # Take action
        with torch.no_grad():
            action = agent(state, deterministic=True)[0]
        
        state, reward, done, goal_reached = infer_env.step(action)
        
        if goal_reached.any():
            print(f"\n✓ Success in {step + 1} steps! (Reward: {reward.item():.2f})")
            return
        
        if done.any():
            print(f"\n✗ Timeout after {step + 1} steps")
            return
    
    print(f"\n✗ Failed to reach goal")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    # Create environment and agent with config unpacking
    env = VectorizedGridWorldEnv(CONFIG['training']['num_envs'], device, 
                                  **CONFIG['shared'], **CONFIG['env'])
    agent = ActorCriticNetwork(**CONFIG['shared'], **CONFIG['model']).to(device)
    
    # Train with all configs unpacked
    train_grpo(agent, env, device, **CONFIG['training'], **CONFIG['grpo'], **CONFIG['logging'])
    
    # Run inference
    infer(agent, env, device)