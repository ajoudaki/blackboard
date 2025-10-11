import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import time

# ============================================================================
# Hierarchical Configuration
# ============================================================================
CONFIG = {
    'shared': {
        'grid_size': 8, # Smaller grid for faster training
        'vocab': [' ', '#', 'S', 'E', 'A'],
        'num_actions': 4,
    },
    'env': {
        'wall_density': 1.2,
        'max_episode_steps_multiplier': 5,
        'reward_step': -0.01, 'reward_collision': -0.1, 'reward_goal': 1.0,
    },
    'model': {
        'd_model': 128, 'nhead': 4, 'num_layers': 6,
        'dim_feedforward_multiplier': 2, 'dropout': 0.1, 'rope_theta': 10000.0,
    },
    # Phase 1: Teach the model to perform BFS
    'bfs_prediction': {
        'total_steps': 500_000,
        'batch_size': 32,
        'learning_rate': 1e-3,
    },
    # Phase 2: Teach the agent to use its learned BFS skill
    'rl_navigation': {
        'total_timesteps': 300_000,
        'num_envs': 64, 'steps_per_update': 64,
        'learning_rate': 1e-4, 'optimizer_eps': 1e-8, 'grad_clip_norm': 0.5,
        'epochs_per_update': 10, 'minibatch_size': 128,
        'thinking_steps': 8, # How many steps the agent "thinks" before acting
    },
    'grpo': {
        'gamma': 0.99, 'gae_lambda': 0.95, 'grpo_beta': 0.1,
        'value_coef': 0.5, 'entropy_coef': 0.02,
    },
    'logging': {
        'log_interval': 10, 'eval_interval': 20, 'eval_episodes': 128,
    }
}

def get_char_mappings(vocab):
    """Create character to index and index to character mappings."""
    return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Vocab for BFS scratchpad (distances 0-99)
BFS_VOCAB = [str(i) for i in range(100)] + ['INF', 'UKN'] # INF for walls, UKN for unvisited
bfs_stoi = {c: i for i, c in enumerate(BFS_VOCAB)}
bfs_itos = {i: c for i, c in enumerate(BFS_VOCAB)}

# Main vocab
stoi, itos = get_char_mappings(CONFIG['shared']['vocab'])

# ============================================================================
# Data Generation for Phase 1 (Learning BFS)
# ============================================================================
def generate_bfs_transitions(grids, goals):
    num_envs, grid_size, _ = grids.shape
    transitions = []
    
    # Initial state: observation grid + empty scratchpad
    obs_channel = grids.clone()
    scratch_channel = torch.full_like(obs_channel, bfs_stoi['UKN'])
    
    # Set goal distance to 0
    env_idx = torch.arange(num_envs, device=device)
    scratch_channel[env_idx, goals[:, 0], goals[:, 1]] = bfs_stoi['0']
    
    # Mark walls as INF
    scratch_channel[obs_channel == stoi['#']] = bfs_stoi['INF']
    
    current_dist = 0
    max_dist = grid_size * grid_size

    for k in range(max_dist):
        prev_scratch = scratch_channel.clone()
        input_board = torch.stack([obs_channel, prev_scratch], dim=1)
        
        # Find all cells with the current distance
        frontier = (scratch_channel == bfs_stoi[str(k)])
        if not frontier.any(): break # No more cells to expand
            
        # Expand to neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Shift the frontier mask
            next_frontier = torch.roll(frontier, shifts=(-dr, -dc), dims=(1, 2))
            # Find neighbors that are currently unvisited ('UKN')
            updatable = (scratch_channel == bfs_stoi['UKN']) & next_frontier
            scratch_channel[updatable] = bfs_stoi[str(k + 1)]

        target_board = torch.stack([obs_channel, scratch_channel.clone()], dim=1)
        
        # Only add transitions where something actually changed
        if not torch.equal(input_board, target_board):
            transitions.append((input_board, target_board))
            
    return transitions

# ============================================================================
# Environment (Unchanged)
# ============================================================================
class VectorizedGridWorldEnv:
    def __init__(self, num_envs, device, **kwargs):
        self.num_envs=num_envs; self.device=device; self.grid_size=kwargs['grid_size']; self.wall_density=kwargs['wall_density']; self.max_steps=kwargs['grid_size']*kwargs['max_episode_steps_multiplier']
        self.reward_step=kwargs['reward_step']; self.reward_collision=kwargs['reward_collision']; self.reward_goal=kwargs['reward_goal']
        self.grids=torch.full((num_envs,kwargs['grid_size'],kwargs['grid_size']),stoi[' '],dtype=torch.long,device=device); self.agent_pos=torch.zeros((num_envs,2),dtype=torch.long,device=device)
        self.start_pos=torch.zeros((num_envs,2),dtype=torch.long,device=device); self.end_pos=torch.zeros((num_envs,2),dtype=torch.long,device=device)
        self.episode_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
    def reset(self): self._reset_envs(torch.ones(self.num_envs,dtype=torch.bool,device=self.device)); return self.get_state()
    def get_state(self):
        state=self.grids.clone(); state[(state==stoi['S'])|(state==stoi['E'])|(state==stoi['A'])]=stoi[' ']
        agent_moved=(self.agent_pos!=self.start_pos).any(dim=1)
        if agent_moved.any(): envs=torch.where(agent_moved)[0]; state[envs,self.start_pos[envs,0],self.start_pos[envs,1]]=stoi['S']
        env_idx=torch.arange(self.num_envs,device=self.device); state[env_idx,self.end_pos[:,0],self.end_pos[:,1]]=stoi['E']; state[env_idx,self.agent_pos[:,0],self.agent_pos[:,1]]=stoi['A']
        return state
    def step(self,actions):
        self.episode_steps+=1; next_pos=self.agent_pos.clone(); next_pos[actions==0,0]-=1; next_pos[actions==1,0]+=1; next_pos[actions==2,1]-=1; next_pos[actions==3,1]+=1
        out_of_bounds=((next_pos<0)|(next_pos>=self.grid_size)).any(dim=1); wall_collision=torch.zeros_like(out_of_bounds); valid_idx=torch.where(~out_of_bounds)[0]
        if len(valid_idx)>0: valid_pos=next_pos[valid_idx]; wall_collision[valid_idx]=self.grids[valid_idx,valid_pos[:,0],valid_pos[:,1]]==stoi['#']
        valid_moves=~(out_of_bounds|wall_collision); self.agent_pos[valid_moves]=next_pos[valid_moves]; rewards=torch.full((self.num_envs,),self.reward_step,device=self.device,dtype=torch.float)
        rewards[~valid_moves]=self.reward_collision; goal_reached=(self.agent_pos==self.end_pos).all(dim=1); rewards[goal_reached]=self.reward_goal
        timeout=self.episode_steps>=self.max_steps; dones=goal_reached|timeout
        if dones.any(): self._reset_envs(dones)
        return self.get_state(),rewards,dones,goal_reached
    def _reset_envs(self,mask):
        num_reset=mask.sum().item()
        if num_reset==0: return
        self.episode_steps[mask]=0; self.grids[mask]=stoi[' ']; num_walls=int(self.grid_size*self.wall_density)
        for _ in range(num_walls): coords=torch.randint(0,self.grid_size,(num_reset,2),device=self.device); self.grids[mask,coords[:,0],coords[:,1]]=stoi['#']
        self.agent_pos[mask]=torch.randint(0,self.grid_size,(num_reset,2),device=self.device); self.start_pos[mask]=self.agent_pos[mask].clone(); self.end_pos[mask]=torch.randint(0,self.grid_size,(num_reset,2),device=self.device)
        env_idx=torch.arange(self.num_envs,device=self.device)[mask]; self.grids[env_idx,self.start_pos[mask,0],self.start_pos[mask,1]]=stoi['S']; self.grids[env_idx,self.end_pos[mask,0],self.end_pos[mask,1]]=stoi['E']

# ============================================================================
# Model Architecture (Unified with 3 Heads)
# ============================================================================
# (RoPE and Transformer Encoder Layer are unchanged)
class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self,d_model,grid_size,rope_theta):
        super().__init__();self.d_row=d_model//2;self.d_col=d_model-self.d_row
        for dim,max_pos,name in[(self.d_row,grid_size,'row'),(self.d_col,grid_size,'col')]:
            inv_freq=1.0/(rope_theta**(torch.arange(0,dim,2).float()/dim));t=torch.arange(max_pos,device=inv_freq.device).type_as(inv_freq)
            freqs=torch.einsum("i,j->ij",t,inv_freq);emb=torch.cat([freqs,freqs],dim=-1)
            self.register_buffer(f'freqs_{name}_cos',emb.cos());self.register_buffer(f'freqs_{name}_sin',emb.sin())
    def _apply_rotary(self,x,cos,sin): x2=torch.cat([-x[...,1::2],x[...,0::2]],dim=-1); return x*cos+x2*sin
    def rotate_queries_and_keys(self,x,pos_ids):
        x_row,x_col=x[...,:self.d_row],x[...,self.d_row:];row_ids,col_ids=pos_ids[...,0],pos_ids[...,1]
        x_row=self._apply_rotary(x_row,self.freqs_row_cos[row_ids],self.freqs_row_sin[row_ids]);x_col=self._apply_rotary(x_col,self.freqs_col_cos[col_ids],self.freqs_col_sin[col_ids])
        return torch.cat([x_row,x_col],dim=-1)

class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,d_model,nhead,dim_feedforward,dropout,batch_first):
        super().__init__(d_model,nhead,dim_feedforward,dropout,batch_first=batch_first)
    def _sa_block(self,x,attn_mask,key_padding_mask,pos_ids=None,rope=None):
        if rope is not None and pos_ids is not None: q=k=rope.rotate_queries_and_keys(x,pos_ids); return self.self_attn(q,k,x,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False)[0]
        return self.self_attn(x,x,x,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False)[0]
    def forward(self,src,src_mask=None,src_key_padding_mask=None,pos_ids=None,rope=None):
        x=src;x=x+self.dropout1(self._sa_block(self.norm1(x),src_mask,src_key_padding_mask,pos_ids,rope));x=x+self.dropout2(self._ff_block(self.norm2(x)));return x

class UnifiedModel(nn.Module):
    def __init__(self, **model_config):
        super().__init__()
        self.grid_size = model_config['grid_size']
        d_model = model_config['d_model']
        dim_feedforward = d_model * model_config['dim_feedforward_multiplier']
        
        # Embeddings for the 2 input channels
        self.obs_embedding = nn.Embedding(len(model_config['vocab']), d_model)
        self.scratch_embedding = nn.Embedding(len(BFS_VOCAB), d_model)
        self.channel_embedding = nn.Embedding(2, d_model) # 0 for obs, 1 for scratch
        
        # Core Transformer
        self.rope = RotaryPositionalEmbedding2D(d_model, self.grid_size, model_config['rope_theta'])
        encoder_layer = RoPETransformerEncoderLayer(d_model, model_config['nhead'], dim_feedforward, model_config['dropout'], batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, model_config['num_layers'])
        
        # Heads for different tasks
        self.bfs_head = nn.Linear(d_model, len(BFS_VOCAB))
        self.actor_head = nn.Linear(d_model, model_config['num_actions'])
        self.critic_head = nn.Linear(d_model, 1)

    def _get_base_output(self, dual_channel_board):
        B, C, H, W = dual_channel_board.shape
        
        # Get embeddings for each channel
        obs_emb = self.obs_embedding(dual_channel_board[:, 0, :, :].view(B, -1))
        scratch_emb = self.scratch_embedding(dual_channel_board[:, 1, :, :].view(B, -1))
        
        # Add channel-specific embeddings
        obs_emb += self.channel_embedding(torch.zeros_like(dual_channel_board[:, 0, :, :].view(B, -1)))
        scratch_emb += self.channel_embedding(torch.ones_like(dual_channel_board[:, 1, :, :].view(B, -1)))
        
        # Combine embeddings and apply Transformer
        x = torch.cat([obs_emb, scratch_emb], dim=1) # Concatenate along sequence dim
        
        # Create position IDs for the combined 2*H*W sequence
        rows = torch.arange(H, device=device).view(1,H,1).expand(B,H,W)
        cols = torch.arange(W, device=device).view(1,1,W).expand(B,H,W)
        pos_ids_single = torch.stack([rows,cols],dim=-1).view(B,H*W,2)
        pos_ids = torch.cat([pos_ids_single, pos_ids_single], dim=1) # Repeat for both channels
        
        for layer in self.transformer.layers:
            x = layer(x, pos_ids=pos_ids, rope=self.rope)
        
        return x

    def forward(self, dual_channel_board, mode, action=None, deterministic=False):
        base_output = self._get_base_output(dual_channel_board)
        
        if mode == 'bfs_predict':
            # Predict the next state of the scratchpad
            scratchpad_tokens = base_output[:, self.grid_size*self.grid_size:]
            return self.bfs_head(scratchpad_tokens)

        elif mode == 'rl_act':
            # Use the combined representation to decide on an action
            # For RL, we can average the representation of both channels
            cls_rep = base_output.mean(dim=1)
            
            value = self.critic_head(cls_rep)
            logits = self.actor_head(cls_rep)
            dist = torch.distributions.Categorical(logits=logits)
            
            if action is None:
                action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            
            return action, dist.log_prob(action), dist.entropy(), value

# ============================================================================
# Phase 1: Train BFS Prediction
# ============================================================================
def train_bfs_prediction(agent, device, **kwargs):
    print("\n=== Starting Phase 1: Learning to Plan (BFS Prediction) ===\n")
    optimizer = optim.AdamW(agent.parameters(), lr=kwargs['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    
    # Use the config to create the env, removing the redundant grid_size
    env = VectorizedGridWorldEnv(kwargs['batch_size'], device, **CONFIG['env'], **CONFIG['shared'])
    
    # Generate all training data upfront
    all_transitions = []
    print("Generating BFS transition data...")
    # Heuristic for number of mazes to generate
    num_mazes_to_generate = kwargs['total_steps'] // (env.grid_size) 
    for _ in range(num_mazes_to_generate // kwargs['batch_size']):
        env.reset()
        all_transitions.extend(generate_bfs_transitions(env.grids, env.end_pos))
    print(f"Generated {len(all_transitions)} transition pairs.")

    num_steps = len(all_transitions) // kwargs['batch_size']
    for step in range(num_steps):
        batch_start = step * kwargs['batch_size']
        batch_end = batch_start + kwargs['batch_size']
        batch = all_transitions[batch_start:batch_end]
        if not batch: continue
        
        input_boards = torch.cat([t[0] for t in batch])
        target_boards = torch.cat([t[1] for t in batch])
        
        optimizer.zero_grad()
        logits = agent(input_boards, mode='bfs_predict')
        
        # The target for the loss is the scratchpad channel of the target boards
        target_scratchpad = target_boards[:, 1, :, :]
        
        # --- THIS IS THE FIX ---
        # Use .reshape() instead of .view() to handle non-contiguous tensors
        loss = loss_fn(logits.reshape(-1, len(BFS_VOCAB)), target_scratchpad.reshape(-1))
        # --- END OF FIX ---
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"BFS Training Step {(step + 1) * kwargs['batch_size']}/{len(all_transitions)}, Loss: {loss.item():.4f}")

    print("\nBFS Prediction training complete!\n")

# ============================================================================
# Phase 2: Train RL Navigation
# ============================================================================
def train_rl_navigation(agent, env, device, **kwargs):
    print("\n=== Starting Phase 2: Learning to Act (RL Navigation) ===\n")
    optimizer = optim.AdamW(agent.parameters(), lr=kwargs['learning_rate'], eps=kwargs['optimizer_eps'])
    num_updates = kwargs['total_timesteps'] // (kwargs['num_envs'] * kwargs['steps_per_update'])
    grid_size = env.grid_size

    # (RL buffers and GAE are the same as before)
    
    next_state = env.reset()
    for update in range(1, num_updates + 1):
        # ... (Rollout collection with thinking loop)
        # ... (GAE computation)
        # ... (GRPO update loop)
        pass # Placeholder for the full RL loop
        if update % kwargs['log_interval'] == 0:
            print(f"RL Update {update}/{num_updates}...")

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    agent = UnifiedModel(**CONFIG['model'], **CONFIG['shared']).to(device)

    # --- PHASE 1 ---
    train_bfs_prediction(agent, device, **CONFIG['bfs_prediction'])
    
    # --- PHASE 2 ---
    # The RL part is conceptually complex to integrate here. The above structure
    # shows how the model is prepared. The RL loop would need to be
    # fully written out, including the "thinking" steps.
    print("\nNOTE: RL training loop is a placeholder. The model is now pre-trained to perform BFS.")