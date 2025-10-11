import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import time
from torch.amp import autocast, GradScaler


# ============================================================================
# Hierarchical Configuration
# ============================================================================
CONFIG = {
    'shared': {
        # MODIFICATION: Added 1 for the IDLE action
        'grid_size': 8, 'vocab': [' ', '#', 'S', 'E', 'A'], 'num_actions': 5,
    },
    'env': {
        'wall_density': 1.2, 'max_episode_steps_multiplier': 5,
        'reward_step': -0.01, 'reward_collision': -0.1, 'reward_goal': 1.0,
    },
    'model': {
        'd_model': 128, 'nhead': 4, 'num_layers': 6,
        'dim_feedforward_multiplier': 2, 'dropout': 0.1, 'rope_theta': 10000.0,
    },
    'bfs_prediction': {
        'total_steps': 500_000, 'batch_size': 48, 'learning_rate': 1e-3,
        'val_interval': 100,
    },
    'rl_navigation': {
        'total_timesteps': 400_000, 'num_envs': 64, 'steps_per_update': 64,
        'learning_rate': 1e-4, 'optimizer_eps': 1e-8, 'grad_clip_norm': 0.5,
        'epochs_per_update': 4, 'minibatch_size': 256,
        # MODIFICATION: thinking_steps is no longer needed as it's learned
    },
    'grpo': {
        'gamma': 0.99, 'clip_eps': 0.2, 'entropy_coef': 0.01,
    },
    'logging': {
        'log_interval':10, 'eval_interval': 10, 'eval_episodes': 256,
    }
}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
print(f"Using device: {device} | Mixed Precision (FP16) Enabled: {use_amp}\n")

# Vocabs & Mappings
BFS_VOCAB = [str(i) for i in range(100)] + ['INF', 'UKN']
bfs_stoi = {c: i for i, c in enumerate(BFS_VOCAB)}; bfs_itos = {i: c for i, c in enumerate(BFS_VOCAB)}
def get_char_mappings(vocab): return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}
stoi, itos = get_char_mappings(CONFIG['shared']['vocab'])

# ============================================================================
# Environment
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
        self.episode_steps+=1
        # MODIFICATION: Handle IDLE action (action index 4). Agent position does not change.
        IDLE_ACTION = CONFIG['shared']['num_actions'] - 1
        is_move = actions != IDLE_ACTION
        
        # Start with current positions
        next_pos = self.agent_pos.clone()
        
        # Calculate next positions only for agents that are moving
        move_indices = torch.where(is_move)[0]
        if len(move_indices) > 0:
            move_actions = actions[move_indices]
            temp_pos = self.agent_pos[move_indices].clone()
            temp_pos[move_actions == 0, 0] -= 1; temp_pos[move_actions == 1, 0] += 1
            temp_pos[move_actions == 2, 1] -= 1; temp_pos[move_actions == 3, 1] += 1
            next_pos[move_indices] = temp_pos

        # Collision checks are only relevant for moving agents
        out_of_bounds = ((next_pos < 0) | (next_pos >= self.grid_size)).any(dim=1)
        wall_collision = torch.zeros_like(out_of_bounds)
        valid_pos_check_indices = torch.where(~out_of_bounds & is_move)[0] # Only check non-OOB movers
        if len(valid_pos_check_indices) > 0:
            valid_pos = next_pos[valid_pos_check_indices]
            wall_collision[valid_pos_check_indices] = self.grids[valid_pos_check_indices, valid_pos[:, 0], valid_pos[:, 1]] == stoi['#']
        
        collided = (out_of_bounds | wall_collision) & is_move
        valid_moves = ~collided
        
        self.agent_pos[valid_moves] = next_pos[valid_moves]
        
        rewards = torch.full((self.num_envs,), self.reward_step, device=self.device, dtype=torch.float)
        rewards[collided] = self.reward_collision
        goal_reached = (self.agent_pos == self.end_pos).all(dim=1)
        rewards[goal_reached] = self.reward_goal
        timeout = self.episode_steps >= self.max_steps; dones = goal_reached | timeout
        if dones.any(): self._reset_envs(dones)
        return self.get_state(),rewards,dones,goal_reached
    def _reset_envs(self,mask):
        num_reset=mask.sum().item()
        if num_reset==0: return
        self.episode_steps[mask]=0; self.grids[mask]=stoi[' ']; num_walls=int(self.grid_size*self.wall_density)
        for _ in range(num_walls): coords=torch.randint(0,self.grid_size,(num_reset,2),device=self.device); self.grids[mask,coords[:,0],coords[:,1]]=stoi['#']
        self.agent_pos[mask]=torch.randint(0,self.grid_size,(num_reset,2),device=self.device); self.start_pos[mask]=self.agent_pos[mask].clone(); self.end_pos[mask]=torch.randint(0,self.grid_size,(num_reset,2),device=self.device)
        env_idx=torch.arange(self.num_envs,device=self.device)[mask]; self.grids[env_idx,self.start_pos[mask,0],self.start_pos[mask,1]]=stoi['S']; self.grids[env_idx,self.end_pos[mask,0],self.end_pos[mask,1]]=stoi['E']

# (Other helper functions like generate_bfs_transitions remain unchanged)
def generate_bfs_transitions(grids, goals):
    num_envs, grid_size, _ = grids.shape; transitions = []
    obs_channel = grids.clone(); scratch_channel = torch.full_like(obs_channel, bfs_stoi['UKN'])
    env_idx = torch.arange(num_envs, device=device); scratch_channel[env_idx, goals[:, 0], goals[:, 1]] = bfs_stoi['0']
    scratch_channel[obs_channel == stoi['#']] = bfs_stoi['INF']
    for k in range(grid_size * grid_size):
        prev_scratch = scratch_channel.clone(); input_board = torch.stack([obs_channel, prev_scratch], dim=1)
        frontier = (scratch_channel == bfs_stoi[str(k)])
        if not frontier.any(): break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_frontier = torch.roll(frontier, shifts=(-dr, -dc), dims=(1, 2))
            updatable = (scratch_channel == bfs_stoi['UKN']) & next_frontier
            scratch_channel[updatable] = bfs_stoi[str(k + 1)]
        target_board = torch.stack([obs_channel, scratch_channel.clone()], dim=1)
        if not torch.equal(input_board, target_board): transitions.append((input_board, target_board))
    return transitions
# ============================================================================
# Model Architecture
# ============================================================================
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
        super().__init__(); self.grid_size=model_config['grid_size']; d_model=model_config['d_model']; dim_feedforward=d_model*model_config['dim_feedforward_multiplier']
        self.obs_embedding=nn.Embedding(len(model_config['vocab']),d_model); self.scratch_embedding=nn.Embedding(len(BFS_VOCAB),d_model); self.channel_embedding=nn.Embedding(2,d_model)
        self.rope=RotaryPositionalEmbedding2D(d_model,self.grid_size,model_config['rope_theta']); encoder_layer=RoPETransformerEncoderLayer(d_model,model_config['nhead'],dim_feedforward,model_config['dropout'],batch_first=True); self.transformer=nn.TransformerEncoder(encoder_layer,model_config['num_layers'])
        self.bfs_head=nn.Linear(d_model,len(BFS_VOCAB))
        # MODIFICATION: Renamed actor_head to nav_head for clarity and adjusted output size for IDLE action.
        self.nav_head=nn.Linear(d_model,model_config['num_actions'])
    def _get_base_output(self,dual_channel_board):
        B,C,H,W=dual_channel_board.shape
        obs_flat=dual_channel_board[:,0,:,:].reshape(B,-1); scratch_flat=dual_channel_board[:,1,:,:].reshape(B,-1)
        obs_emb=self.obs_embedding(obs_flat)+self.channel_embedding(torch.zeros_like(obs_flat)); scratch_emb=self.scratch_embedding(scratch_flat)+self.channel_embedding(torch.ones_like(scratch_flat))
        x=torch.cat([obs_emb,scratch_emb],dim=1)
        rows=torch.arange(H,device=device).view(1,H,1).expand(B,H,W);cols=torch.arange(W,device=device).view(1,1,W).expand(B,H,W);pos_ids_single=torch.stack([rows,cols],dim=-1).view(B,H*W,2);pos_ids=torch.cat([pos_ids_single,pos_ids_single],dim=1)
        for layer in self.transformer.layers: x=layer(x,pos_ids=pos_ids,rope=self.rope)
        return x
    
    # MODIFICATION: The forward pass now handles three distinct modes.
    def forward(self, dual_channel_board, mode, nav_action=None, bfs_action=None, deterministic=False):
        base_output = self._get_base_output(dual_channel_board)
        
        # Mode 1: Pre-training the BFS head (unchanged)
        if mode == 'bfs_predict':
            scratchpad_tokens_out = base_output[:, self.grid_size*self.grid_size:]
            return self.bfs_head(scratchpad_tokens_out)

        # Mode 2: RL Rollout - Sample joint actions
        elif mode == 'rl_rollout':
            # Navigation Head
            cls_rep = base_output.mean(dim=1)
            nav_logits = self.nav_head(cls_rep)
            nav_dist = torch.distributions.Categorical(logits=nav_logits)
            sampled_nav_action = torch.argmax(nav_logits, dim=-1) if deterministic else nav_dist.sample()
            
            # BFS Head
            scratchpad_tokens_out = base_output[:, self.grid_size*self.grid_size:]
            bfs_logits = self.bfs_head(scratchpad_tokens_out)
            bfs_dist = torch.distributions.Categorical(logits=bfs_logits)
            sampled_bfs_action = bfs_dist.sample() # Shape: (B, H*W)

            return sampled_nav_action, sampled_bfs_action, nav_dist, bfs_dist

        # Mode 3: RL Update - Calculate log_probs and entropy for stored actions
        elif mode == 'rl_update':
            # Navigation Head
            cls_rep = base_output.mean(dim=1)
            nav_logits = self.nav_head(cls_rep)
            nav_dist = torch.distributions.Categorical(logits=nav_logits)
            nav_log_prob = nav_dist.log_prob(nav_action)
            nav_entropy = nav_dist.entropy()

            # BFS Head
            scratchpad_tokens_out = base_output[:, self.grid_size*self.grid_size:]
            bfs_logits = self.bfs_head(scratchpad_tokens_out)
            bfs_dist = torch.distributions.Categorical(logits=bfs_logits)
            bfs_log_prob = bfs_dist.log_prob(bfs_action).sum(dim=-1) # Sum log_probs across all grid cells
            bfs_entropy = bfs_dist.entropy().sum(dim=-1) # Sum entropy across all grid cells
            
            return nav_log_prob + bfs_log_prob, nav_entropy + bfs_entropy

# ============================================================================
# Training & Evaluation
# ============================================================================
# MODIFICATION: Evaluation logic is updated for the new dynamic, joint-action policy.
def evaluate_rl_navigation(agent, device, eval_episodes, **kwargs):
    print("  --- Running RL Validation ---")
    eval_env = VectorizedGridWorldEnv(eval_episodes, device, **CONFIG['env'], **CONFIG['shared'])
    obs_state = eval_env.reset()
    
    # Initialize scratchpad
    grid_size = eval_env.grid_size
    scratchpad = torch.full((eval_episodes, grid_size, grid_size), bfs_stoi['UKN'], device=device)
    scratchpad[torch.arange(eval_episodes), eval_env.end_pos[:, 0], eval_env.end_pos[:, 1]] = bfs_stoi['0']
    scratchpad[eval_env.grids == stoi['#']] = bfs_stoi['INF']

    active = torch.ones(eval_episodes, dtype=torch.bool, device=device)
    ep_lengths = torch.zeros(eval_episodes, device=device)
    successes, total_len, total_shortest = 0, 0, 0
    start_pos, end_pos = eval_env.start_pos.clone(), eval_env.end_pos.clone()

    for _ in range(eval_env.max_steps):
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=use_amp):
                current_board_state = torch.stack([obs_state, scratchpad], dim=1)
                nav_actions, bfs_actions, _, _ = agent(current_board_state, mode='rl_rollout', deterministic=True)
                scratchpad = bfs_actions.view(eval_episodes, grid_size, grid_size)
        
        obs_state, _, dones, goal_reached = eval_env.step(nav_actions)
        ep_lengths[active] += 1
        
        # For any env that just finished, reset its scratchpad
        if dones.any():
            scratchpad[dones] = torch.full((1, grid_size, grid_size), bfs_stoi['UKN'], device=device)
            scratchpad[dones, eval_env.end_pos[dones, 0], eval_env.end_pos[dones, 1]] = bfs_stoi['0']
            scratchpad[dones] = torch.where(eval_env.grids[dones] == stoi['#'], bfs_stoi['INF'], scratchpad[dones])

        finished = torch.where(goal_reached & active)[0]
        if len(finished) > 0:
            successes += len(finished)
            for i in finished:
                total_len += ep_lengths[i].item()
                total_shortest += torch.abs(start_pos[i] - end_pos[i]).sum().item()
            active[finished] = False
        if not active.any(): break
            
    return {'success_rate': (successes/eval_episodes)*100, 'avg_len': total_len/successes if successes > 0 else float('nan'), 'efficiency': total_len/max(1, total_shortest) if successes > 0 else float('nan')}

# MODIFICATION: The main RL training loop is overhauled to handle the joint action space.
def train_rl_navigation(agent, env, device, val_transitions, **rl_config):
    print("\n=== Starting Phase 2: Learning to Act & Plan ===\n")
    optimizer = optim.AdamW(agent.parameters(), lr=rl_config['learning_rate'], eps=rl_config['optimizer_eps'])
    num_updates = rl_config['total_timesteps'] // (rl_config['num_envs'] * rl_config['steps_per_update'])
    grid_size = env.grid_size; grpo_params = CONFIG['grpo']
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # Buffer setup
    steps_per_update, num_envs = rl_config['steps_per_update'], rl_config['num_envs']
    states_buffer = torch.zeros((steps_per_update, num_envs, 2, grid_size, grid_size), dtype=torch.long, device=device)
    nav_actions_buffer = torch.zeros((steps_per_update, num_envs), dtype=torch.long, device=device)
    bfs_actions_buffer = torch.zeros((steps_per_update, num_envs, grid_size * grid_size), dtype=torch.long, device=device)
    log_probs_buffer = torch.zeros((steps_per_update, num_envs), dtype=torch.float, device=device)
    rewards_buffer = torch.zeros((steps_per_update, num_envs), dtype=torch.float, device=device)
    terminals_buffer = torch.zeros((steps_per_update, num_envs), dtype=torch.bool, device=device)

    # Initial state
    obs_state = env.reset()
    scratchpad_state = torch.full((num_envs, grid_size, grid_size), bfs_stoi['UKN'], device=device)
    scratchpad_state[torch.arange(num_envs), env.end_pos[:, 0], env.end_pos[:, 1]] = bfs_stoi['0']
    scratchpad_state[env.grids == stoi['#']] = bfs_stoi['INF']

    for update in range(1, num_updates + 1):
        for step in range(steps_per_update):
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=use_amp):
                    current_board_state = torch.stack([obs_state, scratchpad_state], dim=1)
                    
                    # Sample joint action from the policy
                    nav_action, bfs_action, nav_dist, bfs_dist = agent(current_board_state, mode='rl_rollout')
                    
                    # Calculate joint log probability for the sampled actions
                    nav_log_prob = nav_dist.log_prob(nav_action)
                    bfs_log_prob = bfs_dist.log_prob(bfs_action).sum(dim=-1)
                    joint_log_prob = nav_log_prob + bfs_log_prob
            
            # Store rollout data
            states_buffer[step] = current_board_state
            nav_actions_buffer[step] = nav_action
            bfs_actions_buffer[step] = bfs_action
            log_probs_buffer[step] = joint_log_prob
            
            # Step environment and update states
            next_obs_state, reward, done, _ = env.step(nav_action)
            rewards_buffer[step] = reward
            terminals_buffer[step] = done
            
            obs_state = next_obs_state
            scratchpad_state = bfs_action.view(num_envs, grid_size, grid_size)
            
            # Reset scratchpad for environments that are done
            if done.any():
                scratchpad_state[done] = torch.full((1, grid_size, grid_size), bfs_stoi['UKN'], device=device)
                scratchpad_state[done, env.end_pos[done, 0], env.end_pos[done, 1]] = bfs_stoi['0']
                scratchpad_state[done] = torch.where(env.grids[done] == stoi['#'], bfs_stoi['INF'], scratchpad_state[done])

        # Advantage calculation (remains the same)
        with torch.no_grad():
            returns = torch.zeros_like(rewards_buffer)
            for t in reversed(range(steps_per_update)):
                next_return = returns[t+1] if t < steps_per_update - 1 else 0
                returns[t] = rewards_buffer[t] + grpo_params['gamma'] * next_return * (1.0 - terminals_buffer[t].float())
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Prepare batches for update
        b_size = num_envs * steps_per_update
        b_states = states_buffer.reshape(b_size, 2, grid_size, grid_size)
        b_nav_actions = nav_actions_buffer.reshape(b_size)
        b_bfs_actions = bfs_actions_buffer.reshape(b_size, grid_size * grid_size)
        b_log_probs = log_probs_buffer.reshape(b_size)
        b_advantages = advantages.reshape(b_size)
        b_inds = np.arange(b_size)
        pg_losses, ent_losses = [], []
        
        # GPRO Update Loop
        for _ in range(rl_config['epochs_per_update']):
            np.random.shuffle(b_inds)
            for start in range(0, b_size, rl_config['minibatch_size']):
                mb_inds = b_inds[start:start + rl_config['minibatch_size']]
                optimizer.zero_grad()
                with autocast(device_type='cuda', enabled=use_amp):
                    # Get new log_probs and entropy for the joint action
                    new_joint_log_prob, new_joint_entropy = agent(
                        b_states[mb_inds], mode='rl_update', 
                        nav_action=b_nav_actions[mb_inds], 
                        bfs_action=b_bfs_actions[mb_inds]
                    )
                    ratio = torch.exp(new_joint_log_prob - b_log_probs[mb_inds])
                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - grpo_params['clip_eps'], 1 + grpo_params['clip_eps'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    ent_loss = new_joint_entropy.mean()
                    loss = pg_loss - grpo_params['entropy_coef'] * ent_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), rl_config['grad_clip_norm'])
                scaler.step(optimizer)
                scaler.update()
                pg_losses.append(pg_loss.item()); ent_losses.append(ent_loss.item())

        if update % CONFIG['logging']['log_interval'] == 0:
            print(f"RL Update {update}/{num_updates}")
            print(f"  Training -> Policy Loss: {np.mean(pg_losses):.4f} | Entropy: {np.mean(ent_losses):.4f}")
            if update % CONFIG['logging']['eval_interval'] == 0:
                rl_metrics = evaluate_rl_navigation(agent, device, **CONFIG['logging'])
                print(f"  RL Validation -> Success: {rl_metrics['success_rate']:.1f}% | Length: {rl_metrics['avg_len']:.1f} | Efficiency: {rl_metrics['efficiency']:.2f}")
                # We can still validate the BFS head's accuracy against the ground truth pre-training data
                bfs_metrics = validate_bfs_prediction(agent, val_transitions, CONFIG['bfs_prediction']['batch_size'], device)
                print(f"  BFS Head Validation -> Accuracy: {bfs_metrics['accuracy']:.2f}%")
            print()
    print("RL Training complete!")

# (The `train_bfs_prediction` and `validate_bfs_prediction` functions remain unchanged)
def validate_bfs_prediction(agent, validation_data, batch_size, device):
    agent.eval(); total_loss=0; correct_cells=0; total_cells=0; loss_fn=nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0,len(validation_data),batch_size):
            batch=validation_data[i:i+batch_size];
            if not batch: continue
            input_boards=torch.cat([t[0] for t in batch]); target_boards=torch.cat([t[1] for t in batch])
            with autocast(device_type='cuda', enabled=use_amp):
                logits=agent(input_boards,mode='bfs_predict'); target_scratchpad=target_boards[:,1,:,:]
                loss=loss_fn(logits.reshape(-1,len(BFS_VOCAB)),target_scratchpad.reshape(-1)); total_loss+=loss.item()
            preds=logits.argmax(dim=-1); target_flat=target_scratchpad.reshape(-1); preds_flat=preds.reshape(-1)
            correct_cells+=(preds_flat==target_flat).sum().item(); total_cells+=target_flat.numel()
    agent.train(); avg_loss=total_loss/max(1,len(validation_data)/batch_size); accuracy=(correct_cells/total_cells)*100
    return {'loss':avg_loss,'accuracy':accuracy}

def train_bfs_prediction(agent, device, **kwargs):
    print("\n=== Starting Phase 1: Pre-training the Planner (BFS Prediction) ===\n"); optimizer=optim.AdamW(agent.parameters(),lr=kwargs['learning_rate']); loss_fn=nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=use_amp)
    env=VectorizedGridWorldEnv(kwargs['batch_size'],device,**CONFIG['env'],**CONFIG['shared']); all_transitions=[]; print("Generating BFS transition data...")
    num_mazes_to_generate=kwargs['total_steps']//env.grid_size
    for _ in range(num_mazes_to_generate//kwargs['batch_size']): env.reset();all_transitions.extend(generate_bfs_transitions(env.grids,env.end_pos))
    np.random.shuffle(all_transitions); split_idx=int(len(all_transitions)*0.9); train_transitions=all_transitions[:split_idx]; val_transitions=all_transitions[split_idx:]
    print(f"Generated {len(all_transitions)} total transitions. Training on {len(train_transitions)}, Validating on {len(val_transitions)}.")
    num_steps=len(train_transitions)//kwargs['batch_size']
    for step in range(num_steps):
        batch_start=step*kwargs['batch_size']; batch_end=batch_start+kwargs['batch_size']; batch=train_transitions[batch_start:batch_end]
        if not batch: continue
        input_boards=torch.cat([t[0] for t in batch]);target_boards=torch.cat([t[1] for t in batch])
        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=use_amp):
            logits=agent(input_boards,mode='bfs_predict'); target_scratchpad=target_boards[:,1,:,:]
            loss=loss_fn(logits.reshape(-1,len(BFS_VOCAB)),target_scratchpad.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if(step+1)%kwargs['val_interval']==0:
            val_metrics=validate_bfs_prediction(agent,val_transitions,kwargs['batch_size'],device)
            print(f"BFS Pre-training Step {(step+1)*kwargs['batch_size']}/{len(train_transitions)}, Train Loss: {loss.item():.4f}")
            print(f"  Validation -> Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%\n")
    print("\nBFS Pre-training complete!\n"); return val_transitions


# (Main execution and inference functions are updated to reflect the new API)
def infer(agent,device,**kwargs):
    print("\n=== Inference ==="); infer_env = VectorizedGridWorldEnv(1, device, **CONFIG['env'], **CONFIG['shared'])
    obs_state = infer_env.reset(); grid_size=infer_env.grid_size
    scratchpad = torch.full((1, grid_size, grid_size), bfs_stoi['UKN'], device=device)
    scratchpad[0, infer_env.end_pos[0,0], infer_env.end_pos[0,1]] = bfs_stoi['0']
    scratchpad[infer_env.grids == stoi['#']] = bfs_stoi['INF']
    
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "IDLE (THINK)"}

    for step in range(infer_env.max_steps):
        print(f"\n--- Step {step+1} ---")
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=use_amp):
                current_board_state = torch.stack([obs_state, scratchpad], dim=1)
                nav_action, bfs_action, _, _ = agent(current_board_state, mode='rl_rollout', deterministic=True)
                scratchpad = bfs_action.view(1, grid_size, grid_size)
        
        print(f"Agent action: {action_map[nav_action.item()]}")
        obs_state, reward, done, goal_reached = infer_env.step(nav_action)
        grid_vis = obs_state.cpu().numpy()[0]; print("\n".join("".join(itos[c] for c in row) for row in grid_vis))
        if goal_reached.any(): print(f"\n✓ Success in {step+1} steps!"); return
        if done.any(): print(f"\n✗ Timeout after {step+1} steps"); return
    print(f"\n✗ Failed to reach goal")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    agent = UnifiedModel(**CONFIG['model'], **CONFIG['shared']).to(device)
    val_transitions = train_bfs_prediction(agent, device, **CONFIG['bfs_prediction'])
    rl_env = VectorizedGridWorldEnv(CONFIG['rl_navigation']['num_envs'], device, **CONFIG['env'], **CONFIG['shared'])
    train_rl_navigation(agent, rl_env, device, val_transitions, **CONFIG['rl_navigation'])
    infer(agent, device, **CONFIG['rl_navigation'])