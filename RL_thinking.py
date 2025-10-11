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
        'grid_size': 8, 'vocab': [' ', '#', 'S', 'E', 'A'], 'num_actions': 4,
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
        'total_steps': 500_000, 'batch_size': 32, 'learning_rate': 1e-3,
        'val_interval': 100,
    },
    'rl_navigation': {
        'total_timesteps': 400_000, 'num_envs': 64, 'steps_per_update': 64,
        'learning_rate': 1e-4, 'optimizer_eps': 1e-8, 'grad_clip_norm': 0.5,
        'epochs_per_update': 4, 'minibatch_size': 256, 'thinking_steps': 8,
    },
    'grpo': {
        'gamma': 0.99, 'gae_lambda': 0.95, 'grpo_beta': 0.1,
        'value_coef': 0.5, 'entropy_coef': 0.01,
    },
    'logging': {
        'log_interval': 10, 'eval_interval': 10, 'eval_episodes': 256,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Vocabs
BFS_VOCAB = [str(i) for i in range(100)] + ['INF', 'UKN']
bfs_stoi = {c: i for i, c in enumerate(BFS_VOCAB)}; bfs_itos = {i: c for i, c in enumerate(BFS_VOCAB)}
def get_char_mappings(vocab): return {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}
stoi, itos = get_char_mappings(CONFIG['shared']['vocab'])

# ============================================================================
# Data Generation & Environment
# ============================================================================
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
        self.bfs_head=nn.Linear(d_model,len(BFS_VOCAB)); self.actor_head=nn.Linear(d_model,model_config['num_actions']); self.critic_head=nn.Linear(d_model,1)
    
    def _get_base_output(self,dual_channel_board):
        B,C,H,W=dual_channel_board.shape
        obs_flat = dual_channel_board[:, 0, :, :].view(B, -1)
        scratch_flat = dual_channel_board[:, 1, :, :].view(B, -1)
        obs_emb = self.obs_embedding(obs_flat) + self.channel_embedding(torch.zeros_like(obs_flat))
        scratch_emb = self.scratch_embedding(scratch_flat) + self.channel_embedding(torch.ones_like(scratch_flat))
        x=torch.cat([obs_emb,scratch_emb],dim=1)
        rows=torch.arange(H,device=device).view(1,H,1).expand(B,H,W); cols=torch.arange(W,device=device).view(1,1,W).expand(B,H,W); pos_ids_single=torch.stack([rows,cols],dim=-1).view(B,H*W,2); pos_ids=torch.cat([pos_ids_single,pos_ids_single],dim=1)
        for layer in self.transformer.layers: x=layer(x,pos_ids=pos_ids,rope=self.rope)
        return x

    def forward(self,dual_channel_board,mode,action=None,deterministic=False):
        base_output=self._get_base_output(dual_channel_board)
        if mode=='bfs_predict':
            scratchpad_tokens_out = base_output[:,self.grid_size*self.grid_size:]
            return self.bfs_head(scratchpad_tokens_out)
        elif mode=='rl_act':
            cls_rep=base_output.mean(dim=1); value=self.critic_head(cls_rep); logits=self.actor_head(cls_rep); dist=torch.distributions.Categorical(logits=logits)
            if action is None: action=torch.argmax(logits,dim=-1) if deterministic else dist.sample()
            return action,dist.log_prob(action),dist.entropy(),value

# ============================================================================
# Training & Evaluation
# ============================================================================
def validate_bfs_prediction(agent, validation_data, batch_size, device):
    agent.eval(); total_loss=0; correct_cells=0; total_cells=0; loss_fn=nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0,len(validation_data),batch_size):
            batch=validation_data[i:i+batch_size]
            if not batch: continue
            input_boards=torch.cat([t[0] for t in batch]); target_boards=torch.cat([t[1] for t in batch])
            logits=agent(input_boards,mode='bfs_predict'); target_scratchpad=target_boards[:,1,:,:]
            loss=loss_fn(logits.reshape(-1,len(BFS_VOCAB)),target_scratchpad.reshape(-1)); total_loss+=loss.item()
            preds=logits.argmax(dim=-1); target_flat=target_scratchpad.reshape(-1); preds_flat=preds.reshape(-1)
            correct_cells+=(preds_flat==target_flat).sum().item(); total_cells+=target_flat.numel()
    agent.train(); avg_loss=total_loss/(len(validation_data)/batch_size); accuracy=(correct_cells/total_cells)*100
    return {'loss':avg_loss,'accuracy':accuracy}

def train_bfs_prediction(agent, device, **kwargs):
    print("\n=== Starting Phase 1: Learning to Plan (BFS Prediction) ===\n")
    optimizer=optim.AdamW(agent.parameters(),lr=kwargs['learning_rate']); loss_fn=nn.CrossEntropyLoss()
    env=VectorizedGridWorldEnv(kwargs['batch_size'],device,**CONFIG['env'],**CONFIG['shared'])
    all_transitions=[]; print("Generating BFS transition data...")
    num_mazes_to_generate=kwargs['total_steps']//env.grid_size
    for _ in range(num_mazes_to_generate//kwargs['batch_size']): env.reset();all_transitions.extend(generate_bfs_transitions(env.grids,env.end_pos))
    np.random.shuffle(all_transitions); split_idx=int(len(all_transitions)*0.9); train_transitions=all_transitions[:split_idx]; val_transitions=all_transitions[split_idx:]
    print(f"Generated {len(all_transitions)} total transitions. Training on {len(train_transitions)}, Validating on {len(val_transitions)}.")
    num_steps=len(train_transitions)//kwargs['batch_size']
    for step in range(num_steps):
        batch_start=step*kwargs['batch_size']; batch_end=batch_start+kwargs['batch_size']; batch=train_transitions[batch_start:batch_end]
        if not batch: continue
        input_boards=torch.cat([t[0] for t in batch]);target_boards=torch.cat([t[1] for t in batch])
        optimizer.zero_grad(); logits=agent(input_boards,mode='bfs_predict'); target_scratchpad=target_boards[:,1,:,:]
        loss=loss_fn(logits.reshape(-1,len(BFS_VOCAB)),target_scratchpad.reshape(-1)); loss.backward();optimizer.step()
        if(step+1)%kwargs['val_interval']==0:
            val_metrics=validate_bfs_prediction(agent,val_transitions,kwargs['batch_size'],device)
            print(f"BFS Training Step {(step+1)*kwargs['batch_size']}/{len(train_transitions)}, Train Loss: {loss.item():.4f}")
            print(f"  Validation -> Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%\n")
    print("\nBFS Prediction training complete!\n")

def compute_gae(rewards,values,terminals,next_value,gamma,gae_lambda):
    steps=len(rewards);advantages=torch.zeros_like(rewards);last_gae=0
    for t in reversed(range(steps)):
        if t==steps-1: next_non_terminal=1.0-terminals[-1].float(); next_val=next_value
        else: next_non_terminal=1.0-terminals[t+1].float(); next_val=values[t+1]
        delta=rewards[t]+gamma*next_val*next_non_terminal-values[t]; advantages[t]=last_gae=delta+gamma*gae_lambda*next_non_terminal*last_gae
    return advantages,advantages+values

def train_rl_navigation(agent, env, device, **rl_config):
    print("\n=== Starting Phase 2: Learning to Act (RL Navigation) ===\n")
    optimizer=optim.AdamW(agent.parameters(),lr=rl_config['learning_rate'],eps=rl_config['optimizer_eps'])
    num_updates=rl_config['total_timesteps']//(rl_config['num_envs']*rl_config['steps_per_update']); grid_size=env.grid_size; grpo_params=CONFIG['grpo']
    
    buffers=[torch.zeros((rl_config['steps_per_update'],rl_config['num_envs'],*s),dtype=d,device=device)for s,d in[((2,grid_size,grid_size),torch.long),((),torch.long),((),torch.float),((),torch.float),((),torch.float),((),torch.float)]]
    states,actions,log_probs,rewards,values,terminals=buffers
    
    obs_state=env.reset(); next_terminal=torch.zeros(rl_config['num_envs'],device=device)
    
    for update in range(1,num_updates+1):
        for step in range(rl_config['steps_per_update']):
            with torch.no_grad():
                scratchpad=torch.full((rl_config['num_envs'],grid_size,grid_size),bfs_stoi['UKN'],device=device)
                scratchpad[torch.arange(rl_config['num_envs']),env.end_pos[:,0],env.end_pos[:,1]]=bfs_stoi['0']
                scratchpad[env.grids==stoi['#']]=bfs_stoi['INF']
                
                current_board_state=torch.stack([obs_state,scratchpad],dim=1)
                for _ in range(rl_config['thinking_steps']):
                    logits=agent(current_board_state,mode='bfs_predict'); preds=logits.argmax(dim=-1)
                    current_board_state[:,1,:,:]=preds.view(rl_config['num_envs'],grid_size,grid_size)
                
                states[step]=current_board_state
                action,log_prob,_,value=agent(current_board_state,mode='rl_act'); values[step]=value.flatten()

            actions[step]=action;log_probs[step]=log_prob
            obs_state,reward,done,next_terminal=env.step(action);rewards[step]=reward;terminals[step]=next_terminal
        
        with torch.no_grad():
            final_scratchpad=torch.full((rl_config['num_envs'],grid_size,grid_size),bfs_stoi['UKN'],device=device)
            final_scratchpad[torch.arange(rl_config['num_envs']),env.end_pos[:,0],env.end_pos[:,1]]=bfs_stoi['0']
            final_scratchpad[env.grids==stoi['#']]=bfs_stoi['INF']
            final_board_state=torch.stack([obs_state,final_scratchpad],dim=1)
            for _ in range(rl_config['thinking_steps']):
                logits=agent(final_board_state,mode='bfs_predict');preds=logits.argmax(dim=-1)
                final_board_state[:,1,:,:]=preds.view(rl_config['num_envs'],grid_size,grid_size)
            next_value=agent(final_board_state,mode='rl_act')[3].reshape(1,-1)
            advantages,returns=compute_gae(rewards,values,terminals,next_value,grpo_params['gamma'],grpo_params['gae_lambda'])

        b_states=states.reshape(-1,2,grid_size,grid_size);b_actions=actions.reshape(-1);b_log_probs=log_probs.reshape(-1);b_advantages=advantages.reshape(-1);b_returns=returns.reshape(-1)
        b_inds=np.arange(rl_config['num_envs']*rl_config['steps_per_update']);pg_losses,v_losses,ent_losses=[],[],[]
        
        for _ in range(rl_config['epochs_per_update']):
            np.random.shuffle(b_inds)
            for start in range(0,len(b_inds),rl_config['minibatch_size']):
                mb_inds=b_inds[start:start+rl_config['minibatch_size']]
                _,new_log_prob,entropy,new_value=agent(b_states[mb_inds],mode='rl_act',action=b_actions[mb_inds])
                mb_adv=b_advantages[mb_inds];rel_adv=(mb_adv-mb_adv.mean())/(mb_adv.std()+1e-8)
                ratio=torch.exp(new_log_prob-b_log_probs[mb_inds]);pg_loss=-(rel_adv*ratio-grpo_params['grpo_beta']*(ratio-1)**2).mean()
                v_loss=0.5*((new_value.view(-1)-b_returns[mb_inds])**2).mean();ent_loss=entropy.mean()
                loss=pg_loss-grpo_params['entropy_coef']*ent_loss+grpo_params['value_coef']*v_loss
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(agent.parameters(),rl_config['grad_clip_norm']);optimizer.step()
                pg_losses.append(pg_loss.item());v_losses.append(v_loss.item());ent_losses.append(ent_loss.item())
        
        if update%CONFIG['logging']['log_interval']==0:
            print(f"RL Update {update}/{num_updates}");print(f"  Policy: {np.mean(pg_losses):.4f} | Value: {np.mean(v_losses):.4f} | Entropy: {np.mean(ent_losses):.4f}")
            if update%CONFIG['logging']['eval_interval']==0:print("  (RL Evaluation not yet implemented)")
            print()
    print("RL Training complete!")

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    agent = UnifiedModel(**CONFIG['model'], **CONFIG['shared']).to(device)
    train_bfs_prediction(agent, device, **CONFIG['bfs_prediction'])
    rl_env = VectorizedGridWorldEnv(CONFIG['rl_navigation']['num_envs'], device, **CONFIG['env'], **CONFIG['shared'])
    train_rl_navigation(agent, rl_env, device, **CONFIG['rl_navigation'])