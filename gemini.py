import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import numpy as np
import math
import time

# --- 1. Configuration & Task Selection ---
TASK = 'ALIGNMENT' # Options: 'ADDITION', 'ALIGNMENT'

# --- DDP Setup ---
def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), local_rank

# --- 2. Data Generation ---
def generate_addition_transitions(num_digits, grid_h, grid_w):
    a = random.randint(10**(num_digits-1), 10**num_digits - 1); b = random.randint(10**(num_digits-1), 10**num_digits - 1); c = a + b
    board = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    problem_str = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    lines = problem_str.split('\n')
    for r, line in enumerate(lines):
        for col, char in enumerate(line): board[r+1][grid_w - len(line) + col] = char
    state_transitions = []
    carry = 0
    for i in range(num_digits):
        col_idx = grid_w - 1 - i
        digit_a = int(board[1][col_idx]); digit_b = int(board[2][col_idx])
        current_sum = digit_a + digit_b + carry
        result_digit = current_sum % 10; new_carry = current_sum // 10
        prev_board_state = [row[:] for row in board]; board[3][col_idx] = str(result_digit)
        state_transitions.append((prev_board_state, [row[:] for row in board]))
        if new_carry > 0:
            prev_board_state = [row[:] for row in board]; board[0][col_idx - 1] = str(new_carry)
            state_transitions.append((prev_board_state, [row[:] for row in board]))
        carry = new_carry
    if len(str(c)) > num_digits:
        col_idx = grid_w - 1 - num_digits
        prev_board_state = [row[:] for row in board]; board[3][col_idx] = str(c)[0]
        state_transitions.append((prev_board_state, [row[:] for row in board]))
    return state_transitions

def generate_alignment_transitions(seq_len, grid_h, grid_w):
    MATCH_COST = 0; MISMATCH_COST = 1; GAP_COST = 1
    seq1 = ''.join(random.choices("ACGT", k=seq_len)); seq2 = ''.join(random.choices("ACGT", k=seq_len))
    dp_table = np.zeros((len(seq2) + 1, len(seq1) + 1), dtype=int)
    for i in range(len(seq2) + 1): dp_table[i, 0] = i * GAP_COST
    for j in range(len(seq1) + 1): dp_table[0, j] = j * GAP_COST
    for i in range(1, len(seq2) + 1):
        for j in range(1, len(seq1) + 1):
            cost = MATCH_COST if seq1[j-1] == seq2[i-1] else MISMATCH_COST
            dp_table[i, j] = min(dp_table[i-1][j-1] + cost, dp_table[i-1][j] + GAP_COST, dp_table[i][j-1] + GAP_COST)
    board = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    board_offset = 1
    for i, char in enumerate(seq1): board[0][i + board_offset + 1] = char
    for i, char in enumerate(seq2): board[i + board_offset][0] = char
    state_transitions = []
    for k in range(len(seq1) + len(seq2) + 1):
        prev_board = [row[:] for row in board]
        for i in range(k + 1):
            j = k - i
            if i <= len(seq2) and j <= len(seq1):
                board[i + board_offset][j + board_offset] = str(dp_table[i, j])
        state_transitions.append((prev_board, [row[:] for row in board]))
    return state_transitions

# --- 4. Model Architecture (Unchanged) ---
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

class BlackboardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, pe_type='ROPE'):
        super().__init__(); self.grid_h, self.grid_w = grid_size; self.pe_type = pe_type
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if self.pe_type == 'ROPE':
            self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
            encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        else:
            self.register_buffer('pos_embedding_2d_buffer', self._create_2d_sinusoidal_embedding(d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
    def _create_2d_sinusoidal_embedding(self, d_model):
        pos_embedding = torch.zeros(self.grid_h, self.grid_w, d_model); pos = torch.arange(max(self.grid_h, self.grid_w), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_row = torch.sin(pos[:self.grid_h] * div_term); pos_embedding[:, :, 0::2] = pe_row.unsqueeze(1).repeat(1, self.grid_w, 1)
        pe_col = torch.cos(pos[:self.grid_w] * div_term); pos_embedding[:, :, 1::2] = pe_col.unsqueeze(0).repeat(self.grid_h, 1, 1)
        return pos_embedding.flatten(0, 1)
    def forward(self, board):
        B, H, W = board.shape; flat_board = board.view(B, H * W); token_emb = self.token_embedding(flat_board)
        if self.pe_type == 'ROPE':
            rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W); cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
            pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2); x = token_emb
            for layer in self.transformer_encoder.layers: x = layer(x, pos_ids=pos_ids, rope=self.rope)
        else:
            pos_emb = self.pos_embedding_2d_buffer.unsqueeze(0).repeat(B, 1, 1); x = token_emb + pos_emb; x = self.transformer_encoder(x)
        return self.output_head(x)

# --- 5. Generic Training & Inference Functions ---
def get_protected_mask(grid_size, task_name, task_params, device):
    H, W = grid_size; mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    if task_name == 'ADDITION':
        num_digits = task_params['NUM_DIGITS']
        mask[1, :] = True
        mask[2, 0:W - (num_digits+1)] = True
        mask[2, W - (num_digits+1):] = True
        mask[3, W - (num_digits+1):] = True
    elif task_name == 'ALIGNMENT':
        seq_len = task_params['SEQ_LEN']; board_offset = 1
        mask[0, board_offset+1 : board_offset+1+seq_len] = True
        mask[board_offset : board_offset+1+seq_len+1, 0] = True
    return mask.view(1, H, W)

def create_bert_loss_target(input_boards, target_boards, protected_mask, vocab_size, device):
    corrupted_boards = input_boards.clone()
    loss_target = torch.full_like(input_boards, -100)
    primary_target_mask = (input_boards != target_boards)
    loss_target[primary_target_mask] = target_boards[primary_target_mask]
    
    # Candidates for MLM are all non-primary cells (including protected ones)
    candidate_mask = ~primary_target_mask
    candidate_indices = torch.where(candidate_mask)
    num_candidates = candidate_indices[0].size(0)
    num_to_select = int(num_candidates * 0.30)
    
    if num_to_select > 0:
        rand_indices = torch.randperm(num_candidates, device=device)[:num_to_select]
        selected_b, selected_h, selected_w = (candidate_indices[i][rand_indices] for i in range(3))
        
        mlm_mask = torch.zeros_like(input_boards, dtype=torch.bool)
        mlm_mask[selected_b, selected_h, selected_w] = True
        loss_target[mlm_mask] = target_boards[mlm_mask]

        # Of the selected MLM cells, find which ones are NOT protected and can be corrupted
        is_protected_mask = protected_mask.expand_as(input_boards)
        corruptible_mask = mlm_mask & ~is_protected_mask
        corruptible_indices = torch.where(corruptible_mask)
        
        # Corrupt 50% of the *corruptible* subset
        num_corruptible = corruptible_indices[0].size(0)
        num_to_corrupt = int(num_corruptible * 0.5)
        
        if num_to_corrupt > 0:
            rand_corrupt_perm = torch.randperm(num_corruptible, device=device)[:num_to_corrupt]
            corrupt_b = corruptible_indices[0][rand_corrupt_perm]
            corrupt_h = corruptible_indices[1][rand_corrupt_perm]
            corrupt_w = corruptible_indices[2][rand_corrupt_perm]
            
            random_tokens = torch.randint(0, vocab_size, (num_to_corrupt,), device=device)
            corrupted_boards[corrupt_b, corrupt_h, corrupt_w] = random_tokens
            
    return corrupted_boards, loss_target

def preprocess_data_on_gpu(all_transitions, stoi_map, rank, world_size, device):
    if rank == 0:
        if not all_transitions:
            size_tensor = torch.tensor([0,0,0], dtype=torch.long, device=device); dist.broadcast(size_tensor, src=0); return None, None
        print("Preprocessing data on rank 0...")
        input_list = [torch.tensor([[stoi_map.get(c, 0) for c in row] for row in in_b], dtype=torch.long) for in_b, _ in all_transitions]
        target_list = [torch.tensor([[stoi_map.get(c, 0) for c in row] for row in out_b], dtype=torch.long) for _, out_b in all_transitions]
        all_input_boards = torch.stack(input_list).to(device); all_target_boards = torch.stack(target_list).to(device)
        if rank == 0: print(f"Data generated on rank 0 and moved to {device}. Shape: {all_input_boards.shape}")
    
    size_tensor = torch.empty(3, dtype=torch.long, device=device)
    if rank == 0: size_tensor[0], size_tensor[1], size_tensor[2] = all_input_boards.shape
    dist.broadcast(size_tensor, src=0)
    if size_tensor[0] == 0: return None, None
    if rank != 0:
        all_input_boards = torch.empty(tuple(size_tensor.tolist()), dtype=torch.long, device=device)
        all_target_boards = torch.empty(tuple(size_tensor.tolist()), dtype=torch.long, device=device)
    dist.broadcast(all_input_boards, src=0); dist.broadcast(all_target_boards, src=0)
    if rank == 0: print("Data successfully broadcasted to all GPUs.")
    return all_input_boards, all_target_boards

def run_training_loop(model, all_input_boards, all_target_boards, epochs, lr, batch_size, vocab, rank, world_size, task_name, task_params):
    criterion = nn.CrossEntropyLoss(ignore_index=-100); optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(); num_samples = all_input_boards.size(0)
    protected_mask = get_protected_mask((all_input_boards.shape[1], all_input_boards.shape[2]), task_name, task_params, all_input_boards.device)
    if rank == 0: print("Starting DDP training with BERT-style MLM loss...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train(); total_loss = 0
        indices = torch.randperm(num_samples); rank_indices = indices[rank::world_size]
        for i in range(0, len(rank_indices), batch_size):
            batch_indices = rank_indices[i:i+batch_size]
            input_boards = all_input_boards[batch_indices]; target_boards = all_target_boards[batch_indices]
            corrupted_boards, loss_target = create_bert_loss_target(input_boards, target_boards, protected_mask, len(vocab), input_boards.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda'):
                logits = model(corrupted_boards)
                loss = criterion(logits.view(-1, len(vocab)), loss_target.view(-1))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / (len(rank_indices) / batch_size) if len(rank_indices) > 0 else 0
        if rank == 0: print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    dist.barrier()
    if rank == 0: print(f"Training finished in {time.time() - start_time:.2f} seconds.")

def infer_addition(model, num_digits, grid_size, stoi, itos, device):
    model.eval()
    try: a = random.randint(10**(num_digits-1), 10**num_digits-1); b = random.randint(10**(num_digits-1), 10**num_digits-1)
    except ValueError: print(f"Cannot generate a {num_digits}-digit number."); return
    problem_str = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    board = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    lines = problem_str.split('\n')
    for r, line in enumerate(lines):
        if r + 1 < grid_size[0] and grid_size[1] - len(line) >= 0:
            for c, char in enumerate(line): board[r+1][grid_size[1] - len(line) + c] = char
    print("--- Initial Problem ---"); print("\n".join("".join(row) for row in board))
    with torch.no_grad():
        for step in range(num_digits * 2 + 2):
            board_tensor = torch.tensor([[stoi.get(c, 0) for c in row] for row in board], dtype=torch.long).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type='cuda'): logits = model(board_tensor)
            predictions = logits.argmax(dim=-1).view(1, grid_size[0], grid_size[1])
            new_board = [[itos[p.item()] for p in row] for row in predictions[0]]
            if new_board == board: print("\nModel has converged."); break
            board = new_board; print(f"\n--- Step {step+1} ---"); print("\n".join("".join(row) for row in board))
    print("\n--- Final Board State ---"); print("\n".join("".join(row) for row in board)); print(f"Correct Answer: {a+b}")

def infer_alignment(model, seq_len, grid_size, stoi, itos, device):
    model.eval()
    seq1 = ''.join(random.choices("ACGT", k=seq_len)); seq2 = ''.join(random.choices("ACGT", k=seq_len))
    board = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    board_offset = 1
    for i, char in enumerate(seq1): board[0][i+board_offset+1] = char
    for i, char in enumerate(seq2): board[i+board_offset][0] = char
    print("--- Initial Problem ---"); print(f"Seq1: {seq1}\nSeq2: {seq2}"); print("\n".join("".join(row) for row in board))
    with torch.no_grad():
        for step in range((seq_len+1)**2 + 1):
            board_tensor = torch.tensor([[stoi.get(c, 0) for c in row] for row in board], dtype=torch.long).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type='cuda'): logits = model(board_tensor)
            predictions = logits.argmax(dim=-1).view(1, grid_size[0], grid_size[1])
            new_board = [[itos[p.item()] for p in row] for row in predictions[0]]
            if new_board == board: print("\nModel has converged."); break
            board = new_board; print(f"\n--- Step {step+1} ---"); print("\n".join("".join(row) for row in board))
    print("\n--- Final Board State ---"); print("\n".join("".join(row) for row in board))

# --- 6. Task-Specific Training Orchestration ---
def setup_and_train_addition(rank, world_size, local_rank, device):
    if rank == 0: print("--- CONFIGURING FOR ADDITION TASK ---")
    VOCAB = list("0123456789+- ="); stoi = {c: i for i, c in enumerate(VOCAB)}; itos = {i: c for i, c in enumerate(VOCAB)}
    NUM_DIGITS=3; GRID_SIZE = (5, 8); D_MODEL = 128; NHEAD = 8; NUM_LAYERS = 6
    BATCH_SIZE = 128; LEARNING_RATE = 1e-4; NUM_EPOCHS = 20; NUM_SAMPLES = 8000
    task_params = {'NUM_DIGITS': NUM_DIGITS}
    all_transitions = []
    if rank == 0:
        print("Generating training data on rank 0...")
        for _ in range(NUM_SAMPLES): all_transitions.extend(generate_addition_transitions(NUM_DIGITS, GRID_SIZE[0], GRID_SIZE[1]))
    all_input_gpu, all_target_gpu = preprocess_data_on_gpu(all_transitions, stoi, rank, world_size, device)
    if all_input_gpu is None: return
    model = BlackboardTransformer(len(VOCAB), D_MODEL, NHEAD, NUM_LAYERS, GRID_SIZE, pe_type='ROPE').to(device)
    model = DDP(model, device_ids=[local_rank])
    run_training_loop(model, all_input_gpu, all_target_gpu, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, VOCAB, rank, world_size, 'ADDITION', task_params)
    if rank == 0:
        print("\n--- Running Addition Inference ---"); infer_addition(model.module, NUM_DIGITS, GRID_SIZE, stoi, itos, device)
        print("\n--- Running Addition Generalization ---"); infer_addition(model.module, NUM_DIGITS + 1, GRID_SIZE, stoi, itos, device)

def setup_and_train_alignment(rank, world_size, local_rank, device):
    if rank == 0: print("--- CONFIGURING FOR ALIGNMENT TASK ---")
    VOCAB = list("ACGT0123456789-=") + [' ']; stoi = {c: i for i, c in enumerate(VOCAB)}; itos = {i: c for i, c in enumerate(VOCAB)}
    SEQ_LEN = 8; GRID_SIZE = (SEQ_LEN + 3, SEQ_LEN + 3); D_MODEL = 128; NHEAD = 8; NUM_LAYERS = 6
    BATCH_SIZE = 512; LEARNING_RATE = 1e-3; NUM_EPOCHS = 30; NUM_SAMPLES = 20000
    task_params = {'SEQ_LEN': SEQ_LEN}
    all_transitions = []
    if rank == 0:
        print("Generating training data on rank 0...")
        for _ in range(NUM_SAMPLES): all_transitions.extend(generate_alignment_transitions(SEQ_LEN, GRID_SIZE[0], GRID_SIZE[1]))
    all_input_gpu, all_target_gpu = preprocess_data_on_gpu(all_transitions, stoi, rank, world_size, device)
    if all_input_gpu is None: return
    model = BlackboardTransformer(len(VOCAB), D_MODEL, NHEAD, NUM_LAYERS, GRID_SIZE, pe_type='ROPE').to(device)
    model = DDP(model, device_ids=[local_rank])
    run_training_loop(model, all_input_gpu, all_target_gpu, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, VOCAB, rank, world_size, 'ALIGNMENT', task_params)
    if rank == 0:
        print("\n--- Running Alignment Inference ---"); infer_alignment(model.module, SEQ_LEN, GRID_SIZE, stoi, itos, device)

def main():
    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}'
    if TASK == 'ADDITION':
        setup_and_train_addition(rank, world_size, local_rank, device)
    elif TASK == 'ALIGNMENT':
        setup_and_train_alignment(rank, world_size, local_rank, device)
    else:
        raise ValueError(f"Unknown TASK specified: {TASK}")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()