import torch
import torch.nn as nn
import torch.optim as optim
# Removed DataLoader and Dataset imports
import random
import numpy as np
import math
import time

# --- 1. Configuration & Task Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK = 'ALIGNMENT' # Options: 'ADDITION', 'ALIGNMENT'

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
    for i in range(len(seq2) + 1): dp_table[i][0] = i
    for j in range(len(seq1) + 1): dp_table[0][j] = j
    for i in range(1, len(seq2) + 1):
        for j in range(1, len(seq1) + 1):
            cost = MATCH_COST if seq1[j-1] == seq2[i-1] else MISMATCH_COST
            dp_table[i, j] = min(dp_table[i-1][j-1] + cost, dp_table[i-1][j] + GAP_COST, dp_table[i][j-1] + GAP_COST)
    board = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    for i, char in enumerate(seq1): board[0][i+2] = char
    for i, char in enumerate(seq2): board[i+2][0] = char
    state_transitions = []
    for i in range(len(seq2) + 2):
        for j in range(len(seq1) + 2):
            if i == 0 or j == 0 or (i==1 and j==1): continue
            prev_board = [row[:] for row in board]; board[i][j] = str(dp_table[i-1, j-1])
            state_transitions.append((prev_board, [row[:] for row in board]))
    return state_transitions

# --- 3. Dataset (REMOVED) ---

# --- 4. Model Architecture (Remains Generic) ---

class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
    def _sa_block(self, x, attn_mask, key_padding_mask, pos_ids=None, rope=None):
        if rope is not None and pos_ids is not None:
            q = k = rope.rotate_queries_and_keys(x, pos_ids)
            return self.self_attn(q, k, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else: return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_ids=None, rope=None):
        x = src
        x = x + self.dropout1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, pos_ids, rope))
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, d_model, grid_size):
        super().__init__()
        self.d_row, self.d_col = d_model // 2, d_model - d_model // 2
        self.max_h, self.max_w = grid_size
        freqs_row = self._precompute_freqs(self.d_row, self.max_h)
        freqs_col = self._precompute_freqs(self.d_col, self.max_w)
        self.register_buffer('freqs_row_cos', freqs_row['cos']); self.register_buffer('freqs_row_sin', freqs_row['sin'])
        self.register_buffer('freqs_col_cos', freqs_col['cos']); self.register_buffer('freqs_col_sin', freqs_col['sin'])
    def _precompute_freqs(self, dim, max_pos, theta=10000.0):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1); return {'cos': emb.cos(), 'sin': emb.sin()}
    def _apply_rotary_emb(self, x, cos, sin):
        x2 = torch.cat([-x[..., 1::2], x[..., 0::2]], dim=-1); return x * cos + x2 * sin
    def rotate_queries_and_keys(self, x, pos_ids):
        x_row, x_col = x[..., :self.d_row], x[..., self.d_row:]
        row_ids, col_ids = pos_ids[..., 0], pos_ids[..., 1]
        cos_row, sin_row = self.freqs_row_cos[row_ids], self.freqs_row_sin[row_ids]
        cos_col, sin_col = self.freqs_col_cos[col_ids], self.freqs_col_sin[col_ids]
        return torch.cat([self._apply_rotary_emb(x_row, cos_row, sin_row), self._apply_rotary_emb(x_col, cos_col, sin_col)], dim=-1)

class BlackboardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size, pe_type='ROPE'):
        super().__init__()
        self.grid_h, self.grid_w = grid_size; self.pe_type = pe_type
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if self.pe_type == 'ROPE':
            self.rope = RotaryPositionalEmbedding2D(d_model, grid_size)
            encoder_layer = RoPETransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        else: # ABSOLUTE
            self.register_buffer('pos_embedding_2d_buffer', self._create_2d_sinusoidal_embedding(d_model))
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
    def _create_2d_sinusoidal_embedding(self, d_model):
        pos_embedding = torch.zeros(self.grid_h, self.grid_w, d_model)
        pos = torch.arange(max(self.grid_h, self.grid_w), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_row = torch.sin(pos[:self.grid_h] * div_term); pos_embedding[:, :, 0::2] = pe_row.unsqueeze(1).repeat(1, self.grid_w, 1)
        pe_col = torch.cos(pos[:self.grid_w] * div_term); pos_embedding[:, :, 1::2] = pe_col.unsqueeze(0).repeat(self.grid_h, 1, 1)
        return pos_embedding.flatten(0, 1)
    def forward(self, board):
        B, H, W = board.shape; flat_board = board.view(B, H * W)
        token_emb = self.token_embedding(flat_board)
        if self.pe_type == 'ROPE':
            rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W)
            cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
            pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2)
            x = token_emb
            for layer in self.transformer_encoder.layers: x = layer(x, pos_ids=pos_ids, rope=self.rope)
        else: # ABSOLUTE
            pos_emb = self.pos_embedding_2d_buffer.unsqueeze(0).repeat(B, 1, 1)
            x = token_emb + pos_emb
            x = self.transformer_encoder(x)
        return self.output_head(x)


# --- 5. Generic Training & Inference Functions ---

def preprocess_data_on_gpu(all_transitions, stoi_map, device):
    """Tokenizes the entire dataset and moves it to the GPU as two large tensors."""
    if not all_transitions:
        return None, None
    print("Preprocessing and moving all data to GPU...")
    # Tokenize all boards on CPU first using list comprehensions
    input_list = [torch.tensor([[stoi_map[c] for c in row] for row in in_b], dtype=torch.long) for in_b, _ in all_transitions]
    target_list = [torch.tensor([[stoi_map[c] for c in row] for row in out_b], dtype=torch.long) for _, out_b in all_transitions]
    # Stack into two large tensors
    all_input_boards = torch.stack(input_list)
    all_target_boards = torch.stack(target_list)
    # Move to the specified device
    all_input_boards = all_input_boards.to(device)
    all_target_boards = all_target_boards.to(device)
    print(f"Data moved to GPU. Tensor shapes: {all_input_boards.shape}")
    return all_input_boards, all_target_boards

def run_training_loop(model, all_input_boards, all_target_boards, epochs, lr, batch_size, vocab):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    num_samples = all_input_boards.size(0)
    
    print("Starting training with pre-loaded GPU data...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Shuffle indices at the start of each epoch
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Create a mini-batch by slicing the large tensors on the GPU
            input_boards = all_input_boards[batch_indices]
            target_boards = all_target_boards[batch_indices]
            
            optimizer.zero_grad()
            logits = model(input_boards)
            loss = criterion(logits.view(-1, len(vocab)), target_boards.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (num_samples / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

def infer_addition(model, num_digits, grid_size, stoi, itos):
    # (inference code remains the same as previous version)
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
            board_tensor = torch.tensor([[stoi[c] for c in row] for row in board], dtype=torch.long).unsqueeze(0).to(device)
            logits = model(board_tensor)
            predictions = logits.argmax(dim=-1).view(1, grid_size[0], grid_size[1])
            new_board = [[itos[p.item()] for p in row] for row in predictions[0]]
            if new_board == board: print("\nModel has converged."); break
            board = new_board
            print(f"\n--- Step {step+1} ---"); print("\n".join("".join(row) for row in board))
    print("\n--- Final Board State ---"); print("\n".join("".join(row) for row in board)); print(f"Correct Answer: {a+b}")

def infer_alignment(model, seq_len, grid_size, stoi, itos):
    # (inference code remains the same as previous version)
    model.eval()
    seq1 = ''.join(random.choices("ACGT", k=seq_len))
    seq2 = ''.join(random.choices("ACGT", k=seq_len))
    board = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    for i, char in enumerate(seq1): board[0][i+2] = char
    for i, char in enumerate(seq2): board[i+2][0] = char
    print("--- Initial Problem ---"); print(f"Seq1: {seq1}\nSeq2: {seq2}"); print("\n".join("".join(row) for row in board))
    with torch.no_grad():
        for step in range((seq_len+2)**2):
            board_tensor = torch.tensor([[stoi[c] for c in row] for row in board], dtype=torch.long).unsqueeze(0).to(device)
            logits = model(board_tensor)
            predictions = logits.argmax(dim=-1).view(1, grid_size[0], grid_size[1])
            new_board = [[itos[p.item()] for p in row] for row in predictions[0]]
            if new_board == board: print("\nModel has converged."); break
            board = new_board
            print(f"\n--- Step {step+1} ---"); print("\n".join("".join(row) for row in board))
    print("\n--- Final Board State ---"); print("\n".join("".join(row) for row in board))

# --- 6. Task-Specific Training Orchestration ---

def setup_and_train_addition():
    # --- Config ---
    VOCAB = list("0123456789+- ="); stoi = {c: i for i, c in enumerate(VOCAB)}; itos = {i: c for i, c in enumerate(VOCAB)}
    GRID_SIZE = (5, 8); NUM_DIGITS = 3; D_MODEL = 128; NHEAD = 8; NUM_LAYERS = 6
    BATCH_SIZE = 128; LEARNING_RATE = 1e-4; NUM_EPOCHS = 20; NUM_SAMPLES = 8000
    
    print("Generating training data for ADDITION...")
    all_transitions = []
    for _ in range(NUM_SAMPLES):
        all_transitions.extend(generate_addition_transitions(NUM_DIGITS, GRID_SIZE[0], GRID_SIZE[1]))
    
    all_input_gpu, all_target_gpu = preprocess_data_on_gpu(all_transitions, stoi, device)
    
    model = BlackboardTransformer(len(VOCAB), D_MODEL, NHEAD, NUM_LAYERS, GRID_SIZE, pe_type='ROPE').to(device)
    
    run_training_loop(model, all_input_gpu, all_target_gpu, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, VOCAB)

    print("\n--- Running Addition Inference ---")
    infer_addition(model, NUM_DIGITS, GRID_SIZE, stoi, itos)
    print("\n--- Running Addition Generalization ---")
    infer_addition(model, NUM_DIGITS + 1, GRID_SIZE, stoi, itos)

def setup_and_train_alignment():
    # --- Config ---
    VOCAB = list("ACGT0123456789-=") + [' ']; stoi = {c: i for i, c in enumerate(VOCAB)}; itos = {i: c for i, c in enumerate(VOCAB)}
    SEQ_LEN = 5; GRID_SIZE = (SEQ_LEN + 3, SEQ_LEN + 3); D_MODEL = 128; NHEAD = 8; NUM_LAYERS = 6
    BATCH_SIZE = 256; LEARNING_RATE = 1e-4; NUM_EPOCHS = 10; NUM_SAMPLES = 4000
    
    print("Generating training data for ALIGNMENT...")
    all_transitions = []
    for _ in range(NUM_SAMPLES):
        all_transitions.extend(generate_alignment_transitions(SEQ_LEN, GRID_SIZE[0], GRID_SIZE[1]))

    all_input_gpu, all_target_gpu = preprocess_data_on_gpu(all_transitions, stoi, device)

    model = BlackboardTransformer(len(VOCAB), D_MODEL, NHEAD, NUM_LAYERS, GRID_SIZE, pe_type='ROPE').to(device)

    run_training_loop(model, all_input_gpu, all_target_gpu, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE, VOCAB)

    print("\n--- Running Alignment Inference ---")
    infer_alignment(model, SEQ_LEN, GRID_SIZE, stoi, itos)

if __name__ == '__main__':
    print(f"Running Task: {TASK}")
    print(f"Using device: {device}")
    if TASK == 'ADDITION':
        setup_and_train_addition()
    elif TASK == 'ALIGNMENT':
        setup_and_train_alignment()
    else:
        raise ValueError(f"Unknown TASK specified: {TASK}")