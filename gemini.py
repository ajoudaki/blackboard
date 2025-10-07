import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
import time

# --- 1. Configuration & Vocabulary ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary WITHOUT the [MASK] token
VOCAB = list("0123456789+- =")
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# Hyperparameters
GRID_SIZE = (5, 8)
NUM_DIGITS = 3
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY=0.01
NUM_EPOCHS = 25
NUM_SAMPLES = 5000
POS_EMBEDDING_TYPE = 'ROPE'

# --- 2. Data Generation (State Transition Approach) ---

def generate_state_transitions(num_digits, grid_h, grid_w):
    """Generates a sequence of (board_t, board_t+1) state transitions."""
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    c = a + b

    board = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    problem_str = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    lines = problem_str.split('\n')
    for r, line in enumerate(lines):
        for col, char in enumerate(line):
            board[r+1][grid_w - len(line) + col] = char

    state_transitions = []
    carry = 0
    
    # Simulate the algorithm, storing the board state at each step
    for i in range(num_digits):
        col_idx = grid_w - 1 - i
        digit_a = int(board[1][col_idx])
        digit_b = int(board[2][col_idx])
        
        current_sum = digit_a + digit_b + carry
        result_digit = current_sum % 10
        new_carry = current_sum // 10

        # Step 1: Add the result digit
        prev_board_state = [row[:] for row in board]
        board[3][col_idx] = str(result_digit)
        state_transitions.append((prev_board_state, [row[:] for row in board]))

        # Step 2: Add the carry digit
        if new_carry > 0:
            prev_board_state = [row[:] for row in board]
            board[0][col_idx - 1] = str(new_carry)
            state_transitions.append((prev_board_state, [row[:] for row in board]))
        
        carry = new_carry
            
    # Handle final carry if it results in a new digit
    if str(c)[0] == '1' and len(str(c)) > num_digits:
        col_idx = grid_w - 1 - num_digits
        prev_board_state = [row[:] for row in board]
        board[3][col_idx] = '1'
        state_transitions.append((prev_board_state, [row[:] for row in board]))
        
    return state_transitions


# --- 3. Dataset ---

class BlackboardDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_board, target_board = self.data[idx]
        input_tensor = torch.tensor([[stoi[c] for c in row] for row in input_board], dtype=torch.long)
        target_tensor = torch.tensor([[stoi[c] for c in row] for row in target_board], dtype=torch.long)
        return input_tensor, target_tensor

# --- 4. Model Architecture (No changes needed) ---

class RoPETransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)
    def _sa_block(self, x, attn_mask, key_padding_mask, pos_ids=None, rope=None):
        if rope is not None and pos_ids is not None:
            q = k = rope.rotate_queries_and_keys(x, pos_ids)
            return self.self_attn(q, k, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        else:
            return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
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
        self.grid_h, self.grid_w = grid_size
        self.pe_type = pe_type
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
        pe_row = torch.sin(pos[:self.grid_h] * div_term)
        pos_embedding[:, :, 0::2] = pe_row.unsqueeze(1).repeat(1, self.grid_w, 1)
        pe_col = torch.cos(pos[:self.grid_w] * div_term)
        pos_embedding[:, :, 1::2] = pe_col.unsqueeze(0).repeat(self.grid_h, 1, 1)
        return pos_embedding.flatten(0, 1)

    def forward(self, board):
        B, H, W = board.shape
        flat_board = board.view(B, H * W)
        token_emb = self.token_embedding(flat_board)
        if self.pe_type == 'ROPE':
            rows = torch.arange(H, device=board.device).view(1, H, 1).expand(B, H, W)
            cols = torch.arange(W, device=board.device).view(1, 1, W).expand(B, H, W)
            pos_ids = torch.stack([rows, cols], dim=-1).view(B, H * W, 2)
            x = token_emb
            for layer in self.transformer_encoder.layers:
                x = layer(x, pos_ids=pos_ids, rope=self.rope)
        else: # ABSOLUTE
            pos_emb = self.pos_embedding_2d_buffer.unsqueeze(0).repeat(B, 1, 1)
            x = token_emb + pos_emb
            x = self.transformer_encoder(x)
        logits = self.output_head(x)
        return logits

# --- 5. Training and Inference ---

def train():
    print(f"Using device: {device}")
    print(f"Using Positional Embedding: {POS_EMBEDDING_TYPE}")
    
    print("Generating training data...")
    all_transitions = []
    for _ in range(NUM_SAMPLES):
        all_transitions.extend(generate_state_transitions(NUM_DIGITS, GRID_SIZE[0], GRID_SIZE[1]))
    
    dataset = BlackboardDataset(all_transitions)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = BlackboardTransformer(len(VOCAB), D_MODEL, NHEAD, NUM_LAYERS, GRID_SIZE, POS_EMBEDDING_TYPE).to(device)
    
    # Use an ignore_index that is not in our vocabulary
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    print("Starting training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for input_boards, target_boards in dataloader:
            input_boards, target_boards = input_boards.to(device), target_boards.to(device)
            
            optimizer.zero_grad()
            logits = model(input_boards) # (B, H*W, V)
            
            # Create a target for the loss function that only includes the changed cells
            loss_target = target_boards.clone()
            # loss_target[input_boards == target_boards] = -100 # Ignore unchanged cells
            
            loss = criterion(logits.view(-1, len(VOCAB)), loss_target.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f} seconds.")

    print("\n--- Running Inference Example ---")
    infer(model, num_digits=NUM_DIGITS)
    print("\n--- Running Generalization Example (harder) ---")
    infer(model, num_digits=NUM_DIGITS + 1)

def infer(model, num_digits):
    """Solves a problem autoregressively by predicting the next board state."""
    model.eval()
    
    try:
        a = random.randint(10**(num_digits-1), 10**num_digits - 1)
        b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    except ValueError:
        print(f"Cannot generate a {num_digits}-digit number. Skipping inference.")
        return
        
    problem_str = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    board = [[' ' for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
    lines = problem_str.split('\n')
    for r, line in enumerate(lines):
        if r + 1 < GRID_SIZE[0] and GRID_SIZE[1] - len(line) >= 0:
            for c, char in enumerate(line):
                board[r+1][GRID_SIZE[1] - len(line) + c] = char

    def print_board(b, step):
        print(f"\n--- Step {step} ---")
        print("\n".join("".join(row) for row in b))

    print("--- Initial Problem ---")
    print("\n".join("".join(row) for row in board))
    
    with torch.no_grad():
        for step in range(num_digits * 2 + 2): # Max steps for completion
            board_tensor = torch.tensor([[stoi[c] for c in row] for row in board], dtype=torch.long).unsqueeze(0).to(device)
            
            logits = model(board_tensor) # (1, H*W, V)
            predictions = logits.argmax(dim=-1).view(1, GRID_SIZE[0], GRID_SIZE[1])
            
            # Create the new board from predictions
            new_board = [[itos[p.item()] for p in row] for row in predictions[0]]
            
            # If the model outputs the same board, it's done or stuck
            if new_board == board:
                print("\nModel has converged. Solution complete.")
                break
            
            board = new_board
            print_board(board, step + 1)

    print("\n--- Final Board State ---")
    print("\n".join("".join(row) for row in board))
    print(f"Correct Answer: {a+b}")

if __name__ == '__main__':
    train()