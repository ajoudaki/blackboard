import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
import time

# --- 1. Configuration & Vocabulary ---

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary mapping characters to integers
VOCAB = list("0123456789+- =") + ["[CLS]", "[EOS]"]

stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# Hyperparameters
GRID_SIZE = (5, 8)  # (Height, Width) for the blackboard
NUM_DIGITS = 3      # Number of digits for the addition problems
D_MODEL = 128       # Model dimension
NHEAD = 8           # Number of attention heads
NUM_LAYERS = 6      # Number of Transformer layers
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_SAMPLES = 5000  # Number of training samples to generate

# --- 2. Data Generation ---

def generate_addition_problem(num_digits, grid_h, grid_w):
    """
    Generates a multi-digit addition problem and its step-by-step solution trace.
    Each step consists of the board state and the next action (row, col, symbol).
    """
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    c = a + b

    # Format the problem onto the blackboard
    problem = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    lines = problem.split('\n')
    
    board = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            board[r+1][grid_w - len(line) + c] = char
    
    solution_trace = []
    str_c = str(c)
    carry = 0
    
    # Simulate the column-by-column addition algorithm
    for i in range(num_digits):
        col_idx = grid_w - 1 - i
        digit_a = int(board[1][col_idx]) if board[1][col_idx].isdigit() else 0
        digit_b = int(board[2][col_idx]) if board[2][col_idx].isdigit() else 0
        
        current_sum = digit_a + digit_b + carry
        result_digit = current_sum % 10
        carry = current_sum // 10

        # Step 1: Write the result digit
        current_board_state = [row[:] for row in board]
        solution_trace.append((current_board_state, (3, col_idx, str(result_digit))))
        board[3][col_idx] = str(result_digit)

        # Step 2: Write the carry digit (if any)
        if carry > 0 and i < num_digits - 1:
            current_board_state = [row[:] for row in board]
            solution_trace.append((current_board_state, (0, col_idx - 1, str(carry))))
            board[0][col_idx-1] = str(carry)
            
    # Handle final carry
    if carry > 0:
        col_idx = grid_w - 1 - num_digits
        current_board_state = [row[:] for row in board]
        solution_trace.append((current_board_state, (3, col_idx, str(carry))))
        board[3][col_idx] = str(carry)
    
    # Add final state with EOS token
    current_board_state = [row[:] for row in board]
    solution_trace.append((current_board_state, (0, 0, '[EOS]')))

    return solution_trace


# --- 3. Dataset ---

class BlackboardDataset(Dataset):
    """PyTorch Dataset for blackboard reasoning problems."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, (r, c, symbol) = self.data[idx]
        
        # Convert board to tensor of token IDs
        board_tensor = torch.tensor(
            [[stoi[char] for char in row] for row in board_state], dtype=torch.long
        )
        
        # Convert target action to tensors
        target_r = torch.tensor(r, dtype=torch.long)
        target_c = torch.tensor(c, dtype=torch.long)
        target_symbol = torch.tensor(stoi[symbol], dtype=torch.long)
        
        return board_tensor, target_r, target_c, target_symbol

# --- 4. Model Architecture ---

class BlackboardTransformer(nn.Module):
    """
    Transformer model that learns to reason on a 2D blackboard.
    It takes the entire board state and predicts the next (row, col, symbol) action.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, grid_size):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding_2d = self._create_2d_sinusoidal_embedding()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Prediction Heads
        self.row_head = nn.Linear(d_model, self.grid_h)
        self.col_head = nn.Linear(d_model, self.grid_w)
        self.symbol_head = nn.Linear(d_model, vocab_size)

        # Special [CLS] token for aggregating global board state
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def _create_2d_sinusoidal_embedding(self):
        """Creates and registers a fixed 2D sinusoidal positional embedding."""
        pos_embedding = torch.zeros(self.grid_h, self.grid_w, self.d_model)
        pos = torch.arange(max(self.grid_h, self.grid_w), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        # Row embeddings (even dimensions)
        pe_row = torch.sin(pos[:self.grid_h] * div_term)
        pos_embedding[:, :, 0::2] = pe_row.unsqueeze(1).repeat(1, self.grid_w, 1)

        # Column embeddings (odd dimensions)
        pe_col = torch.cos(pos[:self.grid_w] * div_term)
        pos_embedding[:, :, 1::2] = pe_col.unsqueeze(0).repeat(self.grid_h, 1, 1)

        self.register_buffer('pos_embedding_2d_buffer', pos_embedding.flatten(0, 1)) # (H*W, D)
        return self.pos_embedding_2d_buffer

    def forward(self, board):
            B, H, W = board.shape
            
            # 1. Flatten board and add token embeddings
            flat_board = board.view(B, H * W)
            token_emb = self.token_embedding(flat_board) # (B, H*W, D)
            
            # 2. Add 2D positional embeddings
            # Use the registered buffer name 'pos_embedding_2d_buffer' here (CORRECTED)
            pos_emb = self.pos_embedding_2d_buffer.unsqueeze(0).repeat(B, 1, 1) # (B, H*W, D)
            x = token_emb + pos_emb
            
            # 3. Prepend [CLS] token
            cls_tokens = self.cls_token.repeat(B, 1, 1)
            x = torch.cat([cls_tokens, x], dim=1) # (B, 1 + H*W, D)

            # 4. Pass through Transformer
            transformer_out = self.transformer_encoder(x)
            
            # 5. Use the output of the [CLS] token for prediction
            cls_out = transformer_out[:, 0] # (B, D)
            
            # 6. Predict row, column, and symbol
            row_logits = self.row_head(cls_out)
            col_logits = self.col_head(cls_out)
            symbol_logits = self.symbol_head(cls_out)
            
            return row_logits, col_logits, symbol_logits

# --- 5. Training and Inference ---

def train():
    """Main function to generate data, train the model, and run inference."""
    print(f"Using device: {device}")
    
    # 1. Generate Data
    print("Generating training data...")
    all_traces = []
    for _ in range(NUM_SAMPLES):
        trace = generate_addition_problem(NUM_DIGITS, GRID_SIZE[0], GRID_SIZE[1])
        all_traces.extend(trace)
    
    dataset = BlackboardDataset(all_traces)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = BlackboardTransformer(
        vocab_size=len(VOCAB),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        grid_size=GRID_SIZE
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print("Starting training...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in dataloader:
            boards, targets_r, targets_c, targets_s = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            logits_r, logits_c, logits_s = model(boards)
            
            loss_r = criterion(logits_r, targets_r)
            loss_c = criterion(logits_c, targets_c)
            loss_s = criterion(logits_s, targets_s)
            
            loss = loss_r + loss_c + loss_s
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 4. Run Inference Example
    print("\n--- Running Inference Example ---")
    infer(model, num_digits=NUM_DIGITS) # Test on same number of digits
    print("\n--- Running Generalization Example (harder) ---")
    infer(model, num_digits=NUM_DIGITS + 1) # Test on more digits

def infer(model, num_digits):
    """Runs the model autoregressively to solve a new addition problem."""
    model.eval()
    
    # Create a new problem
    a = random.randint(10**(num_digits-1), 10**num_digits - 1)
    b = random.randint(10**(num_digits-1), 10**num_digits - 1)
    problem = f"{a:>{num_digits+1}}\n+{b:>{num_digits}}\n{'-'*(num_digits+1)}"
    lines = problem.split('\n')
    
    board = [[' ' for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
    for r, line in enumerate(lines):
        if r + 1 < GRID_SIZE[0] and len(line) < GRID_SIZE[1]:
          for c, char in enumerate(line):
              board[r+1][GRID_SIZE[1] - len(line) + c] = char

    def print_board(b):
        print("\n".join("".join(row) for row in b))
        print("-" * GRID_SIZE[1])

    print("Initial Problem:")
    print_board(board)
    
    with torch.no_grad():
        for step in range(num_digits * 3): # Set a max number of steps
            board_tensor = torch.tensor(
                [[stoi[c] for c in row] for row in board], dtype=torch.long
            ).unsqueeze(0).to(device)
            
            logits_r, logits_c, logits_s = model(board_tensor)
            
            pred_r = logits_r.argmax(dim=-1).item()
            pred_c = logits_c.argmax(dim=-1).item()
            pred_s_idx = logits_s.argmax(dim=-1).item()
            pred_symbol = itos[pred_s_idx]
            
            if pred_symbol == '[EOS]':
                print(f"\nStep {step+1}: Model predicted [EOS]. Solution complete.")
                break
            
            print(f"\nStep {step+1}: Model writes '{pred_symbol}' at ({pred_r}, {pred_c})")
            
            if 0 <= pred_r < GRID_SIZE[0] and 0 <= pred_c < GRID_SIZE[1]:
                board[pred_r][pred_c] = pred_symbol
                print_board(board)
            else:
                print("Predicted position is out of bounds. Stopping.")
                break
    
    print("\nFinal Board State:")
    print_board(board)
    print(f"Correct Answer: {a+b}")

if __name__ == '__main__':
    train()
