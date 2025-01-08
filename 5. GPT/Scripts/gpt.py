import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import time
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 6
n_layer = 2
dropout = 0.2
# ------------

""" # fastgpt
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 2
n_layer = 2
dropout = 0.1

# antioverfittinggpt
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 6
n_layer = 2
dropout = 0.4 
"""


toSuffleText = False

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    # Added to train a baseline model
    if toSuffleText:
        text = ''.join(random.sample(text, len(text)))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Add a special token for unknown characters
unknown_token = '<UNK>'
chars.append(unknown_token)
vocab_size += 1

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] if c in stoi else stoi[unknown_token] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = time.time()

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end = time.time()
print(f"training time: {end - start} seconds")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

def evaluate_on_test_set(model, test_file_path, results_file):
    """
    Evaluate the model on a test set and save detailed results
    
    Args:
        model: The trained GPT model
        test_file_path: Path to the test file
        results_file: Path to save the results
    """
    # Load test data
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    # Encode test data
    test_data = torch.tensor(encode(test_text), dtype=torch.long)
    test_data = test_data.to(device)
    
    # Evaluation settings
    model.eval()
    block_size = model.position_embedding_table.weight.shape[0]
    
    # Lists to store all losses and positions
    losses = []
    positions = []
    
    with torch.no_grad():
        # Process test data in blocks
        for i in range(0, len(test_data) - block_size, block_size):
            # Get input and target sequences using block_size
            x = test_data[i:i + block_size].unsqueeze(0)
            y = test_data[i + 1:i + block_size + 1].unsqueeze(0)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Store individual loss and position
            losses.append(loss.item())
            positions.append(i)
    
    # Calculate perplexity using average loss
    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Generate text using block_size context
    context_text = test_text[:block_size]
    context_encoded = torch.tensor(encode(context_text), 
                                 dtype=torch.long, 
                                 device=device).unsqueeze(0)
    generated_tokens = model.generate(context_encoded, max_new_tokens=200)
    generated_text = decode(generated_tokens[0].tolist())
    
    # Save all results to file
    with open(results_file, 'w', encoding='utf-8') as f:
        # Write summary statistics
        f.write(f"=== Evaluation Results for {test_file_path} ===\n\n")
        f.write(f"Block size used: {block_size}\n")
        f.write(f"Final Perplexity: {perplexity:.2f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Number of blocks evaluated: {len(losses)}\n\n")
        
        # Write loss progression
        f.write("=== Loss Progression ===\n")
        f.write("Position,Loss\n")
        for pos, loss in zip(positions, losses):
            f.write(f"{pos},{loss}\n")
        
        # Write generated text
        f.write("\n=== Text Generation ===\n")
        f.write(f"Context used ({len(context_text)} characters):\n{context_text}\n\n")
        f.write("Generated text:\n")
        f.write(generated_text[len(context_text):])
    
    # Return data for plotting
    return {
        'positions': positions,
        'losses': losses,
        'perplexity': perplexity,
        'avg_loss': avg_loss
    }

def plot_loss_progression(evaluation_results, test_file, output_file):
    """
    Create and save a plot of the loss progression
    """
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_results['positions'], evaluation_results['losses'])
    plt.title(f'Loss Progression - {test_file}')
    plt.xlabel('Position in Text')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Evaluate on both test sets
test_sets = [
    'input_childSpeech_testSet.txt',
    'input_shakespeare.txt'
]

# Store results for comparison
all_results = {}

for test_file in test_sets:
    try:
        # Create results filename
        results_file = f'detailed_results_{test_file.replace(".txt", "")}_dummy.txt'
        plot_file = f'loss_progression_{test_file.replace(".txt", "")}_dummy.png'
        
        # Evaluate and save results
        results = evaluate_on_test_set(model, test_file, results_file)
        all_results[test_file] = results
        
        # Create loss progression plot
        plot_loss_progression(results, test_file, plot_file)
        
        # Save comparison data
        with open('comparison_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n=== {test_file} ===\n")
            f.write(f"Final Perplexity: {results['perplexity']:.2f}\n")
            f.write(f"Average Loss: {results['avg_loss']:.4f}\n")
            f.write(f"Min Loss: {min(results['losses']):.4f}\n")
            f.write(f"Max Loss: {max(results['losses']):.4f}\n")
            f.write("-" * 50 + "\n")
        
    except FileNotFoundError:
        warning_str = f"\nWarning: Could not find test file {test_file}\n"
        with open(f'error_log.txt', 'a', encoding='utf-8') as f:
            f.write(warning_str)