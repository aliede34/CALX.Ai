import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import random
import os
from torch.optim.lr_scheduler import StepLR

# ---------------------- Tokenizer ----------------------
class CharTokenizer:
    def __init__(self, texts):
        unique_chars = sorted(set(''.join(texts)))
        self.stoi = {ch: i+1 for i, ch in enumerate(unique_chars)}  # 0 = PAD
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi) + 1

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join(self.itos.get(t, '?') for t in tokens)

# ---------------------- Dataset ----------------------
class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=128):
        with open(path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        self.samples = raw_text.strip().split('\n')
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.seq_len]
        padding = [0] * (self.seq_len - len(tokens))
        input_ids = torch.tensor(tokens + padding, dtype=torch.long)
        target_ids = input_ids.clone()
        return input_ids, target_ids

# ---------------------- Model ----------------------
class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=8, max_seq_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.ln(x)
        logits = self.head(x)
        return logits

# ---------------------- Eğitim ----------------------
def train():
    # Ayarlar
    batch_size = 8
    seq_len = 64
    epochs = 5
    lr = 2e-5
    patience = 3
    steps_per_epoch = 500  # Sabit adım sayısı
    best_loss = float('inf')
    patience_counter = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Veri yükleme
    with open('veri.txt', 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    tokenizer = CharTokenizer(lines)
    dataset = ChatDataset('veri.txt', tokenizer, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = GPTMini(tokenizer.vocab_size, max_seq_len=seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Toplam model parametresi: {total_params:,}")

    # Optimizer ve Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  # Learning rate'i her epoch'ta %5 azalt

    # Loss fonksiyonu
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[INFO] Eğitim başlıyor. Toplam örnek: {len(dataset)}")

    # Mixed Precision Training (GPU varsa)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Eğitim döngüsü
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            if step >= steps_per_epoch:  # Sabit adım sayısına ulaşıldığında durdur
                break
            x, y = x.to(device), y.to(device)

            # Mixed precision ile eğitim
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()

            # Gradyan hesapla ve adım at
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")

        avg_loss = total_loss / steps_per_epoch  # Ortalama kayıp hesapla
        print(f"[INFO] Epoch {epoch+1} tamamlandı. Ortalama kayıp: {avg_loss:.4f}")

        # Scheduler
        scheduler.step()

        # Early stopping kontrolü ve model kaydı
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("[INFO] En iyi model kaydedildi: best_model.pth")
        else:
            patience_counter += 1
            print(f"[INFO] Early stopping sayacı: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("[INFO] Eğitim erken durduruldu.")
                break

    print("[INFO] Eğitim tamamlandı.")
if __name__ == '__main__':
    train()
