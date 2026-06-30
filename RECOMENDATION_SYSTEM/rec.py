import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

def split_sequential_data(user_history):
    train_history = {}
    val_history = {}
    test_history = {}
    
    for user_id, items in user_history.items():
        if len(items) < 3:
            continue
        train_history[user_id] = items[:-2]
        val_history[user_id] = items[:-1]
        test_history[user_id] = items
        
    return train_history, val_history, test_history

class SequentialRecDataset(Dataset):
    def __init__(self, user_history, max_seq_len):
        self.user_ids = [u for u, items in user_history.items() if len(items) > 1]
        self.user_history = user_history
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        items = self.user_history[user_id]
        
        input_seq = items[:-1][-self.max_seq_len:]
        target_seq = items[1:][-self.max_seq_len:]
        
        pad_len = self.max_seq_len - len(input_seq)
        
        return {
            'input_seq': torch.tensor([0] * pad_len + input_seq, dtype=torch.long),
            'target_seq': torch.tensor([0] * pad_len + target_seq, dtype=torch.long)
        }

class SASRec(nn.Module):
    def __init__(self, num_items, embed_dim, num_heads, num_layers, max_seq_len, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pred_layer = nn.Linear(embed_dim, num_items + 1)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand_as(x)
        
        emb = self.emb_dropout(self.item_emb(x) + self.pos_emb(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        return self.pred_layer(self.transformer(emb, mask=mask))

def evaluate_model(model, dataset, k=10, device='cpu'):
    model.eval() 
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    total_hits = 0.0
    total_ndcg = 0.0
    total_users = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            
            outputs = model(input_seq)
            last_step_logits = outputs[:, -1, :] 
            last_step_logits[:, 0] = -float('inf')
            
            _, top_k_indices = torch.topk(last_step_logits, k, dim=-1)
            true_targets = target_seq[:, -1].unsqueeze(1) 
            
            hits = (top_k_indices == true_targets).float()
            total_hits += hits.sum().item()
            
            ranks = hits.nonzero(as_tuple=True)[1] + 1.0 
            ndcg = (1.0 / torch.log2(ranks + 1.0)).sum().item()
            total_ndcg += ndcg
            
            total_users += input_seq.size(0)
            
    model.train() 
    return total_hits / total_users, total_ndcg / total_users

def train_sasrec(model, train_dataset, val_dataset, epochs=5, batch_size=64, lr=0.001, device='cpu'):
    model = model.to(device)
    model.train()
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_seq)
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        hr_10, ndcg_10 = evaluate_model(model, val_dataset, k=10, device=device)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val HR@10: {hr_10:.4f} | Val NDCG@10: {ndcg_10:.4f}")

def main():
    print("Загрузка датасета MovieLens 100K")
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    df = pd.read_csv(url, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    df = df.sort_values(by=['user_id', 'timestamp'])
    user_history = df.groupby('user_id')['item_id'].apply(list).to_dict()
    num_items = df['item_id'].max()
    
    train_hist, val_hist, test_hist = split_sequential_data(user_history)
    
    max_seq_len = 50
    train_dataset = SequentialRecDataset(train_hist, max_seq_len)
    val_dataset = SequentialRecDataset(val_hist, max_seq_len)
    test_dataset = SequentialRecDataset(test_hist, max_seq_len)
    
    model = SASRec(
        num_items=num_items, 
        embed_dim=64, 
        num_heads=2, 
        num_layers=2, 
        max_seq_len=max_seq_len, 
        dropout=0.2
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_sasrec(
        model=model, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        epochs=15, 
        batch_size=128, 
        device=device
    )
    
    test_hr, test_ndcg = evaluate_model(model, test_dataset, k=10, device=device)
    print(f"\nTest HR@10: {test_hr:.4f} | Test NDCG@10: {test_ndcg:.4f}")

if __name__ == '__main__':
    main()