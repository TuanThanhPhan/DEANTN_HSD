import torch
import torch.nn as nn
from transformers import AutoModel

class HybridHateSpeechModel(nn.Module):
    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()
        
        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout_bert = nn.Dropout(0.1) 
        self.bert_norm = nn.LayerNorm(2304)
        # Hạ chiều PhoBERT từ 2304 xuống 256 
        self.reduce_phobert = nn.Linear(2304, 256)

        # ===== CharCNN =====
        self.char_embedding = nn.Embedding(char_vocab_size, 50, padding_idx=0)
        # Multi-scale CNN (kernel 2,3,4,5)
        self.convs = nn.ModuleList([
            nn.Conv1d(50, 64, kernel_size=k, padding="same") 
            for k in [2, 3, 4, 5]
        ])
        # Nén output CNN (64*4 = 256) về 128
        self.char_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # ===== Fusion =====
        # Input = 256 (BERT) + 128 (Char) = 384
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ===== BiLSTM =====
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Thêm Residual Connection để giữ lại đặc trưng gốc (giúp tăng F1)
        self.residual_proj = nn.Linear(256, hidden_dim * 2)

        self.post_lstm_norm = nn.LayerNorm(hidden_dim * 2)
        self.post_lstm_dropout = nn.Dropout(0.3)

        # ===== Bộ phân lớp =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3) # 3 nhãn
        )

    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = outputs.hidden_states
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1)
        bert_out = self.dropout_bert(bert_out)
        bert_out = self.bert_norm(bert_out)
        bert_out = torch.relu(self.reduce_phobert(bert_out)) # [B, S, 256]

        # --- CharCNN Branch ---
        B, S, W = char_input.shape
        char_in = char_input.view(-1, W) 
        char_emb = self.char_embedding(char_in).transpose(1, 2)

        char_conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(char_emb))
            c = torch.max(c, dim=2)[0]
            char_conv_outs.append(c)
        
        char_feat = torch.cat(char_conv_outs, dim=1) 
        char_feat = self.char_fc(char_feat).view(B, S, 128) # [B, S, 128]

        # --- Fusion ---
        combined = torch.cat([bert_out, char_feat], dim=-1) 
        combined = self.fusion_fc(combined) # Nén về 256 và chuẩn hóa

        # --- BiLSTM & Residual ---
        lstm_out, _ = self.bilstm(combined)
        lstm_out = lstm_out + self.residual_proj(combined)
        lstm_out = self.post_lstm_norm(lstm_out)
        lstm_out = self.post_lstm_dropout(lstm_out)
        
        # --- Pooling (Mean + Max) ---
        mask = attention_mask.unsqueeze(-1).float()
        mean_pooled = torch.sum(lstm_out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        
        mask_bool = attention_mask.unsqueeze(-1).bool()
        max_pooled = torch.max(lstm_out.masked_fill(~mask_bool, -1e4), dim=1).values
        
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1) # [B, 512]

        return self.classifier(pooled)