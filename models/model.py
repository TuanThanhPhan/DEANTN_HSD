import torch
import torch.nn as nn
from transformers import AutoModel

class HybridHateSpeechModel(nn.Module):
    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()

        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout_bert = nn.Dropout(0.1) 
        
        # Hạ chiều PhoBERT từ 2304 xuống 256 
        self.reduce_phobert = nn.Linear(2304, 256)

        # ===== CharCNN =====
        self.char_embedding = nn.Embedding(char_vocab_size, 50, padding_idx=0)
        # Multi-scale CNN (kernel 2,3,4,5)
        self.convs = nn.ModuleList([
            nn.Conv1d(50, 64, kernel_size=k, padding=k//2) 
            for k in [2, 3, 4, 5]
        ])
        # Nén output CNN (64*4 = 256) về 128
        self.char_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # ===== BiLSTM =====
        # Input: PhoBERT (256) + Char (128) = 384
        # BiLSTM hidden 128 * 2 hướng = 256 chiều đầu ra
        self.bilstm = nn.LSTM(
            input_size=384,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        # Thêm Residual Connection để giữ lại đặc trưng gốc (giúp tăng F1)
        self.residual_proj = nn.Linear(384, 256)

        # ===== Bộ phân lớp =====
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Concat 3 lớp cuối 
        bert_out = torch.cat(outputs.hidden_states[-3:], dim=-1) 
        bert_out = self.dropout_bert(bert_out)
        bert_out = torch.relu(self.reduce_phobert(bert_out)) # Thêm ReLU cho đồng bộ [B, S, 256]

        # --- CharCNN Branch ---
        B, S, L = char_input.shape
        char_x = self.char_embedding(char_input).view(B * S, L, -1).transpose(1, 2)
        
        char_conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(char_x))
            c, _ = torch.max(c, dim=2) # Max-over-time pooling
            char_conv_outs.append(c)
        
        char_feat = torch.cat(char_conv_outs, dim=1) # [B*S, 256]
        char_feat = self.char_fc(char_feat).view(B, S, 128) # [B, S, 128]

        # --- Fusion & BiLSTM ---
        combined = torch.cat((bert_out, char_feat), dim=2) # [B, S, 384]
        
        lstm_out, _ = self.bilstm(combined) # [B, S, 256]
        
        # Residual Connection: Cộng tín hiệu gốc vào output LSTM
        lstm_out = lstm_out + self.residual_proj(combined)

        # --- Masked Mean Pooling (Bỏ qua Padding) ---
        mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        sum_embeddings = torch.sum(lstm_out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask # [B, 256]

        # --- Final Output ---
        return self.classifier(mean_pooled)