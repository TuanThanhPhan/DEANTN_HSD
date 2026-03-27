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
            nn.ReLU()
        )

        # ===== Fusion =====
        self.fusion_fc = nn.Linear(256 + 128, 256)
        self.fusion_dropout = nn.Dropout(0.1)

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
        self.post_lstm_dropout = nn.Dropout(0.2)

        # ===== Bộ phân lớp =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Concat 3 lớp cuối 
        bert_out = torch.cat(outputs.hidden_states[-3:], dim=-1) 
        bert_out = self.dropout_bert(bert_out)
        bert_out = self.bert_norm(bert_out)
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

        char_feat = 0.3 * char_feat

        # ===== Fusion =====
        combined = torch.cat([bert_out, char_feat], dim=-1)
        combined = self.fusion_fc(combined)
        combined = self.fusion_dropout(combined)

        # ===== RESIDUAL GIỮ BERT =====
        combined = combined + bert_out

         # ===== BiLSTM =====
        lstm_out, _ = self.bilstm(combined)

        residual = self.residual_proj(combined)
        lstm_out = lstm_out + residual

        lstm_out = self.post_lstm_norm(lstm_out)
        lstm_out = self.post_lstm_dropout(lstm_out)
        
        # ================= Pooling =================
        # Mean pooling (masked)
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(lstm_out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        # Max pooling (masked)
        mask_bool = attention_mask.unsqueeze(-1).bool()
        lstm_masked = lstm_out.masked_fill(~mask_bool, -1e4)
        max_pooled = torch.max(lstm_masked, dim=1).values
        # concat
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        # ================= Classifier =================
        logits = self.classifier(pooled)

        return logits