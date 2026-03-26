import torch
import torch.nn as nn
from transformers import AutoModel


class HybridHateSpeechModel(nn.Module):

    def __init__(self, phobert_path, char_vocab_size, hidden_dim=128):
        super().__init__()

        # ===== PhoBERT =====
        self.phobert = AutoModel.from_pretrained(phobert_path)
        self.dropout = nn.Dropout(0.1)
        # Thêm lớp Linear hạ chiều PhoBERT từ 2304 xuống 256 (Giúp tín hiệu từ CharCNN không bị lấn át)
        self.reduce_phobert = nn.Linear(2304, 256)
        # ===== CharCNN =====
        # Embedding ký tự 50 chiều
        self.char_embedding = nn.Embedding(char_vocab_size, 50)
        # Sử dụng Multi-scale CNN (kernel 2,3,4) là cách tốt nhất để bắt teencode
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=k, padding=k//2) 
            for k in [2, 3, 4]
        ])
        # Tổng output CNN là 150 (3 * 50), nén về 128
        self.char_fc = nn.Linear(150, 128)
        # ==== BiLSTM ====
        # Input: PhoBERT_reduced (256) + Char_features (128) = 384
        self.bilstm = nn.LSTM(
            input_size=384,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        # ==== Classifier ====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
    def forward(self, input_ids, attention_mask, char_input):
        # --- PhoBERT Branch ---
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = outputs.hidden_states
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1) # [B, S, 2304]
        bert_out = self.dropout(bert_out)

        # Hạ chiều PhoBERT
        bert_out = torch.relu(self.reduce_phobert(bert_out)) # [B, S, 256]

        # --- CharCNN Branch ---
        B, S, L = char_input.shape
        char_x = self.char_embedding(char_input) # [B, S, L, 50]
        char_x = char_x.view(B * S, L, -1).transpose(1, 2) # [B*S, 50, L]

        conv_results = []
        for conv in self.convs:
            c = torch.relu(conv(char_x))
            c, _ = torch.max(c, dim=2) # Max-over-time pooling cho ký tự
            conv_results.append(c)
        
        char_feat = torch.cat(conv_results, dim=1) # [B*S, 150]
        char_feat = torch.relu(self.char_fc(char_feat)) # [B*S, 128]
        char_feat = char_feat.view(B, S, 128) # [B, S, 128]

        # --- Hybrid Combine ---
        combined = torch.cat((bert_out, char_feat), dim=2) # [B, S, 384]
        lstm_out, _ = self.bilstm(combined) # [B, S, 256]

        # --- Masked Mean Pooling ---
        mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        sum_embeddings = torch.sum(lstm_out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # --- Output ---
        return self.classifier(mean_pooled)