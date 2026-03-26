import torch
import torch.nn as nn
from transformers import AutoModel

class ViSoBERTModel(nn.Module):
    def __init__(self, model_path="vinai/visobert-base", num_labels=3):
        super(ViSoBERTModel, self).__init__()
        self.visobert = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        
        # Hạ chiều tương tự để so sánh công bằng
        self.reduce_visobert = nn.Linear(2304, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, attention_mask, char_input=None):
        outputs = self.visobert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Concat 3 lớp cuối: [-1], [-2], [-3] -> [B, S, 2304]
        all_layers = outputs.hidden_states
        bert_out = torch.cat((all_layers[-1], all_layers[-2], all_layers[-3]), dim=-1)
        bert_out = self.dropout(bert_out)
        
        # Hạ chiều về 256
        bert_out = torch.relu(self.reduce_visobert(bert_out)) # [B, S, 256]
        
        # Masked Mean Pooling
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(bert_out * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts

        return self.classifier(mean_pooled)