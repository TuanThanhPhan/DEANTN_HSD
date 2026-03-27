import sys
import os
from pathlib import Path
from utils.dataloader import ViHSDDataset

# Xử lý đường dẫn: Trỏ từ web_demo/ ngược lên thư mục gốc HSD_DEAN_TN
# Path(__file__).resolve() là file app.py
# .parent là thư mục web_demo/
# .parent.parent là thư mục gốc HSD_DEAN_TN
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import pickle
import numpy as np
from flask import Flask, render_template, request
from transformers import AutoTokenizer

# Các module này nằm ở thư mục gốc
import config
from utils.cleantext import clean_text_pipeline
from models.model import HybridHateSpeechModel

app = Flask(__name__)

# --- Cấu hình ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "vinai/phobert-base" 
LABEL_MAP = {0: "Bình thường", 1: "Gây hấn", 2: "Tiêu cực"}
COLORS = {0: "success", 1: "warning", 2: "danger"}

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)

with open(vocab_path, "rb") as f:
    char_to_idx = pickle.load(f)

model = HybridHateSpeechModel(MODEL_NAME, len(char_to_idx) + 2)

# Load mô hình hybrid_best.pt từ Drive
model_path = os.path.join(config.SAVE_DIR, "hybrid_best.pt")
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        raw_text = request.form.get("content")
        if raw_text:
            # Tiền xử lý & Dự đoán
            cleaned_text = clean_text_pipeline(raw_text)

            # BƯỚC 2: Đồng bộ hóa logic Char-level bằng Dataset class
            # labels=[0] là giả lập để Dataset không lỗi, không ảnh hưởng kết quả
            temp_dataset = ViHSDDataset(
                texts=[cleaned_text], 
                labels=[0], 
                tokenizer=tokenizer, 
                max_len=config.MAX_LEN, 
                char_to_idx=char_to_idx
            )

            # Lấy dữ liệu đã được xử lý chuẩn (Tách từ, padding char, long tensor...)
            data_item = temp_dataset[0]

            # BƯỚC 3: Chuẩn bị Tensor
            input_ids = data_item['input_ids'].to(DEVICE)
            mask = data_item['attention_mask'].to(DEVICE)
            # Unsqueeze(0) để biến từ [max_len, 15] thành [1, max_len, 15] (batch size = 1)
            char_tensor = data_item['char_input'].unsqueeze(0).to(DEVICE)

            # BƯỚC 4: Dự đoán
            with torch.no_grad():
                logits = model(input_ids, mask, char_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                
            result = {
                "text": raw_text,
                "label": LABEL_MAP[pred_idx],
                "conf": round(probs[pred_idx] * 100, 2),
                "color": COLORS[pred_idx]
            }
        return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)