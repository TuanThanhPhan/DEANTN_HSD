import pickle
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config
from utils.dataloader import ViHSDDataset
from utils.cleantext import clean_text_pipeline
from models.model import HybridHateSpeechModel

MODEL_TYPE = "hybrid"
MODEL_NAME = "vinai/phobert-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Tải Tokenizer và Dữ liệu cấu trúc Char
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    char_vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    with open(char_vocab_path, "rb") as f:
        char_to_idx = pickle.load(f)

    # Đọc và tiền xử lý DataCNTT.xlsx
    print(f"Đang đọc dữ liệu từ: {config.DATACNTT_PATH}")
    # Đọc file Excel
    df = pd.read_excel(config.DATACNTT_PATH)
    
    text_col = 'free_text' 
    label_col = 'label_id'
    
    df[text_col] = df[text_col].astype(str).apply(clean_text_pipeline)

    # Khởi tạo Dataset và Loader
    dataset = ViHSDDataset(
        df[text_col].values,
        df[label_col].values,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Khởi tạo và tải trọng số mô hình Hybrid
    model = HybridHateSpeechModel(
        MODEL_NAME,
        len(char_to_idx) + 2
    )

    checkpoint_path = os.path.join(config.SAVE_DIR, f"{MODEL_TYPE}_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(DEVICE)
    model.eval()

    # Tiến hành dự đoán
    preds = []
    print("Đang đánh giá hiệu suất...")
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            char_in = batch["char_input"].to(DEVICE)
            
            logits = model(input_ids, mask, char_in)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    # Xuất báo cáo và vẽ Ma trận nhầm lẫn
    target_names = ["Bình thường", "Gây hấn", "Tiêu cực"]
    
    print("\n" + "="*30)
    print(" KẾT QUẢ TRÊN TẬP DATACNTT ")
    print("="*30)
    print(classification_report(df[label_col].values, preds, 
                                target_names=target_names, digits=4))

    plot_confusion_matrix(df[label_col].values, preds, target_names)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Confusion Matrix - DataCNTT Evaluation')
    plt.tight_layout()
    
    save_path = os.path.join(config.SAVE_DIR, "datacntt_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"\nĐã lưu ma trận nhầm lẫn tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()