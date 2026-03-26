import pickle
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config

from utils.dataloader import ViHSDDataset
from utils.cleantext import clean_text_pipeline

from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_type",
    type=str,
    default="hybrid",
    choices=["phobert", "visobert", "hybrid"]
)

parser.add_argument(
    "--model_name",
    type=str,
    default=None
)

parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["dev", "test"]
)

args = parser.parse_args()

# Tự động chọn model_name tương ứng với model_type nếu người dùng không truyền vào
if args.model_name is None:
    if args.model_type in ["phobert", "hybrid"]:
        args.model_name = "vinai/phobert-base"
    elif args.model_type == "visobert":
        args.model_name = "uitnlp/visobert" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


# chọn dataset
if args.split == "dev":
    df = pd.read_csv(config.DEV_PATH)
else:
    df = pd.read_csv(config.TEST_PATH)


df["free_text"] = df["free_text"].apply(clean_text_pipeline)

with open(f"{config.SAVE_DIR}/{config.CHAR_VOCAB_FILE}", "rb") as f:
    char_to_idx = pickle.load(f)

dataset = ViHSDDataset(
    df.free_text.values,
    df.label_id.values,
    tokenizer,
    config.MAX_LEN,
    char_to_idx
)

loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)


# chọn model
if args.model_type == "phobert":
    model = PhoBERTModel(args.model_name)

elif args.model_type == "visobert":
    model = ViSoBERTModel(args.model_name)

elif args.model_type == "hybrid":
    model = HybridHateSpeechModel(
        args.model_name,
        len(char_to_idx) + 2
    )


checkpoint = torch.load(
    f"{config.SAVE_DIR}/{args.model_type}_best.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

preds = []

with torch.no_grad():

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        if args.model_type == "hybrid":
            char_in = batch["char_input"].to(device)
            logits = model(input_ids, mask, char_in)
        else:
            logits = model(input_ids, mask)

        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

# Sau khi tính toán xong preds ở cuối file:
print("\n--- Classification Report ---")
print(classification_report(df.label_id.values, preds, target_names=["Bình thường", "Gây hấn", "Tiêu cực"]))

# --- Phần vẽ Confusion Matrix ---
def plot_confusion_matrix(y_true, y_pred, model_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Bình thường", "Gây hấn", "Tiêu cực"],
                yticklabels=["Bình thường", "Gây hấn", "Tiêu cực"])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.tight_layout()
    # Lưu ảnh vào thư mục SAVE_DIR để bạn có thể xem lại sau
    plt.savefig(f"{config.SAVE_DIR}/{model_type}_confusion_matrix.png")
    plt.show()

print("\nĐang hiển thị Ma trận nhầm lẫn...")
plot_confusion_matrix(df.label_id.values, preds, args.model_type)