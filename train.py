import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.tensorboard import SummaryWriter

import config
from seed import set_seed

from utils.cleantext import clean_text_pipeline
from utils.dataloader import ViHSDDataset
from utils.char_vocab import build_char_vocab

from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel
from trainer import Trainer


def main():
    # ===== Định nghĩa tham số dòng lệnh =====
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
        default="vinai/phobert-base"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from checkpoint"
    )

    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Khởi tạo TensorBoard Writer
    log_dir = os.path.join(config.SAVE_DIR, "logs", args.model_type)
    writer = SummaryWriter(log_dir=log_dir)

    last_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_last.pt")
    best_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_best.pt")

    print("Model type:", args.model_type)
    print("Checkpoint:", last_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = pd.read_csv(config.TRAIN_PATH)
    dev_df = pd.read_csv(config.DEV_PATH)

    train_df["free_text"] = train_df["free_text"].apply(clean_text_pipeline)
    dev_df["free_text"] = dev_df["free_text"].apply(clean_text_pipeline)

    # ===== build char vocab =====
    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    if os.path.exists(vocab_path):
        print("Loading char vocab...")
        with open(vocab_path, "rb") as f:
            char_to_idx = pickle.load(f)

    else:

        print("Building char vocab...")

        char_to_idx = build_char_vocab(train_df.free_text.values)

        with open(vocab_path, "wb") as f:
            pickle.dump(char_to_idx, f)

    train_dataset = ViHSDDataset(
        train_df.free_text.values,
        train_df.label_id.values,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    dev_dataset = ViHSDDataset(
        dev_df.free_text.values,
        dev_df.label_id.values,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE)

    # ===== Chọn model =====
    if args.model_type == "phobert":
        model = PhoBERTModel(args.model_name)
    elif args.model_type == "visobert":
        model = ViSoBERTModel(args.model_name)
    elif args.model_type == "hybrid":
        model = HybridHateSpeechModel(
        args.model_name,
        len(char_to_idx) + 2
    )

    model.to(device)

    # ===== Compute class weights =====
    labels = train_df.label_id.values

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    # ===== Loss function =====
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ===== Optimizer với Differential Learning Rates =====
    if args.model_type == "hybrid":
        # Gom các tham số của PhoBERT gốc (đã có tri thức sẵn)
        phobert_params = list(model.phobert.parameters())
        # Gom các tham số mới hoàn toàn (reduce_phobert, CharCNN, BiLSTM, Classifier)
        # Chúng ta dùng "phobert." (có dấu chấm) để tránh nhầm với "reduce_phobert"
        custom_params = [p for n, p in model.named_parameters() if "phobert." not in n]
        optimizer = optim.AdamW([
            {'params': phobert_params, 'lr': config.LR},      # 2e-5 (Fine-tuning)
            {'params': custom_params, 'lr': 5e-5}             # 5e-5 (Học nhanh các lớp mới)
        ], weight_decay=0.01)
        print(f"Initialized Hybrid Optimizer: PhoBERT LR={config.LR}, Custom LR=5e-4")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.LR)
        print(f"Initialized Baseline Optimizer: LR={config.LR}")

    # ===== Warmup Scheduler =====
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps)

    # Truyền args.model_type vào Trainer
    trainer = Trainer(model, optimizer, criterion, device, scheduler, args.model_type)

     # ===== resume training =====
    start_epoch = 0
    best_f1 = 0
    patience = 0 # Khởi tạo patience mặc định

    if args.resume and os.path.exists(last_ckpt):

        print("Loading checkpoint...")

        checkpoint = torch.load(last_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        # Load patience từ checkpoint để không bị mất Early Stopping
        patience = checkpoint.get("patience", 0)

        print(f"Resumed from epoch: {start_epoch}, Current Best F1: {best_f1:.4f}, Patience: {patience}")

    # ===== training loop =====
    for epoch in range(start_epoch, config.EPOCHS):

        train_loss = trainer.train_epoch(train_loader)
        dev_f1 = trainer.eval_epoch(dev_loader)

        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Dev F1: {dev_f1:.4f}")

        # Ghi log vào TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('F1/dev', dev_f1, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "patience": patience
        }

        # save last checkpoint
        torch.save(checkpoint, last_ckpt)

        # save best checkpoint
        if dev_f1 > best_f1:

            best_f1 = dev_f1
            patience = 0

            checkpoint["best_f1"] = best_f1
            checkpoint["patience"] = patience
            torch.save(checkpoint, best_ckpt)

            print("Saved best model")

        else:
            patience += 1
            print(f"--> Patience: {patience}/{config.PATIENCE}")

        if patience >= config.PATIENCE:
            print("Early stopping")
            break
    # Đóng writer khi training xong
    writer.close()

if __name__ == "__main__":
    main()