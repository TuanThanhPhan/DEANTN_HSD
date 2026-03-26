import pandas as pd
import os

# 1. Đường dẫn
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN_TN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
extra_file = os.path.join(BASE_PATH, "addHSD.xlsx")
output_dir = os.path.join(BASE_PATH, "processed")

os.makedirs(output_dir, exist_ok=True)

print("Đang đọc dữ liệu...")

# 2. Đọc ViHSD
train_df = pd.read_csv(os.path.join(vihsd_path, "train.csv"))
dev_df   = pd.read_csv(os.path.join(vihsd_path, "dev.csv"))
test_df  = pd.read_csv(os.path.join(vihsd_path, "test.csv"))

# 3. Đọc file Excel bổ sung
extra_df = pd.read_excel(extra_file)
print("Số mẫu thêm:", len(extra_df))

# 4. Gộp vào train
final_train = pd.concat([train_df, extra_df], ignore_index=True)

# shuffle
final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
# dev và test giữ nguyên
final_dev = dev_df
final_test = test_df

# 5. Lưu dataset mới
final_train.to_csv(os.path.join(output_dir, "final_train.csv"), index=False, encoding="utf-8-sig")
final_dev.to_csv(os.path.join(output_dir, "final_dev.csv"), index=False, encoding="utf-8-sig")
final_test.to_csv(os.path.join(output_dir, "final_test.csv"), index=False, encoding="utf-8-sig")

print("Hoàn thành. Dataset lưu tại:", output_dir)