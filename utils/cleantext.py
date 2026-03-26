import re
from pyvi import ViTokenizer

# --- Load teencode_dict từ file ---
def load_teencode_dict(path="teencode_dict.txt"):

    teencode_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            key, value = parts
            teencode_dict[key] = value

    return teencode_dict

# load dictionary 1 lần duy nhất
teencode_dict = load_teencode_dict()


def clean_text_pipeline(text):
    if not isinstance(text, str):
        return ""

    # --- B1. Xử lý văn bản đơn giản (Giữ nguyên hoa thường cho PhoBERT cased) ---
    # Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Xóa Email
    text = re.sub(r'\S+@\S+', '', text)
    # Xóa Mention @user
    text = re.sub(r'@\w+', '', text)
    # Chuẩn hóa ký tự lặp
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Xóa ký tự rác nhưng giữ dấu câu và emoji
    text = re.sub(r'[^\w\s!?.,\U0001F300-\U0001FAFF]', ' ', text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # --- B2: Khôi phục Teencode (Duyệt theo word để không bị thay thế nhầm cụm từ) ---
    words = text.split()
    # Kiểm tra lowercase trong dict nhưng giữ nguyên format nếu không có trong dict
    words = [teencode_dict.get(w.lower(), w)
        for w in words
    ]
    text = " ".join(words)

    # --- B3: Tách từ bằng PyVi ---
    text = ViTokenizer.tokenize(text)

    return text