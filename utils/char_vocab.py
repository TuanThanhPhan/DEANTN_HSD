def build_char_vocab(texts):
    """
    Tạo từ điển ký tự từ toàn bộ dữ liệu train
    """
    unique_chars = set()

    # lấy tất cả ký tự xuất hiện
    for text in texts:
        unique_chars.update(list(str(text)))

    # tạo mapping ký tự -> index
    char_to_idx = {
        char: i + 2
        for i, char in enumerate(sorted(list(unique_chars)))
    }

    # thêm token đặc biệt
    char_to_idx["<PAD>"] = 0
    char_to_idx["<UNK>"] = 1

    return char_to_idx