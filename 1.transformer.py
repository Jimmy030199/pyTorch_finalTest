# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import math

# ======================================================
# 第 0 段 — 環境偵測（選擇 CPU / GPU / M1）
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"目前使用的運算裝置: {device}")


# ======================================================
# 1. Raw Data
# ======================================================
import json
# 讀取 JSON
with open("data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# raw_data = [
#     ("I love AI", "我愛人工智慧"),
#     ("Deep learning is fun", "深度學習很有趣"),
#     ("Transformer is powerful", "變壓器模型很強大"),
#     ("This is a long sentence to test padding mechanism", "這是一個用來測試填充機制的句子"),
#     ("GPU makes training faster", "GPU讓訓練變更快"),
#     ("Seq2Seq model is cool", "序列到序列模型很酷"),
# ]


# ======================================================
# 2. Tokenizer
# ======================================================
class SimpleTokenizer:
    def __init__(self, data, lang_idx):
        self.word2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

        vocab = set()
        for pair in data:
            sentence = pair[lang_idx]
            if lang_idx == 0:  # English: split by spaces
                words = sentence.split()
            else:  # Chinese: character-level
                words = list(sentence)
            vocab.update(words)
        # 繼續建自典
        for i, word in enumerate(vocab):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

    def encode(self, text, lang_type="en"):
        if lang_type == "en":
            words = text.split()
        else:
            words = list(text)

        # 對於每個字詞 w：
        # 如果 w 在字典中 → 回傳它的 index
        # 如果不在 → 回傳 3（代表 <UNK>）
        return [self.word2idx.get(w, 3) for w in words]

    def decode(self, indices):
        return "".join([self.idx2word.get(idx, "") for idx in indices if idx not in [0, 1, 2]])

# 建立 英文 tokenizer，負責把英文轉成 token IDs → 給 Encoder 用。
src_tokenizer = SimpleTokenizer(raw_data, 0)
# 建立 中文 tokenizer，負責把中文轉成 token IDs → 給 Decoder 用。
tgt_tokenizer = SimpleTokenizer(raw_data, 1)


# ======================================================
# 3. Dataset -把一筆筆文字資料，轉成 Transformer 能訓練的形式（ID + PAD + Batch）。
# ======================================================
class TranslationDataset(Dataset):
    def __init__(self, data, src_tok, tgt_tok):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    # 提供資料筆數給 DataLoader。
    def __len__(self):
        return len(self.data)
    
    # __getitem__ 公式用意
    # ("I love AI", "我愛人工智慧")
    #     │ English Tokenizer
    #     ▼
    # encode("I love AI") → [10,42,7]
    # 加 <BOS>/<EOS> → [1,10,42,7,2]
    #     │ Chinese Tokenizer
    #     ▼
    # encode("我愛人工智慧") → [11,7,33,20,19,58]
    # 加 <BOS>/<EOS> → [1,11,7,33,20,19,58,2]
    # 轉成 tensor → 回傳

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        src_ids = [1] + self.src_tok.encode(src_text, "en") + [2]  # BOS ... EOS
        tgt_ids = [1] + self.tgt_tok.encode(tgt_text, "ch") + [2]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# 把 Dataset 回傳的「多筆句子」組成一個 batch，
# 並且自動補 <PAD> 讓所有序列長度一致，
# 好讓 Transformer 可以一次處理整個 batch
# DataLoader 會自動呼叫 collate_fn 來組 batch。
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


# ======================================================
# 4. Seq2Seq Transformer Model
# ======================================================
class Seq2SeqTransformer(nn.Module):
    # __init__ 用意:
    # 建立 Transformer 的所有核心組件：
    # 來源 embedding（英文）→ 目標 embedding（中文）→ Encoder/Decoder → 輸出層（預測下一個 token）。
    # 換句話說，
    # 這裡就是整個 Seq2Seq Transformer 翻譯模型的架構定義。
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # forward 用意
    # 把英文 token IDs → 英文 embedding → Encoder → Decoder（帶 causal mask）→ 預測下一個中文字。
    
    def forward(self, src, tgt):
        # math.sqrt(self.d_model)=>Embedding scaling 讓語意與位置訊號維持相同量級，
        # 避免模型初期學不到語意資訊
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # 建立 Decoder Mask（避免看到未來）
        # 例如 tgt length = 5，mask 如下：
        # 位置： 0    1   2    3    4
        # 0    [ 0 -inf -inf -inf -inf ]
        # 1    [ 0   0  -inf -inf -inf ]
        # 2    [ 0   0    0  -inf -inf ]
        # 3    [ 0   0    0    0  -inf ]
        # 4    [ 0   0    0    0    0  ]
        # 再轉成
        # tensor([
        #     [False, True,  True,  True,  True ],
        #     [False, False, True,  True,  True ],
        #     [False, False, False, True,  True ],
        #     [False, False, False, False, True ],
        #     [False, False, False, False, False],
        # ])

        # tgt 的 shape：
        # (2, 7)
        # ↑   ↑
        # batch seq_len
        tgt_seq_len = tgt.size(1) #去得到seq_len
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_mask = (tgt_mask == float("-inf"))

        # Padding Mask（mask 掉 <PAD> token）
        # 讓模型不要對 <PAD>（padding）做 attention。
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        # 這一行完成所有 Transformer
        # outs = 語意向量
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(outs)


# ======================================================
# 5. Training Function (含 Loss & Accuracy 紀錄)
# ======================================================
# 反覆讀取資料 → 前向傳播 → 計算 loss → 反向傳播 → 更新 Transformer 權重。
def train():
    BATCH_SIZE = 2
    EPOCHS = 20
    LR = 0.0001

    dataset = TranslationDataset(raw_data, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_tokenizer.word2idx),
        tgt_vocab_size=len(tgt_tokenizer.word2idx),
        d_model=256,
        nhead=4,
        num_layers=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # ignore_index=0 的意思就是：
    # ❌ 不計算 PAD 的 loss
    # ❌ 不更新 PAD 的梯度
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses = []
    train_accs = []

    model.train()
    print("開始訓練...")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total_tokens = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # tgt:        [BOS,  我,  愛,  你, EOS]
            # Index:        0    1    2    3    4

            # tgt_input:  [BOS,  我,  愛,  你]
            # Index:        0    1    2    3

            # tgt_output: [ 我,  愛,  你, EOS]
            # Index:        1    2    3    4

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()

            logits = model(src, tgt_input)

            # -1 = 自動計算所需的維度大小
            # logits.reshape(-1, vocab_size)
            # 把 (batch_size, seq_len, vocab_size)
            # 攤平成 (batch_size * seq_len, vocab_size)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tgt_output.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ===== 計算 accuracy =====
            pred = logits.argmax(dim=-1)
            mask = (tgt_output != 0)
            correct += ((pred == tgt_output) & mask).sum().item()
            total_tokens += mask.sum().item()

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total_tokens

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # ====== 畫 Loss & Accuracy 圖 ======
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("training_loss_acc.png")   # ⭐ 儲存圖片
    plt.show()

    return model


# ======================================================
# 6. Inference
# ======================================================
# 把英文輸入一句句生成中文 → 一次生成一個 token（自回歸）。
# 這裡的流程跟 GPT 生成文字完全一樣，只是你用的是 Encoder-Decoder 架構。
def translate(model, src_sentence):
    model.eval()

    src_ids = [1] + src_tokenizer.encode(src_sentence, "en") + [2]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    print(f"\n輸入句子: {src_sentence}")
    print("翻譯結果:", end=" ")

    tgt_ids = [1]  # <BOS>

    for _ in range(20):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)

        next_token_id = logits[0, -1].argmax().item()
        tgt_ids.append(next_token_id)

        if next_token_id == 2:  # <EOS>
            break

    result = tgt_tokenizer.decode(tgt_ids)
    print(result)


# ======================================================
# 7. Main
# ======================================================
if __name__ == "__main__":
    trained_model = train()
    translate(trained_model, "I love AI")
    translate(trained_model, "Deep learning is fun")
