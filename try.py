import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset ,DataLoader
from torch.nn.utils.rnn import pad_sequence 
import matplotlib.pyplot as plt
import math

device = torch.device('cpu')
print(f"目前使用的運算裝置: {device}")

import json
with open("data.json","r",encoding="utf-8") as f :
    raw_data = json.load(f)

class SimpleTokenizer:
    def __init__(self ,data,lang_idx):
        self.word2idx ={
            "<PAD>":0,
            "<BOS>":1,
            "<EOS>":2,
            "<UNK>":3,
        }
        self.idx2word ={
            0:"<PAD>",
            1:"<BOS>",
            2:"<EOS>",
            3:"<UNK>",
        }

        vocab = set()
        for pair in data:
            sentence = pair[lang_idx]
            # English: split by spaces
            if lang_idx == 0: 
                words = sentence.split()
            # Chinese: character-level
            else:
                words=list(sentence)
            vocab.update(words)
        
        for  i,word in enumerate(vocab):
            self.word2idx[word] = i+4
            self.idx2word[i+4]= word

    def encode(self,text,lang_type="en"):
        if lang_type =="en":
            words = text.split()
        else:
            words = list(text)
        return [self.word2idx.get(w,3) for w in words]
    
    def decode(self,indices):
        return "".join([self.idx2word.get(idx,"") for idx in indices if idx not in [0,1,2]])

src_tokenizer = SimpleTokenizer(raw_data, 0)
tgt_tokenizer = SimpleTokenizer(raw_data, 1)

class TranslationDataset(Dataset):
    def __init__(self,data,src_tok,tgt_tok):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        src_txt,tgt_txt = self.data[idx]

        src_ids = [1] + self.src_tok.encode(src_txt,'en')+[2]
        tgt_ids = [1] + self.tgt_tok.encode(tgt_txt,'ch')+[2]

        return torch.tensor(src_ids),torch.tensor(tgt_ids)
    
def collate_fn(batch):
    src_batch,tgt_batch=[],[]
    for src_sample ,tgt_sample in batch :
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_padded =pad_sequence(src_batch,batch_first=True,padding_value=0)
    tgt_padded =pad_sequence(tgt_batch,batch_first=True,padding_value=0)
    return src_padded, tgt_padded

class Seq2SeqTransformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,
                d_model=512,nhead=8,num_layers=3,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.src_embedding=nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding=nn.Embedding(tgt_vocab_size, d_model)

        self.transformer=nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out=nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt):
        src_emb = self.src_embedding(src)* math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        tgt_seq_len = tgt.size(1)
        tgt_mask =self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_mask = (tgt_mask == float("-inf"))

        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(outs)
    
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
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            logits = model(src, tgt_input)

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

    plt.show()

    return model
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
if __name__ == "__main__":
    trained_model = train()
    translate(trained_model, "I love AI")
    translate(trained_model, "Deep learning is fun")