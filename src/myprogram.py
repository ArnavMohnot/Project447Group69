import os
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import json
import string
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

class ArsenalBarcaEagles(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x)
        out, _ = self.gru(h)
        return self.head(out)

class CharDataset(Dataset):
    def __init__(self, text: str, char_to_id: dict, seq_len: int = 128):
        self.seq_len    = seq_len
        self.unk_id     = char_to_id.get("<UNK>", 1)

        self.ids = torch.tensor(
            [char_to_id.get(c, self.unk_id) for c in text],
            dtype=torch.long,
        )

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.ids[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

class SuperBowlModel:
    SEQ_LEN    = 128
    EMBED_DIM  = 128
    HIDDEN_DIM = 256
    N_LAYERS   = 2

    def __init__(self, vocab: list | None = None, top_chars: str = "e t"):
        self.vocab      = vocab or []
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.top_chars  = top_chars

        self.device = (
            torch.device("cuda")  if torch.cuda.is_available()  else
            torch.device("mps")   if torch.backends.mps.is_available() else
            torch.device("cpu")
        )
        print(f"Device set to: {self.device}")
        self.model: ArsenalBarcaEagles | None = None

    @staticmethod
    def build_vocab(text: str, max_vocab: int = 4096) -> list[str]:
        always_include = set(string.printable)
        counts = Counter(c for c in text if c not in always_include)
        extra  = [c for c, _ in counts.most_common(max_vocab - len(always_include))]
        vocab = ["<PAD>", "<UNK>"] + sorted(always_include) + extra
        return vocab

    @classmethod
    def predictions_found(cls, predictions, fname):
        with open(fname, "wt", encoding="utf-8") as f:
            for p in predictions:
                f.write(f"{p}\n")

    @classmethod
    def load_train(cls, local_path, hf_split_key="train", limit=None):
        combined = ""
        
        if not os.path.exists(local_path):
            print(f"Dataset not found at {local_path}. Downloading WikiText-103...")
            os.makedirs(local_path, exist_ok=True)
            zip_path = os.path.join(local_path, "wikitext-103-v1.zip")
            urllib.request.urlretrieve(
                "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip", 
                zip_path
            )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_path)
            print("Download and extraction complete.")

        train_file = os.path.join(local_path, "wikitext-103", "wiki.train.tokens")
        if not os.path.exists(train_file):
            train_file = os.path.join(local_path, "wiki.train.tokens")

        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                chunk = f.read(limit // 2 if limit else None)
            combined += chunk
            print(f"Loaded {len(chunk):,} chars from WikiText")

        splits = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}
        url = f"hf://datasets/papluca/language-identification/{splits.get(hf_split_key, 'train.csv')}"
        try:
            df  = pd.read_csv(url)
            hf_text = "\n".join(df["text"].dropna().tolist())
            if limit:
                hf_text = hf_text[: limit - len(combined)]
            combined += "\n" + hf_text
            print(f"Loaded {len(hf_text):,} chars from HuggingFace ({hf_split_key})")
        except Exception as e:
            print(f"Warning: HuggingFace load failed — {e}")

        return combined

    @classmethod
    def load_test(cls, fname):
        with open(fname, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def train_run(
        self,
        train_doc: str,
        work_dir: str,
        epochs: int = 5,
        batch_size: int = 256,
        max_vocab: int = 4096,
        lr: float = 1e-3,
    ):
        print("Building vocabulary...")
        self.vocab      = self.build_vocab(train_doc, max_vocab)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}

        counts = Counter(train_doc)
        self.top_chars = "".join(c for c, _ in counts.most_common(3))

        print(f"Vocab size: {len(self.vocab)}")

        self.model = ArsenalBarcaEagles(
            vocab_size = len(self.vocab),
            embed_dim  = self.EMBED_DIM,
            hidden_dim = self.HIDDEN_DIM,
            num_layers = self.N_LAYERS,
        ).to(self.device)

        dataset    = CharDataset(train_doc, self.char_to_id, seq_len=self.SEQ_LEN)
        dataloader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = 2,
            pin_memory  = True,
            drop_last   = True,
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        total_steps  = epochs * len(dataloader)
        warmup_steps = total_steps // 10
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr      = lr,
            total_steps = total_steps,
            pct_start   = warmup_steps / total_steps,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        print("Training GRU LM...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss   = loss_fn(
                    logits.reshape(-1, len(self.vocab)),
                    y.reshape(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                
                if step % 10 == 0:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })

            print(f"Epoch {epoch+1}/{epochs} | Avg loss {total_loss/len(dataloader):.4f}")

    def pred_run(self, data: list[str], batch_size: int = 256) -> list[str]:
        self.model.eval()
        self.model.to(self.device)

        unk_id = self.char_to_id.get("<UNK>", 1)
        pad_id = self.char_to_id.get("<PAD>", 0)

        predictions: list[str] = [""] * len(data)
        contexts:    list[list[int]] = []
        valid_idx:   list[int] = []

        for i, line in enumerate(data):
            ctx = line[-self.SEQ_LEN:]
            ids = [self.char_to_id.get(c, unk_id) for c in ctx]

            if not ids:
                predictions[i] = self.top_chars
                continue

            if len(ids) < self.SEQ_LEN:
                ids = [pad_id] * (self.SEQ_LEN - len(ids)) + ids

            contexts.append(ids)
            valid_idx.append(i)

        with torch.no_grad():
            for start in range(0, len(contexts), batch_size):
                try:
                    batch = contexts[start : start + batch_size]
                    x     = torch.tensor(batch, dtype=torch.long).to(self.device)

                    logits   = self.model(x)
                    last_log = logits[:, -1, :]

                    top3 = torch.topk(last_log, 3, dim=-1).indices.tolist()

                    for j, indices in enumerate(top3):
                        pred = "".join(self.id_to_char.get(idx, "?") for idx in indices)
                        predictions[valid_idx[start + j]] = pred
                
                except Exception as e:
                    print(f"Warning: Batch inference failed - {e}")
                    for j in range(len(batch)):
                        if not predictions[valid_idx[start + j]]:
                            predictions[valid_idx[start + j]] = self.top_chars

        return predictions

    def save(self, work_dir: str):
        os.makedirs(work_dir, exist_ok=True)
        meta = {"vocab": self.vocab, "top_chars": self.top_chars}
        with open(os.path.join(work_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        torch.save(self.model.state_dict(), os.path.join(work_dir, "model.pt"))
        print(f"Model saved to {work_dir}/")

    @classmethod
    def load(cls, work_dir: str) -> "SuperBowlModel":
        with open(os.path.join(work_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        inst = cls(vocab=meta["vocab"], top_chars=meta["top_chars"])
        inst.model = ArsenalBarcaEagles(
            vocab_size = len(inst.vocab),
            embed_dim  = cls.EMBED_DIM,
            hidden_dim = cls.HIDDEN_DIM,
            num_layers = cls.N_LAYERS,
        )
        inst.model.load_state_dict(
            torch.load(
                os.path.join(work_dir, "model.pt"),
                map_location=inst.device,
                weights_only=True,
            )
        )
        inst.model.to(inst.device)
        return inst
  
if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--work_dir",     default="work")
    parser.add_argument("--test_data",    default="example/input.txt")
    parser.add_argument("--test_output",  default="pred.txt")
    parser.add_argument("--train_data",   default="wikitext-103")
    parser.add_argument("--train_split",  default="train",
                        choices=["train", "validation", "test"])
    args = parser.parse_args()

    if args.mode == "train":
        os.makedirs(args.work_dir, exist_ok=True)
        m    = SuperBowlModel()
        text = SuperBowlModel.load_train(
            local_path   = args.train_data,
            hf_split_key = args.train_split,
            limit        = 5_000_000, 
        )
        m.train_run(text, args.work_dir)
        m.save(args.work_dir)

    elif args.mode == "test":
        m    = SuperBowlModel.load(args.work_dir)
        data = SuperBowlModel.load_test(args.test_data)
        pred = m.pred_run(data)
        SuperBowlModel.predictions_found(pred, args.test_output)