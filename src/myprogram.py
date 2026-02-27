#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import string
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class TextSequenceDataset(Dataset):
    def __init__(self, text, window_context, n_gram_size, ngram_to_id, char_to_id):
        self.text = text
        self.window_context = window_context
        self.n = n_gram_size
        self.ngram_to_id = ngram_to_id
        self.char_to_id = char_to_id
        self.unk_ngram_id = ngram_to_id.get('<UNK>', 1)
        self.unk_char_id = char_to_id.get('<UNK_CHAR>', 0)

    def __len__(self):
        return len(self.text) - self.window_context

    def __getitem__(self, idx):
        context_str = self.text[idx : idx + self.window_context]
        target_char = self.text[idx + self.window_context]
        
        ngram_ids = [
            self.ngram_to_id.get(context_str[j : j + self.n], self.unk_ngram_id) 
            for j in range(len(context_str) - self.n + 1)
        ]
        
        x = torch.tensor(ngram_ids, dtype=torch.long)
        y = torch.tensor(self.char_to_id.get(target_char, self.unk_char_id), dtype=torch.long)
        return x, y

class CharNgramLSTM(nn.Module):
    def __init__(self, vocab_size, n_gram_vocab_size, embedding_dim, hidden_dim=256):
        super(CharNgramLSTM, self).__init__()
        self.embeddings = nn.Embedding(n_gram_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear_connection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeddings = self.embeddings(x) 
        lstm_out, (hidden, cell) = self.lstm(embeddings)
        final_state = lstm_out[:, -1, :] 
        return self.linear_connection(final_state)

class ArsenalBarcaEaglesModel: 
    def __init__(self, vocab=None, ngram_to_id=None, n=3, top_chars=None):
        self.window_context = 23 
        self.vocab_found = vocab or []
        self.reverse_id_map = {character: index for index, character in enumerate(self.vocab_found)}
        
        self.ngram_to_id = ngram_to_id or {'<PAD>': 0, '<UNK>': 1}
        self.n = n 
        self.top_chars = top_chars or "aei" 
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(self.device)
            
        self.model = None 

    @classmethod
    def predictions_found(clss, predictions, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for i in predictions:
                f.write('{}\n'.format(i))
    @classmethod
    def load_train(clss, local_path, hf_split_key='train', limit=None): 
        combined_text = ""
        
        # Split the limit evenly between the two datasets
        local_limit = limit // 2 if limit else None
        hf_limit = limit - local_limit if limit else None

        print(f"Loading local data from: {local_path}...")
        train_file = local_path if os.path.isfile(local_path) else os.path.join(local_path, 'wiki.train.tokens')
            
        if os.path.exists(train_file): 
            with open(train_file, 'r', encoding='utf-8') as f:
                combined_text += f.read(local_limit) if local_limit else f.read()
        else:
            print(f"Warning: Local training file not found at {train_file}")

        print(f"Downloading '{hf_split_key}' split from Hugging Face via pandas...")
        splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
        csv_file = splits.get(hf_split_key, 'train.csv')
        
        url = f"hf://datasets/papluca/language-identification/{csv_file}"
        try:
            df = pd.read_csv(url)
            print("Processing DataFrame into a text sequence...")
            hf_text = "\n".join(df['text'].dropna().tolist())
            
            combined_text += "\n" + (hf_text[:hf_limit] if hf_limit else hf_text)
        except Exception as e:
            print(f"Warning: Could not load Hugging Face dataset. Error: {e}")
        
        return combined_text

    @classmethod
    def load_test(clss, fname):
        data = []
        with open(fname, 'r', encoding='utf-8') as f: 
            for line in f:
                data.append(line.rstrip('\n')) 
        return data

    def train_run(self, train_doc, work_dir, epochs=5, batch_size=1024, max_char_vocab=2000): 
        print(f"Running training on device: {self.device}")
        if not train_doc:
            print("Error: Empty training document. Aborting training.")
            return

        print("Building vocabulary...")
        char_counts = Counter(train_doc) 
        
        common_chars = [char for char, count in char_counts.most_common(max_char_vocab)]
        self.vocab_found = ['<UNK_CHAR>'] + sorted(list(set(string.printable) | set(common_chars)))
        self.reverse_id_map = {character: index for index, character in enumerate(self.vocab_found)}

        top_three = char_counts.most_common(3)
        self.top_chars = "".join([char for char, count in top_three])

        print("Counting n-grams...")
        ngrams = (train_doc[i : i + self.n] for i in range(len(train_doc) - self.n))
        counts_for_ngrams = Counter(ngrams)
        most_common_pairs = counts_for_ngrams.most_common(155000) 

        self.ngram_to_id = {'<PAD>': 0, '<UNK>': 1}
        for i, (ng, count) in enumerate(most_common_pairs):
            self.ngram_to_id[ng] = i + 2
        
        print("Initializing model and dataset...")
        self.model = CharNgramLSTM(len(self.vocab_found), len(self.ngram_to_id), 256).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005) 
        ce_loss = nn.CrossEntropyLoss() 

        dataset = TextSequenceDataset(
            train_doc, 
            self.window_context, 
            self.n, 
            self.ngram_to_id, 
            self.reverse_id_map
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

        print("Training...")
        for epoch in range(epochs): 
            self.model.train() 
            total_L = 0.0 
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch) 
                loss = ce_loss(logits, y_batch) 
                loss.backward() 
                optimizer.step()
                
                total_L += loss.item()
                
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)} | Current Loss: {loss.item():.4f}")

            avg_loss = total_L / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} Completed | Average Loss: {avg_loss:.4f}")
       
    def pred_run(self, data, batch_size=1024):
        print(f"Running inference on device: {self.device}")
        self.model.eval() 
        self.model.to(self.device)
        
        predictions = [""] * len(data) 
        all_contexts = []
        valid_indices = []
        
        for i, line in enumerate(data):
            context = line[-self.window_context:] 
            ngrams_context = []

            for j in range(len(context) - self.n + 1):
                ngram_string = context[j : j + self.n]
                ngram_id = self.ngram_to_id.get(ngram_string, self.ngram_to_id.get('<UNK>', 1))
                ngrams_context.append(ngram_id)
            
            max_seq_length = self.window_context - self.n + 1
            if len(ngrams_context) > 0 and len(ngrams_context) < max_seq_length:
                padding = [0] * (max_seq_length - len(ngrams_context))
                ngrams_context = padding + ngrams_context
            
            if not ngrams_context:
                predictions[i] = self.top_chars 
            else:
                all_contexts.append(ngrams_context)
                valid_indices.append(i)
                
        with torch.no_grad():
            for i in range(0, len(all_contexts), batch_size):
                batch_contexts = all_contexts[i:i+batch_size]
                x = torch.tensor(batch_contexts, dtype=torch.long).to(self.device)
                
                logits = self.model(x) 
                top_indices = torch.topk(logits, 3, dim=-1).indices.tolist() 
                
                for j, top_idx in enumerate(top_indices):
                    str_prediction = "".join([self.vocab_found[idx] for idx in top_idx]) 
                    original_idx = valid_indices[i + j]
                    predictions[original_idx] = str_prediction
                    
        return predictions

    def save(self, work_dir):
        meta = {'vocab': self.vocab_found, 'ngram_to_id': self.ngram_to_id, 'n': self.n, 'top_chars': self.top_chars}
        with open(os.path.join(work_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        torch.save(self.model.state_dict(), os.path.join(work_dir, 'model.pt'))

    @classmethod
    def load(clss, work_dir):
        with open(os.path.join(work_dir, 'meta.json'), 'r') as f:
            meta = json.load(f)
        inst = clss(vocab=meta['vocab'], ngram_to_id=meta['ngram_to_id'], n=meta['n'], top_chars=meta['top_chars'])
        inst.model = CharNgramLSTM(len(inst.vocab_found), len(inst.ngram_to_id), 256) 
        
        inst.model.load_state_dict(torch.load(os.path.join(work_dir, 'model.pt'), map_location=inst.device, weights_only=True))
        inst.model.to(inst.device)
        return inst

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'))
    parser.add_argument('--work_dir', default='work')
    parser.add_argument('--test_data', default='example/input.txt')
    parser.add_argument('--test_output', default='pred.txt')
    parser.add_argument('--train_data', default='wikitext-103', help="Path to local training data")
    parser.add_argument('--train_split', default='train', choices=['train', 'validation', 'test'], help="Which dataset split to pull from HF via pandas")
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir): os.makedirs(args.work_dir)
        m = ArsenalBarcaEaglesModel(n=3) 
        text_data = ArsenalBarcaEaglesModel.load_train(local_path=args.train_data, hf_split_key=args.train_split, limit=25000000) 
        
        m.train_run(text_data, args.work_dir)
        m.save(args.work_dir)
    elif args.mode == 'test':
        m = ArsenalBarcaEaglesModel.load(args.work_dir)
        data = m.load_test(args.test_data)
        pred = m.pred_run(data)
        m.predictions_found(pred, args.test_output)