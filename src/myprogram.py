#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import string
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import Counter

# Character n-gram model based on the Word2Vec architecture
class CharNgramW2V(nn.Module):
    # Initialize the model
    def __init__(self, vocab_size, n_gram_vocab_size, embedding_dim):
        # The superclass refers to nn.Module
        super(CharNgramW2V, self).__init__()

        # Contains the weight matrix and the bias vector
        self.linear_connection = nn.Linear(embedding_dim, vocab_size)

        # self.model_embeddnigs is a lookup table (map) that maps n-gram indices 
        # to vectors representing their meaning
        self.model_embeddings = nn.Embedding(n_gram_vocab_size, embedding_dim)

    # Processing input today to produce output
    def forward(self, x):
        # Gets the embeddings based on the input x
        embeddings = self.model_embeddings(x) 

        # Averages n-grams together to get a single vector representation for the n-preceding context
        avg_vector = torch.mean(embeddings, dim=1) 

        # Applies the linear transformation
        return self.linear_connection(avg_vector)


class ArsenalBarcaEaglesModel: # We named the class to something based on our interests
    def __init__(self, vocab=None, ngram_to_id=None, n=3, top_chars=None):
        # Reverse mapping for characters to indices
        self.window_context = 23 # We set the window context to 23 characters for the prediction phase
        self.reverse_id_map = {}
        index = 0
        self.vocab_found = vocab or []
        for character in self.vocab_found:
            self.reverse_id_map[character] = index
            index += 1
       
        self.ngram_to_id = ngram_to_id or {}
        self.n = n # n depends on the n-gram size
        self.top_chars = top_chars or " aeiou" # We initially fallback on the vowels because those are common characters, but we adjust later
        self.device = torch.device('cpu') # We will use CPU since we have M1 Mac
        self.model = None # Setting the model to None for now

    @classmethod
    # Writes predictions to a file, one per line
    def predictions_found(clss, predictions, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for i in predictions:
                f.write('{}\n'.format(i))

    @classmethod
    # Accesses the training data up to the limit
    def load_train(clss, dir, limit=5000000): # Adjusted limit for optimal training speed (30M)
        train_file = os.path.join(dir, 'wiki.train.tokens') # Adjusting path to actual training file
        if not os.path.exists(train_file): # We added this as a safety check
            return ""
        with open(train_file, 'r', encoding='utf-8') as f:
            return f.read(limit) # We read only up to the limit in training so training does not take too long

    @classmethod
    # Passing in the class test data
    def load_test(clss, fname):
        data = []
        with open(fname, 'r', encoding='utf-8') as f: # read the file
            for line in f:
                data.append(line.rstrip('\n')) # We remove the newline character at the end of each line (if it exists) before appending
        return data

   
    # Runs the training process for our model
    def train_run(self, train_doc, work_dir, epochs=5, batch_size=1024): # Adjusted epochs and batch size for training performance
        print(f"running training: ")
        
        char_counts = Counter(train_doc) # Tracks character counts

        # We don't only use string.printable because wikitext-103 could have unique characters 
        # not included in that set, so we take the union
        self.vocab_found = sorted(list(set(string.printable) | set(char_counts.keys())))

        # We reset the reverse_id_map based on the new vocab
        self.reverse_id_map = {}
        index = 0
        for character in self.vocab_found:
            self.reverse_id_map[character] = index
            index += 1

        # Reset the top 5 fallback based on frequency 
        top_five = char_counts.most_common(5)
        self.top_chars = ""
        for char, count in top_five:
            self.top_chars = self.top_chars + char

        print("getting n-grams: ")
        ngrams = [] # Holds every single n-gram we find
        for i in range(len(train_doc) - self.n):
            ngram = train_doc[i : i + self.n]
            ngrams.append(ngram)

        counts_for_ngrams = Counter(ngrams)

        # Get the most frequent pairs of n-grams that are found together
        most_common_pairs = counts_for_ngrams.most_common(155000) # We chose 155k as a good number of n-grams to keep for training

        # Initialize an empty list for just the strings
        most_common_ngs = []

        # Orders the n-grams by their frequency of most common occurrence
        for ng, count in most_common_pairs:
            most_common_ngs.append(ng)
            
        # Initialize the n_gram to id mapping based on commonality
        self.ngram_to_id = {}

        for i, ngram in enumerate(most_common_ngs):
            self.ngram_to_id[ngram] = i
        
        print("vectorizing: ")
        full_context = [] # holds the n-gram indices for the context of each character
        full_labels = [] # holds ground truth labels
        for i in range(self.window_context, len(train_doc) - 1):
            target_char = train_doc[i]
            if target_char not in self.reverse_id_map: continue # Skip characters not in vocab for now, we plan to later implement <unk> handling
            
            curr_context = train_doc[i-self.window_context:i]
            ngram_context = [self.ngram_to_id[curr_context[j : j + self.n]] 
            for j in range(len(curr_context)-self.n+1) 
                if curr_context[j : j + self.n] in self.ngram_to_id]
            
            if ngram_context: # Safety check
                full_context.append(torch.tensor(ngram_context))
                full_labels.append(self.reverse_id_map[target_char])

        self.model = CharNgramW2V(len(self.vocab_found), len(self.ngram_to_id), 256) # Chose embedding size of 256 instead of 128 based on performance
        optimizer = optim.Adam(self.model.parameters(), lr=0.005) # Adjusted LR to 0.005 for better convergence
        ce_loss = nn.CrossEntropyLoss() # Calculating CE loss

        print(f"training: ")
        for epoch in range(epochs): # Iterating through every epoch
            self.model.train() # Set the model to training mode
            total_L = 0 # Tracking total loss
            for i in range(0, len(full_context), batch_size):
                # Turns x_batch and y_batch into single rectangular tensors
                x_batch = torch.nn.utils.rnn.pad_sequence(full_context[i:i+batch_size], batch_first=True)
                y_batch = torch.tensor(full_labels[i:i+batch_size])

                # Clears memory of gradients from previous step
                optimizer.zero_grad()
                logits = self.model(x_batch) # Getting logits
                loss = ce_loss(logits, y_batch) # Updating loss
                loss.backward() # Backpropagation
                optimizer.step()
                total_L += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_L/(len(full_context)/batch_size):.4f}")

    # Runs prediction process with the test set        
    def pred_run(self, data):
        self.model.eval() # Sets model to eval mode
        predictions = [] # Creates predictions list
        with torch.no_grad():
            for line in data:
                context = line[-self.window_context:] # Gets the context of the line based on the window size
                ngrams_context = []

                # Loop through to find every possible n-gram pos
                for j in range(len(context) - self.n + 1):
                    ngram_string = context[j : j + self.n]

                    # If n_gram string exists from our training
                    if ngram_string in self.ngram_to_id:
                        # Look up and append ID number
                        ngram_id = self.ngram_to_id[ngram_string]
                        ngrams_context.append(ngram_id)
                
                if not ngrams_context:
                    predictions.append(self.top_chars) # Using updated top chars fallback if no n-grams are found in the context
                    continue
                
                x = torch.tensor([ngrams_context])
                logits = self.model(x) # Getting logits with forwarding the model
                top_idx = torch.topk(logits, 3).indices[0].tolist() # Getting the top 3 predictions for the next character (program requirement)
                
                str_prediction = "".join([self.vocab_found[i] for i in top_idx]) # Regular join for prediction string
                predictions.append(str_prediction)
        return predictions

    # Saves the model and metadata
    def save(self, work_dir):
        # Metadata to save the vocab, ngram_to_id mapping, n value, and top characters for fallback
        meta = {'vocab': self.vocab_found, 'ngram_to_id': self.ngram_to_id, 'n': self.n, 'top_chars': self.top_chars}
        with open(os.path.join(work_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        torch.save(self.model.state_dict(), os.path.join(work_dir, 'model.pt'))

    @classmethod
    # Loads the model and metadata
    def load(clss, work_dir):
        with open(os.path.join(work_dir, 'meta.json'), 'r') as f:
            meta = json.load(f)
        # Instantiate the class with the loaded metadata and then load the model state dict
        inst = clss(vocab=meta['vocab'], ngram_to_id=meta['ngram_to_id'], n=meta['n'], top_chars=meta['top_chars'])
        inst.model = CharNgramW2V(len(inst.vocab_found), len(inst.ngram_to_id), 256) # Matching model architecture
        inst.model.load_state_dict(torch.load(os.path.join(work_dir, 'model.pt'), map_location=inst.device))
        return inst

if __name__ == '__main__':
    # We set up the argument parser to handle command line arguments for training and testing (explained in the README)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'))
    parser.add_argument('--work_dir', default='work')
    parser.add_argument('--test_data', default='example/input.txt')
    parser.add_argument('--test_output', default='pred.txt')
    parser.add_argument('--train_data', default='wikitext-103')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir): os.makedirs(args.work_dir)
        m = ArsenalBarcaEaglesModel(n=3) # We do a tri-gram model based on performance testing
        text_data = ArsenalBarcaEaglesModel.load_train(args.train_data)
        m.train_run(text_data, args.work_dir)
        m.save(args.work_dir)
    elif args.mode == 'test':
        m = ArsenalBarcaEaglesModel.load(args.work_dir)
        data = m.load_test(args.test_data)
        pred = m.pred_run(data)
        m.predictions_found(pred, args.test_output)