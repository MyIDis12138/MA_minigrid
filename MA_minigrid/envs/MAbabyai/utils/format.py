import os
import json

class Vocabulary:
    def __init__(self, file_path='../vocab/vocab1.txt'):
        self.file_path = file_path
        self.vocab = {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for word in lines:
                token, part_of_speech = word.split()
                self.vocab[token] = len(self.vocab) + 1
                if part_of_speech == 'noun':
                    self.noun_idx = len(self.vocab)
                if part_of_speech == 'adj':
                    self.adj_idx = len(self.vocab)

        self.max_size = len(self.vocab) + 2
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            print("UnSEEN!!!", token)
            raise NotImplementedError
            if len(self.vocab) >= self.max_size:
                print(token)
                raise ValueError("Maximum vocabulary capacity reached")
            old_vocab_len = len(self.vocab)
            self.vocab[token] = old_vocab_len + 1
            self.inverse_vocab[old_vocab_len + 1] = token

        return self.vocab[token]

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}