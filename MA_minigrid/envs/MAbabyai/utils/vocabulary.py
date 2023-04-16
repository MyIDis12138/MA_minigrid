"""
Copied and adapted from https://github.com/IouJenLiu/AFK
"""

from __future__ import annotations

import os
import json
from .. import utils


class Vocabulary:
    def __init__(self, max_size=30, file_path:str=''):
        """
        Vocabulary class for the dataset
        Vocabulary is a dictionary of words and their corresponding indices
        Example of a vocabulary file:
            word1 nonn
            ...
            wordn noun
            wordn+1 adj
            ...
            wordm adj
            others type
            ...
        """
        assert os.path.exists(self.path), "Vocabulary file path must be provided"
        self.path = file_path
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

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
