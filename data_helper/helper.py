# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 23:24:36 2018

@author: flyin
"""

# -*- coding: utf-8 -*-
import json
import re
import numpy as np

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-?!.,\(\)]+",
                          re.UNICODE)

def tokenizer(iterator):
    """Tokenizer generator.
    Args:
        iterator: Input iterator with strings.
    Yields:
        array of tokens per each value in the input.
    """
    for value in iterator:
        yield TOKENIZER_RE.findall(value)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'[^\x00-\x7F]+', '', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def load_all_data_and_labels(filenames):
    utterances = []
    labels = []
    for filename in filenames:
        with open(filename) as f:
            dialogues = json.loads(f.read())
            for dialogue in dialogues:
                for line in dialogue:
                    utterances.append(re.sub(r'[^\x00-\x7F]+', '', line['utterance']))
                    labels.append(line['emotion'])
    return utterances, labels

def load_sub_dialogues(filename, num_seq=5):
    sub_dialogues = []
    last_labels = []
    with open(filename) as f:
        dialogues = json.loads(f.read())
        for dialogue in dialogues:
            utterances = []
            labels = []
            for line in dialogue:
                utterances.append(clean_str(line['utterance']))
                labels.append(line['emotion'])
            for ii in range(len(utterances)):
                sub_dialogue = utterances[max(0, ii-num_seq + 1):ii+1]
                while len(sub_dialogue) < num_seq:
                    sub_dialogue = ['START'] + sub_dialogue
                sub_dialogues.append(sub_dialogue)
                last_labels.append(labels[ii])
    return sub_dialogues, last_labels


def batch_iter(data, batch_size, num_epochs, shuffle = True):
    """
    Generates a batch iterator for a dataset
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
