# -*- coding: utf-8 -*-
import json
import re
import numpy as np

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
