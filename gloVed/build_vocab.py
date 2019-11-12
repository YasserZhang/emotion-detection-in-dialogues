# -*- coding: utf-8 -*-
import numpy as np
import pickle
import json
import re
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding = 'utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        except:
            print(line)
    print("Done.",len(model)," words loaded!")
    return model

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'[^\x00-\x7F]+', ' ', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
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
                    utterances.append(clean_str(line['utterance']))
                    labels.append(line['emotion'])
    return utterances, labels
if __name__ == '__main__':
    #modify the addess to where you store the glove file.
    glove_model = loadGloveModel("H:\\glove.840B.300d.txt")
    filenames = ['../EmotionLines/Friends/friends_train.json', '../EmotionLines/Friends/friends_dev.json']
    x_text, y = load_all_data_and_labels(filenames)
    vocabs = {}
    for line in x_text:
        for word in line.split():
            if word in glove_model:
                if word not in vocabs:
                    vocabs[word] = glove_model[word]
            else:
                if word not in vocabs: 
                    vocabs[word] = np.random.normal(0, 1, 300) + glove_model['UNK']
    #add sentence starter 'START'
    vocabs['START'] = glove_model['START']
    with open('vocabs.pickle', 'wb') as f:
        pickle.dump(vocabs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    