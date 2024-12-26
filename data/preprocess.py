import json
import os
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

nltk.data.path.append(os.path.join(os.path.dirname(__file__), '..', 'nltk'))

nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()



def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0


    return bag
         


