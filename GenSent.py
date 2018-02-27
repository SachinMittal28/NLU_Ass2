import random
import nltk
from nltk.corpus import brown
from collections import Counter
from nltk.corpus import gutenberg



def remove_punctuation(corpus):
    cleaned_corpus=[]
    
    punctuations = ['!','(',')','-','[',']','{',';',':',"'",'\\','<','>','.','/','?','~','&',"''",',','--','``','"']
    for sent in corpus :
        sent1=[]
        for word in sent:
            if word not in punctuations:
                sent1.append(word)
        cleaned_corpus.append(sent1)  
    return cleaned_corpus


fieldids_brown = nltk.corpus.brown.fileids()
corpus_brown = remove_punctuation(list(brown.sents(fieldids_brown)))
random.shuffle(corpus_brown)

corpus = corpus_brown




def get_model(training_corpus,n):
    '''returns a tuple of dict objects (unigrams, bigrams, trigrams) that map from n-grams to counts'''
    from  collections import defaultdict
    model = defaultdict(lambda: defaultdict(lambda: 0))
    

    for sentence in training_corpus:
        for ngram in nltk.ngrams(sentence[:-1],n, pad_right=True, pad_left=True,left_pad_symbol='<s>', right_pad_symbol="</s>"):
            model[ngram[:-1]][ngram[-1]] += 1
    # Let's transform the counts to probabilities
    for history_gram in model:
        total_count = float(sum(model[history_gram].values()))
        for next_gram in model[history_gram]:
            model[history_gram][next_gram] /= total_count
    
    return model

n=4
model = get_model(corpus,n)

import random
while(1):
    text = ['<s>']*(n-1)

    sentence_finished = False

    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-(n-1):])].keys():
            accumulator += model[tuple(text[-(n-1):])][word]

            if accumulator >= r:
                text.append(word)
                break

        if text[-(n-1):] == ['</s>']*(n-1):
            sentence_finished = True

    sentence = ' '.join([t for t in text if t not in ['</s>', '<s>']])
    if(len(sentence)>=10):
        print(sentence)
        break

print("\n")
