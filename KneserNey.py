import random
import nltk

from collections import Counter




START = '<s>'
STOP = '</s>'
DEV = False
UNK="UNK"



def get_corpusSetting(n):
        from nltk.corpus import brown
        from nltk.corpus import gutenberg

        fieldids_brown = nltk.corpus.brown.fileids()
        fieldids_gutenberg = nltk.corpus.gutenberg.fileids()

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
        corpus_brown = remove_punctuation(list(brown.sents(fieldids_brown)))
        random.shuffle(corpus_brown)
        corpus_gutenberg = remove_punctuation(list(gutenberg.sents(fieldids_gutenberg)))
        random.shuffle(corpus_gutenberg)
        corpus_brown_gutenberg = corpus_brown + corpus_gutenberg
        if n==1:
            training_corpus = corpus_brown[:int(len(corpus_brown)*.9)]
            test_corpus = corpus_brown[int(len(corpus_brown)*.9):]
        if n==2:
            
            training_corpus = corpus_gutenberg[:int(len(corpus_gutenberg)*.9)]
            test_corpus = corpus_gutenberg[int(len(corpus_gutenberg)*.9):]
        if n==3:
            training_corpus = corpus_brown[:int(len(corpus_brown)*.9)] + corpus_gutenberg
            test_corpus = corpus_brown[int(len(corpus_brown)*.9):]
        if n==4:
            training_corpus = corpus_brown + corpus_gutenberg[:int(len(corpus_gutenberg)*.9)]
            test_corpus = corpus_gutenberg[int(len(corpus_gutenberg)*.9):]

        return training_corpus,test_corpus

    

def get_model(training_corpus):
    '''returns a tuple of dict objects (unigrams, bigrams, trigrams) that map from n-grams to counts'''
    import collections
    unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)
    unigrams_to_w_to_counts = dict()
    preceding_unique_words = dict()
    max_UNK=0
    unks = set()
    for sentence in training_corpus:
        tokens0 = sentence[:-1]             #removing last '.'
        tokens1 = tokens0 + [STOP]
        tokens2 = [START] + tokens0 + [STOP]
        # unigrams
        for unigram in tokens1:
            unigram_c[(unigram,)] += 1  #unigram_c is unigram count
    '''   '''
    for unigram, count in unigram_c.items():
        if (count == 1 and max_UNK<=1000) :
            unks.add(unigram[0])
            max_UNK+=1

    for word in unks:
        del unigram_c[(word,)]

    unigram_c[(UNK,)] = max_UNK
    
    
    for sentence in training_corpus:
        tokens0 = [token if token not in unks else UNK for token in sentence[:-1]]  #removing last '.'
        tokens1 = tokens0 + [STOP]
        tokens2 = [START] + tokens0 + [STOP]
        # bigrams
        for bigram in nltk.bigrams(tokens2):
            bigram_c[bigram] += 1
            add_n_gram_counts(bigram, unigrams_to_w_to_counts,preceding_unique_words)
    
    unigram_c[(START,)] = len(training_corpus)
    bigram_c[(START, START)] = len(training_corpus)
    model = (unigram_c,bigram_c)
    return model, unigrams_to_w_to_counts,preceding_unique_words
    
def add_n_gram_counts(n_gram, d,d1):
    if n_gram[:-1] not in d:
        d[n_gram[:-1]] = Counter()
    d[n_gram[:-1]][n_gram[-1]] +=1 #No of times that history_ngram follows last word of ngram.

    if n_gram[1:] not in d1:        #n_gram[1:]  #skip first word #n_gram[:-1]  #skip last word
        d1[n_gram[1:]] = set()       
    d1[n_gram[1:]].add(n_gram[0])
    
def eval_model(test_corpus, model, log_prob_func):
    '''Returns the perplexity of the model on a specified test set.'''

    log_prob_sum = 0
    word_count = 0

    for sentence in test_corpus:
        prob = eval_sentence(sentence, model, log_prob_func)
        log_prob_sum += prob
        word_count += len(sentence)
        
    average_log_prob = log_prob_sum / word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


def eval_sentence(sentence, model, log_prob_func):
    '''Returns log probability of a sentence and how many tokens were in the sentence.'''
    tokens0 = [token if (token,) in model[0] else UNK for token in sentence[:-1]]
    tokens1 = tokens0 + [STOP]
    tokens2 = [START] + tokens0 + [STOP]
    
    log_prob_sum = 0
    for n_gram in nltk.bigrams(tokens2):
        next_prob = log_prob_func(n_gram, model)
        log_prob_sum += next_prob

    return log_prob_sum
def main():
    import time
    import math
    # returns the log probability of a specified n-gram
    def get_log_prob(n_gram, model):
        
       
        next_gram = n_gram[1:]
        history_gram = n_gram[:-1]
        
        
        first_term = max(model[1][n_gram]-discount, 0 ) / model[0][history_gram]
        second_term = get_lambda(history_gram)*get_PCONTINUATION(next_gram)
        prob = first_term + second_term
        
        log_prob = math.log(prob, 2)
        return log_prob 
        
    def get_lambda(history_gram):
        '''
        λ(w_i-1) = discount* |{w: C(w_i-1 w)>0}| / ∑_v C(w_i-1 v)
        '''   
        numer = discount*len(unigrams_to_w_to_counts[history_gram].keys())
        denom = sum(unigrams_to_w_to_counts[history_gram].values())
        return numer/denom
        
    def get_PCONTINUATION(next_gram):
        
        numer=len(preceding_unique_words[next_gram])
        denom = len(model[1].keys())
        return numer/denom
    
    for n in range (1,5) :
        print("Setting : ",n)
        training_corpus,test_corpus =  get_corpusSetting(n)
        print("\ttraining_corpus = ", len(training_corpus))
        print("\ttest_corpus = ", len(test_corpus))
        
        tr_corpus = training_corpus[:int(len(training_corpus)*.9)]
        dev_corpus = training_corpus[int(len(training_corpus)*.9):]

        model,unigrams_to_w_to_counts,preceding_unique_words= get_model(tr_corpus) 

        Discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1.0,1.2,1.5,1.7,2.0]
        #tuning Hyperparameter Discount
        print("\t# tuning Hyperparameter 'discount' # ",end = " ")
        perplexityList = {}
        for discount in Discounts:

            #print('Discount: ' + str(discount) )
            #start = time.time()
            caches = (dict(), dict(), dict())

            perplexity = eval_model(dev_corpus, model, get_log_prob)

            #print("perplexity  on Dev  = ", perplexity)
            perplexityList[discount]=perplexity
            print("..",end="")

        bestDiscount = min(perplexityList, key=perplexityList.get)
        discount= bestDiscount
        print("\n\n\tfinal discount after tuning on dev corpus = ", discount)
        model,unigrams_to_w_to_counts,preceding_unique_words= get_model(training_corpus) 
        perplexity = eval_model(test_corpus, model, get_log_prob)           
        print("\n\tperplexity on test data using dicount = ",discount," is ", perplexity,end="\n\n\n")



        '''
        model= get_model(training_corpus) 
        perplexity = eval_model(corpus, model, get_log_prob)
        print("perplexity on training_corpus is = ", perplexity)
        '''



    
    
    
#calling function main
    
main()
