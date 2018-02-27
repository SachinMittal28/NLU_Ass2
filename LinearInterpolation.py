import random
import nltk


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
    unks=set()
    max_UNK=0
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
        tokens3 = [START] + [START] + tokens0 + [STOP]

        # bigrams
        for bigram in nltk.bigrams(tokens2):
            bigram_c[bigram] += 1

        # trigrams
        for trigram in nltk.trigrams(tokens3):
            trigram_c[trigram] += 1
    
    unigram_c[(START,)] = len(training_corpus)
    bigram_c[(START, START)] = len(training_corpus)
    model = (unigram_c,bigram_c,trigram_c)
    return model 
    

def eval_model(test_corpus, model, log_prob_func,caches):
    '''Returns the perplexity of the model on a specified test set.'''

    log_prob_sum = 0
    word_count = 0

    for sentence in test_corpus:
        prob = eval_sentence(sentence, model, log_prob_func,caches)
        log_prob_sum += prob
        word_count += len(sentence)
        
    average_log_prob = log_prob_sum / word_count
    perplexity = 2**(-average_log_prob)
    return perplexity


def eval_sentence(sentence, model, log_prob_func,caches):
    '''Returns log probability of a sentence and how many tokens were in the sentence.'''

    tokens0 = [token if (token,) in model[0] else UNK for token in sentence[:-1]]
    tokens1 = tokens0 + [STOP]
    tokens2 = [START] + tokens0 + [STOP]
    tokens3 = [START] + [START] + tokens0 + [STOP]
    
    log_prob_sum = 0
    # trigrams
    for n_gram in nltk.trigrams(tokens3):
        next_prob = log_prob_func(n_gram, model,caches)
        log_prob_sum += next_prob

    return log_prob_sum

def main():
    import math
    # returns the log probability of a specified n-gram
    def get_log_prob(n_gram, model, caches):
        if n_gram in caches[2]:
            return caches[2][n_gram]

        # uni-gram part
        if n_gram[2:] in caches[0]:
            unigram_part = caches[0][n_gram[2:]]
        else:
            uni_numer = model[0][n_gram[2:]]
            uni_denom = sum(model[0].values()) - model[0][(START,)]
            unigram_part = 0
            if uni_denom != 0:
                unigram_part = LAMBDA_3 * uni_numer / uni_denom
            caches[0][n_gram[2:]] = unigram_part

        # bi-gram part
        if n_gram[1:] in caches[1]:
            bigram_part = caches[1][n_gram[1:]]
        else:
            bi_numer = model[1][n_gram[1:]]
            bi_denom = model[0][n_gram[2:]]
            bigram_part = 0
            if bi_denom != 0:
                bigram_part = LAMBDA_2 * bi_numer / bi_denom
            caches[1][n_gram[1:]] = bigram_part

        # tri-gram part
        tri_numer = model[2][n_gram]
        tri_denom = model[1][n_gram[1:]]
        trigram_part = 0
        if tri_denom != 0:
            trigram_part = LAMBDA_1 * tri_numer / tri_denom

        prob = trigram_part + bigram_part + unigram_part
        log_prob = math.log(prob, 2)
        caches[2][n_gram] = log_prob
        return log_prob
    

    for n in range (1,5) :
        print("Setting : ",n)
        training_corpus,test_corpus =  get_corpusSetting(n)
        print("\ttraining_corpus = ", len(training_corpus))
        print("\ttest_corpus = ", len(test_corpus))

        tr_corpus = training_corpus[:int(len(training_corpus)*.9)]
        dev_corpus = training_corpus[int(len(training_corpus)*.9):]

        model= get_model(tr_corpus) 
        print("\ttuning hyperparameters λ1,λ2 and λ3 using dev corpus",end= "")
        LAMBDA_1s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        LAMBDA_2s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        perplexityList = dict()
        for i in LAMBDA_1s:
            for j in LAMBDA_2s:
                LAMBDA_1 = i
                LAMBDA_2 = j
                if LAMBDA_1 + LAMBDA_2 < 0.9:
                    LAMBDA_3 = 1 - LAMBDA_1 - LAMBDA_2
                   # print('λ1: ' + str(LAMBDA_1) + ' λ2: ' + str(LAMBDA_2) + ' λ3: ' + str(LAMBDA_3))

                    #start = time.time()
                    caches = (dict(), dict(), dict())

                    perplexity = eval_model(dev_corpus, model, get_log_prob,caches)

                    #print("perplexity  is = ", perplexity)
                    perplexityList[(LAMBDA_1,LAMBDA_2,LAMBDA_3)]=perplexity
                    print("..",end="")

        Lamdas = min(perplexityList, key=perplexityList.get)
        LAMBDA_1=Lamdas[0]
        LAMBDA_2=Lamdas[1]
        LAMBDA_3=Lamdas[2]

        print("\n\n\tlamdas to be used by Linear Interpolation  => λ1=",LAMBDA_1," λ2=",LAMBDA_2," λ3=",LAMBDA_3  )
        model =get_model(training_corpus) 
        perplexity = eval_model(test_corpus, model, get_log_prob,caches)     
        print("\n\tperplexity on test data using λ1=",LAMBDA_1," λ2=",LAMBDA_2," λ3=",LAMBDA_3," is ", perplexity, end = "\n\n\n")




        '''
        model= get_model(training_corpus) 
        perplexity = eval_model(corpus, model, get_log_prob)
        print("perplexity on training_corpus is = ", perplexity)
        '''

    
    
    
    #print("Minimum at LAMBDA1: ",min(perplexityList, key=perplexityList.get)[0], " LAMBDA2: ",min(perplexityList, key=perplexityList.get)[1], " LAMBDA3: ",min(perplexityList, key=perplexityList.get)[2])
    
#calling function main
main()
