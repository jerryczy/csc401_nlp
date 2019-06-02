from preprocess import preprocess
from lm_train import lm_train
from math import log2
import sys

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
    apply add-delta smoothing

    INPUTS:
    sentence :    (string) The PROCESSED sentence whose probability we wish to compute
    LM :        (dictionary) The LM structure (not the filename)
    smoothing : (boolean) True for add-delta smoothing, False for no smoothing
    delta :     (float) smoothing parameter where 0<delta<=1
    vocabSize :    (int) the number of words in the vocabulary

    OUTPUT:
    log_prob :    (float) log probability of sentence
    """

    # senity check
    if smoothing and (delta > 1 or delta < 0):
        sys.exit("invalid delta value(0 < delta <= 1)")
    elif not smoothing:
        delta = 0

    prob = 1
    words = sentence.split()
    for i in range(len(words)-1):
        curr_word = words[i]
        next_word = words[i+1]
        count_uni = 0
        count_bi = 0

        if curr_word in LM['uni']:
            count_uni = LM['uni'][curr_word]
            if next_word in LM['bi'][curr_word]:
                count_bi = LM['bi'][curr_word][next_word]

        if not smoothing and (count_uni == 0 or count_bi == 0):
            prob += float('-inf')
        else:
            prob += log2((count_bi + delta) / (count_uni + delta * vocabSize))

    return prob
