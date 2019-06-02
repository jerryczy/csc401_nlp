
from math import log2, exp
from decode import decode
from preprocess import preprocess
from lm_train import lm_train
from log_prob import log_prob
from align_ibm1 import align_ibm1

def BLEU_score(candicate, references, n):
    """
    Compute the LOG probability of a candicate, given a language model and whether or not to
    apply add-delta smoothing
    
    INPUTS:
    candicate :    (string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references:   (list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :           (int) one of 1,2,3. N-Gram level.

    
    OUTPUT:
    bleu_score :  (float) The BLEU score
    """
    c_len = len(candicate)
    len_diff = []
    for sent in references:
        len_diff.append(abs(c_len - len(sent)))
    r = references[len_diff.index(min(len_diff))]

    brevity = len(r) / c_len
    BP = 1
    if brevity >= 1:
        BP = exp(1 - brevity)

    pp = 1
    for i in range(n):
    	pp *= precision(candicate, r, i)
    
    bleu_score = BP * (pp**(1/n))
    return bleu_score

def precision(candicate, reference, order):
	c = candicate.split()
	count = 0
	for i in range(len(c) - order):
		string = ' '.join(c[i:i+order])
		if string in reference:
			count += 1
	return count/len(c)

def read_file(file, language):
    # read French senteces
    sents = []
    with open(file, 'r') as fd:
        for _ in range(25):
            sentence = fd.readline()[:-1]
            sents.append(preprocess(sentence, language))
    return sents


F_FILE = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f'
E_H_FILE = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e'
E_G_FILE = '/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e'

TRAIN_DIR = '/u/cs401/A2_SMT/data/Hansard/Training/'

fre_sents = read_file(F_FILE, 'f')
eng_h_sents = read_file(E_H_FILE, 'e')
eng_g_sents = read_file(E_G_FILE, 'e')

# produce LM
LM = lm_train(TRAIN_DIR, 'e', 'eval_lm')
# print('finish LM training.')

fd = open('task5_data.txt', 'w')
for size in [1, 10, 15, 30]:
    # produce AM
    AM = align_ibm1(TRAIN_DIR, size*1000, 25, 'eval_am')
    # print('finish AM training of size ' + str(size) + 'K.')
    fd.write('AM size: ' + str(size) + 'K.\n')
    for i in range(25):
        eng = decode(fre_sents[i], LM, AM)
        fd.write('sentence ' + str(i+1) + '\n')
        for n in [1, 2, 3]:
            ref = [eng_h_sents[i], eng_g_sents[i]]
            score = BLEU_score(eng, ref, n)
            # print('score of AM size ' + str(size) + 'K with n = ' + str(n) + ' is ' + str(score))
            fd.write('score of order ' + str(n) + ' is ' + str(score) + '\n')
fd.close()
