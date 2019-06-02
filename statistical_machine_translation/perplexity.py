from log_prob import log_prob
from preprocess import preprocess
from lm_train import lm_train
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0):
    """
	Computes the preplexity of language model given a test corpus
	
	INPUT:
	
	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
    """
	
    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])
    
    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue
        
        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)
            
            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
    	pp = 2**(-pp/N)
    return pp

#test
TRAIN_DIR = '/u/cs401/A2_SMT/data/Hansard/Training/'
TEST_DIR = '/u/cs401/A2_SMT/data/Hansard/Testing/'
fd = open('task3_data.txt', 'w')
test_LM = lm_train(TRAIN_DIR, "e", "e_temp")
fd.write('English\n')
print('no smoothing.')
fd.write('no smooth: ')
perplexity = preplexity(test_LM, TEST_DIR, "e")
print(perplexity)
fd.write(str(perplexity) + '\n')
for delta in [0.2, 0.4, 0.6, 0.8, 1]:
    test_LM = lm_train(TRAIN_DIR, "e", "e_temp_" + str(delta*5))
    print('smooth with delta ' + str(delta))
    fd.write('delta = ' + str(delta) + ': ')
    perplexity = preplexity(test_LM, TEST_DIR, "e", True, delta)
    print(perplexity)
    fd.write(str(perplexity) + '\n')

test_LM = lm_train(TRAIN_DIR, "f", "f_temp")
fd.write('French\n')
print('no smoothing.')
fd.write('no smooth: ')
perplexity = preplexity(test_LM, TEST_DIR, "f")
print(perplexity)
fd.write(str(perplexity) + '\n')
for delta in [0.2, 0.4, 0.6, 0.8, 1]:
    test_LM = lm_train(TRAIN_DIR, "f", "f_temp_" + str(delta*5))
    print('smooth with delta ' + str(delta))
    fd.write('delta = ' + str(delta) + ': ')
    perplexity = preplexity(test_LM, TEST_DIR, "f", True, delta)
    print(perplexity)
    fd.write(str(perplexity) + '\n')

print('Done!')
fd.close()