import pickle
import os
from math import log
from collections import Counter
from lm_train import lm_train
from log_prob import log_prob
from preprocess import preprocess

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm.
    We assume that we are implemented P(foreign|english)

    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter :      (int) the maximum number of iterations of the EM algorithm
    fn_AM :         (string) the location to save the alignment model

    OUTPUT:
    AM :            (dictionary) alignment model structure

    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
    is the computed expectation that the foreign_word is produced by english_word.

            LM['house']['maison'] = 0.5
    """
    AM = {}
    data = {} # a dictionary keep track coresponding English-French translation
    # Read training data
    train_corpus = read_hansard(train_dir, num_sentences)
    # print('finish AM reading')
    # count for all possible English-French translation
    for pair in train_corpus:
        data_sent = initialize(pair)
        for key in data_sent:
            # only add unique French word to the list
            if key in data:
                for word in data_sent[key]:
                    if word not in data[key]:
                        data[key].append(word)
            else:
                data[key] = data_sent[key]

    # Initialize AM uniformly
    # Set tcount(f, e) and total(e) to 0 first for reference
    tcount_init = {}
    total_init = {}

    for e_word in data:
        AM[e_word] = {}
        tcount_init[e_word] = {}
        total_init[e_word] = 0
        count = len(data[e_word])
        for f_word in data[e_word]:
            AM[e_word][f_word] = 1/count
            tcount_init[e_word][f_word] = 0
    # print('finish AM initialization')

    # Iterate between E and M steps
    for _ in range(max_iter):
        tcount = tcount_init
        total = total_init
        for pair in train_corpus:
            F = Counter(pair[0].split()[1:-1])
            for f in F:
                denom_c = 0
                E = Counter(pair[1].split()[1:-1])
                for e in E:
                    denom_c += AM[e][f] * F[f]
                for e in E:
                    tcount[e][f] += AM[e][f] * F[f] * E[e] / denom_c
                    total[e] += AM[e][f] * F[f] * E[e] / denom_c
        for e in total:
            for f in tcount[e]:
                AM[e][f] = tcount[e][f] / total[e]

    AM['SENTSTART'] = {'SENTSTART':1}
    AM['SENTEND'] = {'SENTEND':1}
    #Save Model
    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM

# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.

    INPUTS:
    train_dir :     (string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider


    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

    Make sure to read the files in an aligned manner.
    """
    sent_count = 0
    data = []
    for subdir, dirs, files in os.walk(train_dir):
        # assume we only have pairs of .e and .f file with same file name
        file_names = []
        # first preprocess to get array of file name pair
        for file in files:
            if file[:-2] not in file_names:
                file_names.append(file[:-2])
        for file in file_names:
            eng_f = os.path.join(subdir, file + '.e')
            fre_f = os.path.join(subdir, file + '.f')
            data_e = []
            data_f = []
            with open(eng_f, 'r') as f:
                data_e = f.readlines()
            with open(fre_f, 'r') as f:
                data_f = f.readlines()
            assert len(data_e) == len(data_f), \
            "English file and French file have different length!\nFile name: " + file

            for line in range(len(data_e)):
                pre = (preprocess(data_f[line][:-1], 'f'), preprocess(data_e[line][:-1], 'e'))
                data.append(pre)
                sent_count += 1
                if sent_count >= num_sentences:
                    return data
    return data

def initialize(pair):
    """
    INPUT:
    pair: a French-English alignment

    OUTPUT:
    data: a dictionary that use English word as key and track all possible unique coresponding
        French words

    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """
    data = {}
    # remove SENTSTART and SENTEND
    fre = pair[0].split()[1:-1]
    eng = pair[1].split()[1:-1]
    for word in eng:
        if word not in data:
            data[word] = []
        for f_word in fre:
            if f_word not in data[word]:
                data[word].append(f_word)
    return data
