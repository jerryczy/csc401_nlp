import numpy as np
import sys
import argparse
import os
import json
import re, csv, warnings

# set up dictionaries for search data
bglFile = '/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv'
bgl = {}
with open(bglFile, ) as csv_bgl:
    reader = csv.DictReader(csv_bgl)
    for row in reader:
        if row['WORD'] != '':
            bgl[row['WORD']] = [float(row['AoA (100-700)']), float(row['IMG']), float(row['FAM'])]

warrFile = '/u/cs401/Wordlists/Ratings_Warriner_et_al.csv'
warr = {}
with open(warrFile, ) as csv_warr:
    reader = csv.DictReader(csv_warr)
    for row in reader:
        if row['Word'] != '':
            warr[row['Word']] = [float(row['V.Mean.Sum']), float(row['A.Mean.Sum']), float(row['D.Mean.Sum'])]

# extract feature data
feat_alt = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
feat_center = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
feat_left = np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
feat_right = np.load('/u/cs401/A1/feats/Right_feats.dat.npy')

def id_index(addr):
    count = 0
    lines = []
    with open(addr, 'r') as fd:
        lines = fd.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
    return lines

# extract id index pair into dictionary for faster look up
id_alt = id_index('/u/cs401/A1/feats/Alt_IDs.txt')
id_center = id_index('/u/cs401/A1/feats/Center_IDs.txt')
id_left = id_index('/u/cs401/A1/feats/Left_IDs.txt')
id_right = id_index('/u/cs401/A1/feats/Right_IDs.txt')

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features
    '''
    # get the category of this comment
    feats = np.zeros(29)
    token_sum = 0
    tokenSplit = comment.split()
    # init arrays to store feature data
    aoa = np.zeros(len(tokenSplit))
    img = np.zeros(len(tokenSplit))
    fam = np.zeros(len(tokenSplit))
    vms = np.zeros(len(tokenSplit))
    ams = np.zeros(len(tokenSplit))
    dms = np.zeros(len(tokenSplit))

    for token in range(len(tokenSplit)):
        match = re.match('.*/', tokenSplit[token])
        # debug
        # if not match:
        #     print (tokenSplit[token])
        #     sys.exit(1)
        index = match.span()[1]
        vocab = tokenSplit[token][:index-1]
        tag = tokenSplit[token][index:]
        if len(tag) > 0 and tag[-1] == '\n':
            tag = tag[:-1]
        token_sum += len(vocab)
        # extract data for feature 18-29
        if vocab in bgl:
            aoa[token] = bgl[vocab][0]
            img[token] = bgl[vocab][1]
            fam[token] = bgl[vocab][2]
        if vocab in warr:
            vms[token] = warr[vocab][0]
            ams[token] = warr[vocab][1]
            dms[token] = warr[vocab][2]

        # extract features 1-13
        if tag in ['PRP', 'PRP$']:
            # first person
            if vocab in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']:
                feats[0] += 1
            # second person
            elif vocab in ['you', 'your', 'yours', 'u', 'ur', 'urs']:
                feats[1] += 1
            # third person
            elif vocab in ['he', 'him', 'his', 'she', 'her', 'hers', 'it', \
            'its', 'they', 'them', 'their', 'theirs']:
                    feats[2] += 1
        # coordi    nte conjunction
        elif tag == 'CC':
            feats[3] += 1
        # past tense verb
        elif tag == 'VBD':
            feats[4] += 1
        # future tense verb (partial, the rest outside loop)
        elif vocab in ["'ll", 'gonna', 'will']:
            if vocab != 'will':
                feats[5]+=1
            elif tag == 'MD':
                feats[5]+=1
        # comma
        elif tag == ',':
            feats[6] += 1
        #  multi-character punctuation
        elif tag in ['.'] and len(vocab) > 1:
            feats[7] += 1
        # common noun
        elif tag in ['NN', 'NNS']:
            feats[8] += 1
        # proper noun
        elif tag in ['NNP', 'NNPS']:
            feats[9] += 1
        # adverb
        elif tag in ['RB', 'RBR', 'RBS']:
            feats[10] += 1
        # wh- words 
        elif tag in ['WDT', 'WP', 'WP$', 'WRB']:
            feats[11] += 1
        # slang
        elif tag in ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'roﬂ', \
        'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', \
        'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', \
        'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez', 'f2f', \
        'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz',  'ru', \
        'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml']:
            feats[12] += 1
        # upper case
        elif len(vocab) > 3:
            cap = True
            for c in vocab:
                if c.islower():
                    cap = False
            if cap:
                feats[13] += 1

    # future tense verb (partial, the rest inside loop)
    goingto = 'going/VBG to/TO'
    search = re.search(goingto, comment)
    while search:
        end = search.span()[1]
        next = re.match('.*/', comment[end:])
        if next and comment[end:][next.span()[1]:next.span()[1]+2] == 'VB':
            fest[5]+=1
        search = re.search(goingto, comment[end:])

    newlineSplit = comment.split('\n')
    # average length of setence in tokens
    if len(newlineSplit) > 0:
        feats[14] = len(tokenSplit)/len(newlineSplit)
    # average length of tokens
    if len(tokenSplit) > 0:
        feats[15] = token_sum/len(tokenSplit)
    # number of sentence
    feats[16] = len(newlineSplit)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # average AoA
        feats[17] =  np.mean(aoa)
        # average IMG
        feats[18] =  np.mean(img)
        # average FAM
        feats[19] =  np.mean(fam)
        # standard deviation of AoA
        feats[20] =  np.std(aoa)
        # standard deviation of IMG
        feats[21] =  np.std(img)
        # standard deviation of FAM
        feats[22] =  np.std(fam)
        # average V.Mean.Sum 
        feats[23] =  np.mean(vms)
        # average A.Mean.Sum 
        feats[24] =  np.mean(ams)
        # average D.Mean.Sum 
        feats[25] =  np.mean(dms)
        # standard deviation of V.Mean.Sum 
        feats[26] =  np.std(vms)
        # standard deviation of A.Mean.Sum 
        feats[27] =  np.std(ams)
        # standard deviation of D.Mean.Sum 
        feats[28] =  np.std(dms)

        for i in range(17, 29):
            if feats[i] != feats[i]:
                feats[i] = 0

    return feats

# LIWC/Receptiviti features
def liwc(id, cat):
    liwc = np.zeros(144)
    if cat == 'Left':
        liwc = np.append(feat_left[id_left.index(id)], 0)
    elif cat == 'Center':
        liwc = np.append(feat_center[id_center.index(id)], 1)
    elif cat == 'Right':
        liwc = np.append(feat_right[id_right.index(id)], 2)
    elif cat == 'Alt':
        liwc = np.append(feat_alt[id_alt.index(id)], 3)
    else:
        print('error')
        sys.exit(1)
    return liwc

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))
    for j in range(len(data)):
        # if j%100 == 0:
        #     print(j)
        feature = extract1(data[j]['body'])
        liwcFeat = liwc(data[j]['id'], data[j]['cat'])
        feats[j] = np.append(feature, liwcFeat)

    np.savez_compressed( args.output, feats)
    # print('Done!')
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
    
    main(args)
