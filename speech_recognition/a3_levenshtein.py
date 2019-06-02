import os, sys
import re
from string import punctuation as PUNC
from math import inf as INF
import numpy as np

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
        O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    n = len(r) + 1
    m = len(h) + 1
    # edge cases
    if n == 0:
        return INF, 0, m, 0
    if m == 0:
        return 1.0, 0, 0, n

    R = np.zeros((n, m))
    for i in range(n):
        R[i][0] = i
    for j in range(m):
        R[0][j] = j

    for i in range(1, n):
        for j in range(1, m):
            if r[i-1] == h[j-1]:
                R[i][j] = min(R[i-1][j] + 1, R[i-1][j-1], R[i][j-1] + 1)
            else:
                R[i][j] = min(R[i-1][j] + 1, R[i-1][j-1] + 1, R[i][j-1] + 1)

    sub = 0
    ins = 0
    det = 0
    i = n-1
    j = m-1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and R[i][j] == R[i-1][j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and R[i][j] == R[i-1][j-1] + 1:
            i -= 1
            j -= 1
            sub += 1
        elif i > 0 and R[i][j] == R[i-1][j] + 1:
            i -= 1
            det += 1
        elif j > 0 and R[i][j] == R[i][j-1] + 1:
            j -= 1
            ins += 1

    # senity check
    if R[n-1][m-1] != det + sub + ins:
        print('Error!')
    WER = R[n-1][m-1] / (n-1)

    return WER, sub, ins, det

def file_process(speaker, name):
    path = os.path.join(os.path.join(dataDir, speaker), 'transcripts' + name + '.txt')
    with open(path, 'r') as fd:
        text = fd.readlines()
    return text

def line_pre_process(text):
    # remove header
    text = text[re.match('.*:[A-Z]+ ', text).span()[1]:]
    # remove brackets
    words = text.split()
    processed = []
    for word in words:
        if word[0] not in '<[':
            # remove punctuation
            new_word = ''
            for c in word:
                if c not in PUNC:
                    new_word += c
            if new_word != '':
                processed.append(new_word)
    return processed

if __name__ == "__main__":
    google_wer = []
    kaldi_wer = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            human = file_process(speaker, '')
            google = file_process(speaker, '.Google')
            kaldi = file_process(speaker, '.Kaldi')
            # no reference
            if not human:
                continue
            valid = []
            for i in [google, kaldi]:
                if i:
                    valid.append(1)
                else:
                    valid.append(0)
            for idx, line in enumerate(human):
                ref = line_pre_process(line)
                if valid[0] == 1:
                    hypo = line_pre_process(google[idx])
                    WER, sub, ins, det = Levenshtein(ref, hypo)
                    google_wer.append(WER)
                    data = str(WER) + ' S:' + str(sub) + ', I:' + str(ins) + ', D:' + str(det)
                    print(speaker + ' Google ' + str(idx) + ' ' + data)
                if valid[1] == 1:
                    hypo = line_pre_process(kaldi[idx])
                    WER, sub, ins, det = Levenshtein(ref, hypo)
                    kaldi_wer.append(WER)
                    data = str(WER) + ' S:' + str(sub) + ', I:' + str(ins) + ', D:' + str(det)
                    print(speaker + ' Kaldi ' + str(idx) + ' ' + data)
    # calculate mean and sd
    g_wer = np.asarray(google_wer)
    k_wer = np.asarray(kaldi_wer)
    g_data = 'Google average: '+str(np.mean(g_wer))+', standard deviation: '+str(np.std(g_wer))
    k_data = 'Kaldi average: '+str(np.mean(k_wer))+', standard deviation: '+str(np.std(k_wer))
    print(g_data + '; ' + k_data)
