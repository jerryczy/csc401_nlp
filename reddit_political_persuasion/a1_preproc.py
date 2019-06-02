import sys
import argparse
import os
import json
import html, string, re
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

indir = '/u/cs401/A1/data/'
abbFile = '/u/cs401/Wordlists/abbrev.english'
stopFile = '/u/cs401/Wordlists/StopWords'
nlp = spacy.load('en', disable=['parser', 'ner']) 

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    # modified initial modComm, so that modComm variable can be use as input in step 2-10
    modComm = comment

    # remove newline
    if 1 in steps:
        newcomment = ''
        for c in comment:
            if c != '\n':
                newcomment += c
            else:
                newcomment += ' '
        modComm = newcomment

    # remove html character
    if 2 in steps:
        modComm = html.unescape(modComm)

    # remove url
    if 3 in steps:
        splitcomment = modComm.split()
        newcomment = ''
        for piece in splitcomment:
            newcomment += ' ' + matchWeb(piece).strip()
        modComm = newcomment.strip()

    # splite punctuation
    if 4 in steps:
        newcomment = ''
        doubleQuote = '".*"'
        text = modComm
        search = re.search(doubleQuote, text)
        while search:
            (indexStart, indexEnd) = search.span()
            newcomment += ' ' + wordProcess(text[:indexStart - 1])
            partition = wordProcess(text[indexStart+1:indexEnd-1])
            if len(partition) > 0 and partition[0] in string.punctuation:
                newcomment += ' "' + partition
            elif len(partition) > 0:
                newcomment += ' " ' + partition
            if len(partition) > 0 and partition[-1] in string.punctuation:
                newcomment += '"'
            elif len(partition) > 0:
                newcomment += ' "'
            text = text[indexEnd:]
            search = re.search(doubleQuote, text)
        if len(text) > 0:
            newcomment += ' ' + wordProcess(text)
        modComm = newcomment.strip()

    # split clitics
    if 5 in steps:
        splitcomment = modComm.split()
        newcomment = ''
        for piece in splitcomment:
            if len(piece) > 1 and re.search("'", piece) and piece[0] not in string.punctuation:
                for i in range(len(piece)):
                    if piece[i] == "'":
                        newcomment += ' ' + piece[:i] + ' ' + piece[i:]
                        break
            else:
                newcomment += ' ' + piece
        modComm = newcomment.strip()

    # tagging
    if 6 in steps:
        newcomment = ''
        splitcomment = modComm.split()
        doc = spacy.tokens.Doc(nlp.vocab, words=splitcomment)
        doc = nlp.tagger(doc)
        for token in doc:
            newcomment += ' ' + token.text + '/' + token.tag_
        modComm = newcomment.strip()

    # remove stop word
    if 7 in steps:
        stop = []
        with open(stopFile, 'r') as f:
            stop = f.readlines()
        # remove newline character in stop
        for i in range(len(stop)):
            stop[i] = stop[i][:-1]
        newcomment = ''
        splitcomment = modComm.split()
        for piece in splitcomment:
            if re.match('.*/', piece).group()[:-1].lower() not in stop:
                newcomment += ' ' + piece
        modComm = newcomment.strip()

    # lemmalization
    if 8 in steps:
        newcomment = ''
        splitcomment = modComm.split()
        tag = []
        for i in range(len(splitcomment)):
            piece = splitcomment[i]
            splitcomment[i] = re.search('.*/', piece).group()[:-1]
            tag.append(re.search('/.*', piece).group()[1:])
        lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        for i in range(len(splitcomment)):
            lemmas = lemmatizer(splitcomment[i], unitag(tag[i]))
            newcomment += ' ' + lemmas[0] + '/' + tag[i]
        modComm = newcomment.strip()

    # add newline character after each sentence
    if 9 in steps:
        newcomment = ''
        sentence = './\.'
        text = modComm
        search = re.search(sentence, text)
        while search:
            indexEnd = search.span()[1]
            newcomment += text[:indexEnd] + '\n'
            text = text[indexEnd:]
            search = re.search(sentence, text)
        if len(text) > 0:
            newcomment += text
        modComm = newcomment.strip()

    # lower case
    if 10 in steps:
        newcomment = ''
        splitcomment = modComm.split()
        tag = []
        for i in range(len(splitcomment)):
            piece = splitcomment[i]
            splitcomment[i] = re.match('.*/', piece).group()[:-1]
            tag.append(re.search('/.*', piece).group()[1:])
        for i in range(len(splitcomment)):
            newcomment += ' ' + splitcomment[i].lower() + '/' + tag[i]
            if tag[i] == '.':
                newcomment += '\n'
        modComm = newcomment.strip()
        
    return modComm

def matchWeb(text):
    comment = ''
    webAddrWithQuote = [re.match('("http)|("www)', text), re.match("('http)|('www)", text)]
    if re.match('(http)|(www)', text):
        endline = text[-1]
        # endline character might be /, which is part of valid web address
        if endline in string.punctuation and endline != '/':
            comment += text[-1]
    # web address might contains in quotation
    elif webAddrWithQuote[0] or webAddrWithQuote[1]:
        # only one element in webAddrWithQuote can be true
        for match in webAddrWithQuote:
            if match and match.endpos > len(text):
                # if there are any characters after the quotation mark, it has to ba a punctuation
                comment += text[match.endpos:]
                break
    # not a web address, assuming there will always a space before address
    else:
        comment += ' ' + text
    return comment

def wordProcess(text):
    abb = []
    with open(abbFile, 'r') as f:
        abb = f.readlines()
    # remove newline character in abb
    for i in range(len(abb)):
        abb[i] = abb[i][:-1]

    comment = ''
    if len(text) == 0:
        return comment
    if text[0] == '"':
        comment += '"'
        text = text[1:]
    for item in text.split():
        if item in abb:
            comment += ' ' + item
        elif item[0] in string.punctuation or item[-1] in string.punctuation and item[-1] != "'":
            comment += ' ' + seperatPunc(item).strip()
        # punctuation in the middle of word would indicate apostrophes
        # else a word without any punctuation
        else:
            comment += ' ' + item
    return comment.strip()

def seperatPunc(text):
    comment = ' '
    for i in range(len(text)):
        if text[i] not in string.punctuation:
            comment += text[:i] + ' '
            text = text[i:]
            break

    for i in range(len(text)-1, -1, -1):
        if text[i] not in string.punctuation:
            comment += text[:i+1] + ' ' + text[i+1:]
            break
    return comment

def unitag(tag):
    uni_tag = ''
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        uni_tag = 'NOUN'
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']: 
        uni_tag = 'VERB'
    elif tag in ['JJ', 'JJR', 'JJS']:
        uni_tag = 'ADJ'
    elif tag in ['#', '$', '.', ',', ':', '(', ')', '"', "'", "''", '``']:
        uni_tag = 'PUNCT'
    else:
        uni_tag = tag
    return uni_tag

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print ("Processing " + fullFile)

            data = json.load(open(fullFile))

            # extract part of data we need
            start = args.ID[0]%(len(data))
            end = start + int(args.max)
            if end > len(data):
                end = end - len(data)
            if start < end:
                data = data[start : end]
            else:
                data = data[start:] + data[:end]
            
            for line in data: 
                j = json.loads(line)
                for key in []:
                    j.pop(key)
                j['cat'] = file
                newbody = preproc1(j['body'])
                j['body'] = newbody
                allOutput.append(j)
    # print('Done!')
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print ("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    main(args)
