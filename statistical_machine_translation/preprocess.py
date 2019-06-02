import re, string

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
    
    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language    : (string) either 'e' (English) or 'f' (French)
                   Language of in_sentence
                   
    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    out_sentence = 'SENTSTART'
    sentence = in_sentence.split()
    for word in sentence:
        word = word.lower()
        if language == 'f' and word not in ["d'abord", "d'accord", "d'ailleurs", "d'habitude"]:
            match = re.match("(l')|(qu')|(t')|(j')", word)
            if len(word) == 9 and word[:7] in ["puisqu'", "lorsqu'"] and word[7:] in ['on', 'il']:
                out_sentence += ' ' + word[:7] + ' ' + word[7:]
            elif match:
                out_sentence += ' ' + word[:match.endpos] + ' ' + word[match.endpos:]
            else: # empty string
                out_sentence += ' ' + seperate_puct(word)
        elif language == 'e':
            out_sentence += ' ' + seperate_puct(word)
    out_sentence += ' SENTEND'
    return out_sentence

# seperate leading and punctuation(s)
def seperate_puct(text):
    newtext = ' '
    # seperate leading puncuations
    for i in range(len(text)):
        if text[i] in string.punctuation:
            newtext += text[i] + ' '
        else:
            text = text[i:]
            break

    # seperate the ending punctuation
    for i in range(len(text)-1, -1, -1):
        if text[i] not in string.punctuation:
            newtext += text[:i+1] + ' '
            for j in text[i+1:]:
                newtext += j + ' '
            break
    return newtext.strip()
