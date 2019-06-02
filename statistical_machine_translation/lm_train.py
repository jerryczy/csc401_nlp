from preprocess import preprocess
import pickle
import os, sys

def lm_train(data_dir, language, fn_LM):
    """
    This function reads data from data_dir, computes unigram and bigram counts,
    and writes the result to fn_LM
    
    INPUTS:
    
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    
    OUTPUT
    
    LM      : (dictionary) a specialized language model
    
    The file fn_LM must contain the data structured called "LM", which is a dictionary
    having two fields: 'uni' and 'bi', each of which holds sub-structures which 
    incorporate unigram or bigram counts
    
    e.g., LM['uni']['word'] = 5 # The word 'word' appears 5 times
          LM['bi']['word']['bird'] = 2  # The bigram 'word bird' appears 2 times.
    """
    # init
    language_model = {'uni':{}, 'bi':{}}

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            # only process current language
            if file[-1] != language:
                continue
            full_file = os.path.join(subdir, file)
            # print("Processing " + file)
            data = []
            with open(full_file, 'r') as f:
                data = f.readlines()
            for line in data:
                parse = preprocess(line[:-1], language).split()
                for i in range(len(parse)):
                    if i < len(parse)-1:
                        curr_word, next_word = parse[i:i+2]
                        if curr_word in language_model['uni']:
                            language_model['uni'][curr_word] += 1
                            if next_word in language_model['bi'][curr_word]:
                                language_model['bi'][curr_word][next_word] += 1
                            else:
                                language_model['bi'][curr_word][next_word] = 1
                        else:
                            language_model['uni'][curr_word] = 1
                            language_model['bi'][curr_word] = {next_word:1}

    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return language_model

# if __name__ == '__main__':
#     DIR = '/u/cs401/A2_SMT/data/Hansard/Training/'
#     lm_train(DIR, 'e', 'lm_english')
#     lm_train(DIR, 'f', 'lm_french')
