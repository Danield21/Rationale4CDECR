import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from thefuzz import fuzz
import ast

def stemize(sentence):
    ''' Stemize the given sentence
        Inputs: 
            a complete sentence 
        Outputs: 
            a sentence after stemming
    '''
    tokens=sentence.split(' ')
    ps = PorterStemmer()
    sentence_after_lemma=' '.join(list(map(lambda x:ps.stem(x),tokens)))
    return sentence_after_lemma

def get_trigger_1(x):
    ''' Extract trigger_1 from text_1, using with df.apply() 
        Inputs: 
            x[0]: complete sentence of the first text snippt
            x[1]: the start offset of the head lemma of trigger_1
            x[2]: the start offset of the head lemma of trigger_2
        Outputs: full head lemma of trigger_1
    >>> ecb_aug['trigger_1'] = ecb_aug[['text_1','trigger_1_abs_start','trigger_1_abs_end']].apply(get_trigger_1,axis=1)
    '''
    text1=x[0]
    start = x[1]
    end = x[2]
    text1_list = text1.split(' ')
    trigger_1 = ' '.join(text1_list[start:end+1])
    return trigger_1

def get_fuzz_ratio(x):
    '''Calculate the fuzz ratio between two strings, using with df.apply() 
        Inputs: 
            x[0]: full head lemma of trigger_1
            x[1]: full head lemma of trigger_2
        Outputs: 
            fuzz ratio between two head lemmas
    '''
    given_hl=x[0].lower()
    detect_hl=x[1].lower()
    fuzz_ratio=fuzz.ratio(stemize(given_hl),stemize(detect_hl))
    return fuzz_ratio

def coref_avg_lexical_similarity(df, data_type):
    """Calculate the average lexical similarity between triggers in coreferential cases.  
       Add a new column 'triggers_lexical_similarity' to the dataset.
       When triggers_lexical_similarity>=80, it is lexically-similar; otherwise, lexically
       -divergent.
       Inputs:
            df: dataset in dataframe structure
            data_type: the name of the dataset
       Outputs:
            Original dataset with a new column 'triggers_lexical_similarity'
            Print lexical similarity information for coreferential cases of the dataset
    """
    if 'triggers_lexical_similarity' not in df.keys():
        assert 'trigger_1' in df.keys() and 'trigger_2' in df.keys(), 'plz check the dataset, trigger_1 or trigger_2 is missing'
        df['triggers_lexical_similarity'] = df[['trigger_1','trigger_2']].apply(get_fuzz_ratio,axis=1)
    else: 
        pass
    assert 'label' in df.keys(), 'plz check the dataset, the ground_truth label is missing'
    coref_data = df[df.label== 1.0]
    avg_lex_sim = coref_data['triggers_lexical_similarity'].mean()
    lex_sim_proportion = len(df[(df.label == 1.0)&(df.triggers_lexical_similarity>=80)])/len(coref_data)
    print(f'Avg triggers lexical similarity for {data_type} coreferential cases is: {avg_lex_sim:.2f}, lexically-similar ones takes {100*lex_sim_proportion:.2f}%.')
    return df



