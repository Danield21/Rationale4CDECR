from nltk.stem import PorterStemmer
from thefuzz import fuzz

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
            x[0]: text_1, complete sentence of the first text snippt
            x[1]: trigger_1_abs_start, the start offset of the head lemma of trigger_1
            x[2]: trigger_1_abs_end, the start offset of the head lemma of trigger_2
        Outputs: full head lemma of trigger_1
        Example:
        >>> ecb_aug['extracted_trigger_1'] = ecb_aug[['text_1','trigger_1_abs_start','trigger_1_abs_end']].apply(get_trigger_1,axis=1)
    '''
    text1=x[0]
    start = x[1]
    end = x[2]
    text1_list = text1.split(' ')
    trigger_1 = ' '.join(text1_list[start:end+1])
    return trigger_1

def get_trigger_2(x):
    ''' Extract trigger_2 from text_2, using with df.apply() 
        Inputs: 
            x[0]: text_1, the complete first text snippet with prefix_1, text_1 and suffix_1
            x[1]: text_2, the complete second text snippet with prefix_2, text_2 and suffix_2
            x[2]: trigger_2_abs_start, the absolute start offset of the head lemma of trigger_2
            x[3]: trigger_2_abs_end, the absolute end offset of the head lemma of trigger_2
        Outputs: full head lemma of trigger_2
        Example:
        >>> ecb_aug['extracted_trigger_2'] = ecb_aug[['text_1', 'text_2', 'trigger_2_abs_start','trigger_2_abs_end']].apply(get_trigger_2,axis=1)
    '''
    text1=x[0]
    text2=x[1]
    start = x[2]
    end = x[3]
    text1_text2_list = text1.split(' ') + text2.split(' ')
    trigger_2 = ' '.join(text1_text2_list[start:end+1])
    return trigger_2

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
        Example:
        >>> ecb_ORI = coref_avg_lexical_similarity(ecb_ORI, 'ORI')
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


