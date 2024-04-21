import pandas as pd

def combine_prefix_text_suffix(x):
    ''' Combine the prefix, text, and suffix of two strings, using with df.apply() 
        Inputs: 
            x[0]: prefix of the first string
            x[1]: text of the first string
            x[2]: suffix of the first string
        Outputs: 
            the combined string
    '''        
    prefix=x[0]
    text=x[1]
    suffix=x[2]
    if prefix=='' and suffix!='':
        combined_text=' '.join([text,suffix])
    elif prefix!='' and suffix=='':
        combined_text=' '.join([prefix,text])
    elif prefix!='' and suffix!='':
        combined_text=' '.join([prefix,text,suffix])
    else:
        combined_text=text
    return combined_text

def calculate_sentence_length(x):
    ''' Calculate the length of a sentence, using with df.apply() 
        Inputs: 
            x: a complete sentence
        Outputs: 
            the length of the sentence
    '''
    if x=='':
        return 0
    else:
        return len(x.split(' ')) 


def get_trigger_1_abs_pos(x):
    ''' Calculate the absolute position of trigger_1 in pairwise input, using with df.apply() 
        Inputs: 
            x[0]: trigger_1_rel_start/trigger_1_rel_end. i.e., trigger_1 start/end token offset in text_1
            x[1]: prefix_1_length, i.e., the sentence length of prefix_1
        Outputs: 
            the absolute position of trigger_1
        Example:
        >>>  df['trigger_1_abs_start']=df[['trigger_1_rel_start','prefix_1_length']].apply(get_trigger_1_abs_pos,axis=1)
        >>> df['trigger_1_abs_end']=df[['trigger_1_rel_end','prefix_1_length']].apply(get_trigger_1_abs_pos,axis=1)
    '''
    return x[0]+x[1]



def get_trigger_2_abs_pos(x):
    ''' Calculate the absolute position of trigger_2 in pairwise input, using with df.apply() 
        Inputs: 
            x[0]: combined_text_1_length, i.e., the length of the first text snippet: prefix_1 + text_1 (event occuring sentence) + suffix_1 
            x[1]: prefix_2_length, i.e., the length of prefix_2
            x[2]: trigger_2_rel_start/trigger_2_rel_end,i .e., trigger_2 start/end token offset in text_2
        Outputs: 
            the absolute position of trigger_2
        Example:
        >>> df['trigger_2_abs_start']=df[['combined_text_1_length','prefix_2_length','trigger_2_rel_start']].apply(get_trigger_2_abs_pos,axis=1)
        >>> df['trigger_2_abs_end']=df[['combined_text_1_length','prefix_2_length','trigger_2_rel_end']].apply(get_trigger_2_abs_pos,axis=1)
    '''
    return x[0]+x[1]+x[2]

def organize_data(input_df):
    ''' Collect the details of the data
        Inputs: A dataframe with cols 'prefix_1','text_1','suffix_1', 'trigger_1' ; 'prefix_2','text_2','suffix_2','trigger_2'; 'label'
        Outputs: A dataframe with cols 'text_1','text_2', 'trigger_1_abs_start','trigger_1_abs_end','trigger_2_abs_start','trigger_2_abs_end',
            'total_tokens_num','label','trigger_1', 'trigger_2'
        The outputs data structure contains all we need for training in AD_train_crossencoder.py in src/all_models
    '''
    # combine the prefix, text, and suffix for the 1st/2nd text snippets
    input_df['combined_text_1']=input_df[['prefix_1','text_1','suffix_1']].apply(combine_prefix_text_suffix,axis=1)
    input_df['combined_text_2']=input_df[['prefix_2','text_2','suffix_2']].apply(combine_prefix_text_suffix,axis=1)
    # calculate the length of the 1st/2nd text snippets
    input_df['combined_text_1_length']=input_df['combined_text_1'].apply(lambda x:calculate_sentence_length(x))
    input_df['combined_text_2_length']=input_df['combined_text_2'].apply(lambda x:calculate_sentence_length(x))
    # calculate the length of prefix_1 and prefix_2
    input_df['prefix_1_length']=input_df['prefix_1'].apply(lambda x:calculate_sentence_length(x))
    input_df['prefix_2_length']=input_df['prefix_2'].apply(lambda x:calculate_sentence_length(x))
    # calculate the absolute position of trigger_1 and trigger_2
    input_df['trigger_1_abs_start']=input_df[['trigger_1_rel_start','prefix_1_length']].apply(get_trigger_1_abs_pos,axis=1)
    input_df['trigger_1_abs_end']=input_df[['trigger_1_rel_end','prefix_1_length']].apply(get_trigger_1_abs_pos,axis=1)
    input_df['trigger_2_abs_start']=input_df[['combined_text_1_length','prefix_2_length','trigger_2_rel_start']].apply(get_trigger_2_abs_pos,axis=1)
    input_df['trigger_2_abs_end']=input_df[['combined_text_1_length','prefix_2_length','trigger_2_rel_end']].apply(get_trigger_2_abs_pos,axis=1)
    input_df['total_tokens_num']=input_df[['combined_text_1_length','combined_text_2_length']].apply(get_trigger_1_abs_pos,axis=1)
    # Due to too much information in input_df, we only keep the following cols in output_df
    output_df=input_df[['combined_text_1','combined_text_2','trigger_1_abs_start','trigger_1_abs_end','trigger_2_abs_start','trigger_2_abs_end','total_tokens_num','label','trigger_1', 'trigger_2']]
    output_df=output_df.rename(columns={'combined_text_1': 'text_1', 'combined_text_2': 'text_2'})
    return output_df