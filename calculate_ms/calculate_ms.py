import numpy as np
from moverscore_v2 import get_idf_dict, word_mover_score
from collections import defaultdict
from typing import List
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import os

Data_path={
    'CAD': 'ablation_study/CAD.csv',
    'TIA': 'ablation_study/TIA.csv',
    'CIA': 'ablation_study/CIA.csv',
    'TCAD': 'ablation_study/TCAD.csv',
}

def sentence_score_batch(hypothesis: List[str], references: List[str]):
    # https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/example.py
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    # hypothesis = [hypothesis] * len(references)
    sentence_score = 0
    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    return scores
def create_data_for_moverscore(AD_df):
    ref=AD_df[AD_df.index%3==0]
    ref_2=ref.copy(deep=True)
    ref_3=ref.copy(deep=True)
    ref_repeat=pd.concat([ref,ref_2,ref_3]).sort_index().reset_index()
    del ref_repeat['index']
    Only_AD_ref=ref_repeat[ref_repeat.index%3!=0]
    Only_AD=AD_df[AD_df.index%3!=0]
    Only_AD['combined_text']=Only_AD[['text_1','text_2']].apply(lambda x: x[0]+' '+x[1],axis=1)
    Only_AD_ref['combined_text']=Only_AD_ref[['text_1','text_2']].apply(lambda x: x[0]+' '+x[1],axis=1)
    return Only_AD, Only_AD_ref

def calculate_full_moverscore(data_name,batch_size, sample_num='all', seed=42):
    all_ms = []
    ms_sum = 0 
    AD_df=pd.read_csv(Data_path[data_name],index_col=0)
    Only_AD, Only_AD_ref=create_data_for_moverscore(AD_df)
    if sample_num =='all':
        print('computing moverscore for the full augmented dataset')
        hyps=list(Only_AD['combined_text'])
        refs=list(Only_AD_ref['combined_text'])  
    else:
        # if not isinstance(sample_num): 
        #     rasie Exception('sample_num should be int type')
        # if sample_num < 0 or sample_num > len(Only_AD_ref):
        #     rasie Exception(f'Plz make sure sample_num be in range [0, {len(Only_AD_ref)}]')
        hyps=list(Only_AD.sample(n=sample_num,random_state=seed)['combined_text'])
        refs=list(Only_AD_ref.sample(n=sample_num,random_state=seed)['combined_text'])
    
    for i in tqdm(range(0, len(hyps),batch_size)): #按照batch进行迭代
        batch_hypotheses = hyps[i:i + batch_size]
        batch_references = refs[i:i + batch_size]
        scores_for_batch=sentence_score_batch(batch_hypotheses,batch_references)
        all_ms.append(scores_for_batch)
            # ms_sum = ms_sum + np.sum(scores_for_batch)
            # computed_exp_num = i + batch_size
            # computed_ms_avg = ms_sum / computed_exp_num
            # print(f'Avg MoverScore {computed_ms_avg} on {computed_exp_num} examples')
        # except:
        #     with open(os.path.join('outputs/CAD2','moverscore_results'),'wb') as f:
        #         pickle.dump(all_ms,f)
    # run=run+1
    res=np.concatenate(all_ms)
    #print(res)
    print('Avg ms: ',np.mean(res))
    return res

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # assign data name
    parser.add_argument('-data','--data_name',type=str, default='CAD', help='Calculate moverscore for ...')
    # assign batch size
    parser.add_argument('-bz','--batch_size',type=int, default=200, help='Batch size for computing ms')
    # Assign output_dir
    parser.add_argument('-outdir', '--output_dir', help='Dir of the saved reuslts', default=None, required=False)
    
    #parser.add_argument('--gpu_num',type=int, default=0, help='A single GPU number')
    args = vars(parser.parse_args())
    res=calculate_full_moverscore(args['data_name'],args['batch_size'], sample_num='all')
    with open(os.path.join(args['output_dir'],'moverscore_results'),'wb') as f:
        pickle.dump(res,f)
    print('save the ms for {} successfully!'.format(args['data_name']))