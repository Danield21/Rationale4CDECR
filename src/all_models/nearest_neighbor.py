import argparse
import pandas as pd
from coarse import *
import faiss
import numpy as np
from scipy import stats
import os
import sys
import torch
from tqdm import tqdm
import _pickle as cPickle
from transformers import RobertaTokenizer
from typing import Iterator, Iterable, List, Union


for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))
from classes import *

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# print('Loading fixed dev set')
# ecb_dev_set=pd.read_pickle('data_split/pairs_generated_dev_data.pkl')
# print('Loading fixed test set')
# ecb_test_set=pd.read_pickle('data_split/pairs_generated_test_data.pkl')

# nn_generated_fixed_eval_pairs={
#     'ecb':
#         {
#             'dev':ecb_dev_set,
#             'test':ecb_test_set
#             },
#     'fcc':
#         {
#             'dev':None,
#             'test':None
#         },
#     'gvc':
#         {
#             'dev':None,
#             'test':None
#         }
# }



def dataset_to_docs(dataset):
    docs = [
        document for topic in dataset.topics.values()
        for document in topic.docs.values()
    ]
    return docs


def generate_singleton_set(docs):
    clusters = {}
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            for mention in sentences[sentence_id].gold_event_mentions:
                if mention.gold_tag not in clusters:
                    clusters[mention.gold_tag] = [mention.mention_id]
                else:
                    clusters[mention.gold_tag].append(mention.mention_id)
    singletons = [
        mention for cluster_id in clusters for mention in clusters[cluster_id]
        if len(clusters[cluster_id]) == 1
    ]
    return set(singletons)


def build_mention_reps(docs, model, events=True, remove_singletons=False):
    processed_dataset = []
    labels = []
    mentions = []
    label_vocab_size = 0
    singleton_set = set()
    if remove_singletons:
        singleton_set = generate_singleton_set(docs)
    print(len(singleton_set))
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            sentence = sentences[sentence_id]
            #得到该sentence中的所有event mentions
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            if len(sentence_mentions) == 0:
                continue
            #对于某个事件mention的表征集成该mention前5句和后5句的信息
            lookback = max(0, sentence_id - 5)
            lookforward = min(sentence_id + 5, max(sentences.keys())) + 1
            tokenization_input = ([
                sentences[_id] for _id in range(lookback, lookforward)
            ], sentence_id - lookback)#构成了一个11句左右的句子discourse
            tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
                tokenization_input[0], tokenizer, tokenization_input[1])
            #tokenized_sentence为roberta的tokenizer分词后得到的embedding
            #通过训练好的candidate generator model获取对应的sentence embedding
            sentence_vec = model.get_sentence_vecs(
                torch.tensor([tokenized_sentence]).to(model.device))
            for mention in sentence_mentions:#遍历该sentence_mentions中的所有mention
                #if remove_singletons and mention in singleton_set:
                if remove_singletons and mention.mention_id in singleton_set:
                    continue
                #得到trigger的start_piece
                start_piece = torch.tensor([[
                    tokenization_mapping[sent_offset + mention.start_offset][0]
                ]])
                #得到trigger的end_piece
                end_piece = torch.tensor([[
                    tokenization_mapping[sent_offset + mention.end_offset][-1]
                ]])
                #input sentence vec, trigger start piece, trigger end piece.得到event mention embedding
                mention_rep = model.get_mention_rep(
                    sentence_vec, start_piece.to(model.device),
                    end_piece.to(model.device))
                processed_dataset.append(mention_rep.detach().cpu().numpy()[0])#将gpu处理好的mention_rep,传到spu中
                labels.append((mention.mention_str, mention.gold_tag))
                mentions.append(mention)

    return np.concatenate(processed_dataset, axis=0), labels, mentions


def build_cluster_rep(cluster, model, docs):
    cluster_rep = []
    for mention in cluster.mentions.values():
        sentence = docs[mention.doc_id].get_sentences()[mention.sent_id]
        tokenized_sentence, tokenization_mapping, sent_offset = tokenize_and_map(
            [sentence], tokenizer, 0)
        sent_rep = model.get_sentence_vecs(
            torch.tensor([tokenized_sentence]).to(model.device))
        start_piece = torch.tensor(
            [[tokenization_mapping[mention.start_offset][0]]])
        end_piece = torch.tensor(
            [[tokenization_mapping[mention.end_offset][-1]]])
        mention_rep = model.get_mention_rep(sent_rep,
                                            start_piece.to(model.device),
                                            end_piece.to(model.device))
        cluster_rep.append(mention_rep)
    return torch.cat(cluster_rep, dim=0).mean(dim=0).detach().cpu().numpy()


def build_cluster_reps(clusters, model, docs):
    cluster_reps = []
    sent_reps = {}
    for cluster in clusters:
        cluster_rep = build_cluster_rep(cluster, model, docs)
        cluster_reps.append(cluster_rep)
    return np.concatenate(cluster_reps, axis=0)


def mean_reciprocal_rank(rs):
    '''
    平均倒数排序值：某一个mention的邻域内聚类情况越好，rank指越接近低；否则，rank指越接近1（5个neighbour全错，则rank=1）
    '''
    #np.asarray(r).nonzero()得到r中所有非0元素的索引
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    '''
    input：某个mention邻域内的预测结果 r
    '''
    r = np.asarray(r) != 0 #将r中的1-->true; 0-->false
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def nn_cluster_pairs(clusters, model, docs, k=10):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = []
    for i, cluster in enumerate(clusters):
        nearest_neighbor_indexes = I[i][1:]
        nearest_neighbors = [(cluster, clusters[j])
                             for j in nearest_neighbor_indexes]
        pairs.extend(nearest_neighbors)
    return pairs


def create_cluster_index(clusters, model, docs):
    with torch.no_grad():
        vectors = build_cluster_reps(clusters, model, docs)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index


def create_mention_index(docs, model):
    with torch.no_grad():
        vectors = build_mention_reps(docs, model)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    return index

#-----------------------------------------------------------------------------#
#基于faiss
#np.random.seed(5)
#faiss.omp_set_num_threads(1)
def nn_generate_pairs(data, model, k=10, events=True, remove_singletons=False):
    vectors, labels, mentions = build_mention_reps(
        data, model, events, remove_singletons=remove_singletons)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = set()
    for i, mention in enumerate(mentions):
        nearest_neighbor_indexes = I[i]
        nearest_neighbors = set()
        for nn_index, j in enumerate(nearest_neighbor_indexes):
            if mention.mention_id != mentions[j].mention_id:
                nearest_neighbors.add(frozenset([mention, mentions[j]]))
        pairs = pairs | nearest_neighbors
    return pairs

#基于numpy
def nn_generate_pairs_numpy(data, model, k=10, events=True, remove_singletons=False):
    vectors, labels, mentions = build_mention_reps(
        data, model, events, remove_singletons=remove_singletons)
    #index = faiss.IndexFlatIP(1536)
    #index.add(vectors)
    #D, I = index.search(vectors, k + 1)
    similarity_matrix = np.dot(vectors, vectors.T)
    sorted_neighbor_indexes = np.argsort(-similarity_matrix)[:,:k+1]
    pairs = set()
    for i, mention in enumerate(mentions):
        nearest_neighbor_indexes = sorted_neighbor_indexes[i]
        nearest_neighbors = set()
        for nn_index, j in enumerate(nearest_neighbor_indexes):
            if mention.mention_id != mentions[j].mention_id:
                nearest_neighbors.add(frozenset([mention, mentions[j]]))
        pairs = pairs | nearest_neighbors
    return pairs

# def nn_generate_pairs(data, model, k=10, events=True, remove_singletons=False):
#     '''
#     以set container存放各个mention的nearest_neighbors
#     '''
#     vectors, labels, mentions = build_mention_reps(
#         data, model, events, remove_singletons=remove_singletons)
#     index = faiss.IndexFlatIP(1536)
#     index.add(vectors)
#     D, I = index.search(vectors, k + 1)
#     pairs = set()
#     for i, mention in enumerate(mentions):
#         nearest_neighbor_indexes = I[i]
#         nearest_neighbors = set()
#         for nn_index, j in enumerate(nearest_neighbor_indexes):
#             if mention.mention_id != mentions[j].mention_id:
#                 nearest_neighbors.add(frozenset([mention, mentions[j]]))
#         pairs = pairs | nearest_neighbors
#     return pairs, mentions

def nn_generate_mention_neighbors(
    data,
    model,
    k=10,
    events=True,
    remove_singletons=False
):
    '''
    以list container存放各个mention的nearest_neighbors
    '''
    vectors, labels, mentions = build_mention_reps(
        data, model, events, remove_singletons=remove_singletons
    )
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    pairs = list()
    for i, mention in enumerate(mentions):
        nearest_neighbor_indexes = I[i]
        nearest_neighbors = list()
        for nn_index, j in enumerate(nearest_neighbor_indexes):
            if mention.mention_id != mentions[j].mention_id:
                nearest_neighbors.append([mention, mentions[j]])
        pairs.extend(nearest_neighbors)
    return pairs, mentions



# def nn_generate_pairs(data, model, k=10, events=True, remove_singletons=False):
#     vectors, labels, mentions = build_mention_reps(
#         data, model, events, remove_singletons=remove_singletons)
#     index = faiss.IndexFlatIP(1536)
#     index.add(vectors)
#     D, I = index.search(vectors, k + 1)
#     #pairs = set()
#     pairs=[]
#     for i, mention in enumerate(mentions):
#         nearest_neighbor_indexes = I[i]
#         #nearest_neighbors = set()
#         nearest_neighbors = []
#         #print('Here is {}_th mention'.format(i))
#         #neighbour_num=0
#         for nn_index, j in enumerate(nearest_neighbor_indexes):
#             if mention.mention_id != mentions[j].mention_id:
#         #        neighbour_num=neighbour_num+1
#                 #nearest_neighbors.add(frozenset([mention, mentions[j]]))
#                 nearest_neighbors.append([mention, mentions[j]])
            
#         #        print('Neighbour ',neighbour_num)
#         #        print('trigger_1:',mention.get_tokens())
#         #        print('trigger_2:',mentions[j].get_tokens())
#         #        print('--------------------')
#         #print('--------delimiter---------')
                
#         pairs.extend(nearest_neighbors)
    
#     return pairs
#-----------------------------------------------------------------------------#

#----------------------------------------------#
#我们提出的采样方法：对于每个mention，取5个难分的negative samples；以及5个难分的positive samples
def retrieve_hard_pairs(data, model, k=10, events=True, remove_singletons=False,is_train=True):
    vectors, labels, mentions = build_mention_reps(
        data, model, events, remove_singletons=remove_singletons)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    D, I = index.search(vectors, k + 1)
    hard_coref_pairs = set()
    hard_non_coref_pairs = set()
    all_hard_pairs=set()
    
    if is_train==True:
        for i, mention in enumerate(mentions):#遍历3808个mentions
            nearest_neighbor_indexes = I[i]#得到第i个mention的所有邻居,相似度从高到低排序
            reversed_nearest_neighbor_indexes=list(reversed(nearest_neighbor_indexes))[1:]
            nearest_neighbors_coref = set()
            nearest_neighbors_non_coref = set()
            count_1=0
            count_2=0
            #Collect hard non coref examples
            for nn_index, j in enumerate(nearest_neighbor_indexes):#
                if count_1==k:
                    break
                else:
                    if mention.mention_id != mentions[j].mention_id and mention.gold_tag!=mentions[j].gold_tag:
                        nearest_neighbors_non_coref.add(frozenset([mention, mentions[j]]))
                        count_1=count_1+1
                    else:
                        continue
            #Collect hard coref examples            
            for fn_index, fn_j in enumerate(reversed_nearest_neighbor_indexes):#
                if count_2==k:
                    break
                else:
                    if mention.mention_id != mentions[fn_j].mention_id and mention.gold_tag==mentions[fn_j].gold_tag:
                        nearest_neighbors_coref.add(frozenset([mention, mentions[fn_j]]))
                        count_2=count_2+1
                    else:
                        continue   
            all_hard_pairs=(all_hard_pairs|nearest_neighbors_non_coref)|nearest_neighbors_coref
            hard_non_coref_pairs = hard_non_coref_pairs | nearest_neighbors_non_coref
            hard_coref_pairs = hard_coref_pairs | nearest_neighbors_coref        
         
    elif is_train==False:
        for i, mention in enumerate(mentions):#遍历3808个mentions
            nearest_neighbor_indexes = I[i]#得到第i个mention的所有邻居,相似度从高到低排序
            reversed_nearest_neighbor_indexes=list(reversed(nearest_neighbor_indexes))[1:]
            nearest_neighbors_coref = set()
            nearest_neighbors_non_coref = set()
            count_1=0
            count_2=0
            #Collect hard non coref examples
            for nn_index, j in enumerate(nearest_neighbor_indexes):#
                if count_1==k:
                    break
                else:
                    if mention.mention_id != mentions[j].mention_id:
                        nearest_neighbors_non_coref.add(frozenset([mention, mentions[j]]))
                        count_1=count_1+1
                    else:
                        continue
            #Collect hard coref examples            
            for fn_index, fn_j in enumerate(reversed_nearest_neighbor_indexes):#
                if count_2==k:
                    break
                else:
                    if mention.mention_id != mentions[fn_j].mention_id:
                        nearest_neighbors_coref.add(frozenset([mention, mentions[fn_j]]))
                        count_2=count_2+1
                    else:
                        continue           
            all_hard_pairs=(all_hard_pairs|nearest_neighbors_non_coref)|nearest_neighbors_coref
            hard_non_coref_pairs = hard_non_coref_pairs | nearest_neighbors_non_coref
            hard_coref_pairs = hard_coref_pairs | nearest_neighbors_coref
    return all_hard_pairs,hard_non_coref_pairs,hard_coref_pairs
#-------------------------------------------------------#


def nn_eval(eval_data, model, k=5):
    '''
    eval_data：待评价数据（train_data/eval_data/test_data）
    model: bi-encoder模型
    k=5的情形
    '''
    vectors, labels, _ = build_mention_reps(dataset_to_docs(eval_data),
                                            model,
                                            events=True)
    index = faiss.IndexFlatIP(1536)
    index.add(vectors)
    # Add 1 since the first will be identity
    D, I = index.search(vectors, k + 1)
    relevance_matrix = []
    tp = 0
    precision = 0
    singletons = 0
    #[[labels[i] for i in row] for row in I]中的每一个元素是一个list，这个list保存了某个mention对应本身，以及K个neighbours
    #的事件标签信息。eg:（‘riots’，‘ACT17819737684267059’）
    for results in [[labels[i] for i in row] for row in I]:
        original_str, true_label = results[0]#original_str, true_label 保存了该mention本身的head lemma和gold_tag
        if "Singleton" in true_label:#若当前mention是个"Singleton"
            singletons += 1#singletons计数+1
            continue
        matches = results[1:]#所有neighbours对应的信息
        relevance = [label == true_label for _, label in matches]#看neighbours和当前label的匹配情况，匹配则返回1，不匹配则返回0
        num_correct = np.sum(relevance)#看看仅仅通过聚类，一个neighbours内能分对几个
        precision += num_correct / k#可以理解为mention邻域内的k个neighbours都预测为与mention同类别。于是可以计算出该mention对应的precision
        if num_correct >= 1:#只要邻域中的5个neighbours至少有一个预测正确
            tp += 1  #tp计数加一
        relevance_matrix.append(relevance)
    
    #分别得到recall，mean_reciprocal_rank, mean_average_precision,mean_precision_k四个评价指标
    return (tp / float(len(I) - singletons),
            mean_reciprocal_rank(relevance_matrix),
            mean_average_precision(relevance_matrix),
            precision / float(len(I) - singletons))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Doing a Nearest Neighbor Eval of Dense Space Encoder')

    parser.add_argument('--dataset', type=str, help='Dataset to Evaluate on')
    parser.add_argument('--model', type=str, help='Model to Evaluate')
    parser.add_argument('--cuda',
                        dest='use_cuda',
                        action='store_true',
                        help='Use CUDA/GPU for prediction')
    parser.set_defaults(use_cuda=False)

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    with open(args.dataset, 'rb') as f:
        eval_data = cPickle.load(f)
    with open(args.model, 'rb') as f:
        params = torch.load(f)
        model = EncoderCosineRanker("cuda:0")
        model.load_state_dict(params)
        model.eval()
    model.device = torch.device("cuda:0" if args.use_cuda else "cpu")
    model = model.to(model.device)
    recall, mrr, maP, mean_precision_k = nn_eval(eval_data, model)
    tqdm.write(
        "Recall: {:.6f} - MRR: {:.6f} - MAP: {:.6f} - Mean Precision @ K: {:.6f}"
        .format(recall, mrr, maP, mean_precision_k))
