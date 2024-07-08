#import packages
import sys
import os
import json
import pickle
import numpy as np
import random
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import _pickle as cPickle
import torch
from transformers import RobertaTokenizer


#import necessary functions
for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))
sys.path.append("/src/shared/")
from fine import *
from nearest_neighbor import nn_generate_pairs, dataset_to_docs

parser = argparse.ArgumentParser(description='Generate mention pairs')
parser.add_argument('--config_path',
                    type=str,
                    help=' The path configuration json file')

parser.add_argument('--dataset',
                    type=str,
                    help='ECB+, FCC or GVC')

parser.add_argument('--data_split',
                    type=str,
                    help='train/dev/test split')

parser.add_argument('--out_dir',
                    type=str,
                    help=' The directory to the output folder')

parser.add_argument('--random_seed',
                    type=int,
                    default=5,
                    help=' Random Seed')

parser.add_argument('--gpu_num',
                    type=int,
                    default=0,
                    help=' A single GPU number')

args = parser.parse_args()
out_dir = args.out_dir
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

if args.gpu_num != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    args.use_cuda = True
else:
    args.use_cuda = False


seed = args.random_seed
random.seed(seed)
np.random.seed(seed)
if args.use_cuda:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

args.dataset_path=config_dict['dataset_path']
args.encoder_model_path=config_dict['encoder_model_path']
args.events_or_not=config_dict['events']
args.mention_pairs_num=config_dict['mention_pairs_num']
args.remove_singletons=config_dict["remove_singletons"]
args.all_positives=config_dict["all_positives"]
args.add_wd_pairs=config_dict["add_wd_pairs"]


#Generate mention pairs
def all_positives(docs, events):
    gold_clusters = defaultdict(lambda: [])
    for doc in docs:
        for sentence in doc.get_sentences().values():
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            for mention in sentence_mentions:
                gold_clusters[mention.gold_tag].append(mention)
    pairs = set()
    for value in gold_clusters.values():
        for i, mention_1 in enumerate(value):
            for mention_2 in value[i + 1:]:
                pairs.add(frozenset([mention_1, mention_2]))
    return pairs


def wd_comparisons(docs, events):
    pairs = set()
    for doc in docs:
        doc_mentions = []
        for sentence in doc.get_sentences().values():
            sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
            doc_mentions.extend(sentence_mentions)
        for mention_1 in doc_mentions:
            for mention_2 in doc_mentions:
                if not mention_1.mention_id == mention_2.mention_id:
                    pairs.add(frozenset([mention_1, mention_2]))
    return pairs


def get_sents(sentences, sentence_id, window=3):
    lookback = max(0, sentence_id - window)
    lookforward = min(sentence_id + window, max(sentences.keys())) + 1
    return ([sentences[_id]
             for _id in range(lookback, lookforward)], sentence_id - lookback)

def generate_mention_pairs(data_set,
                           event_encoder,
                           events=args.events_or_not,
                           k=args.mention_pairs_num,
                           is_train=False):
    docs = dataset_to_docs(data_set) #all docs 
    if is_train:
        docs = docs[:int(len(docs) * 1)]
    pairs = nn_generate_pairs(
        docs,
        event_encoder,
        k=k,
        events=events,
        remove_singletons=args.remove_singletons)
    if args.all_positives and is_train:
        pairs = pairs | all_positives(docs, events)
    if args.add_wd_pairs:
        pairs = pairs | wd_comparisons(docs, events)
    # pairs = list(pairs)
    return  pairs

def main():
    assert args.dataset in ['ecb','fcc','gvc'], 'Not consider such dataset.'
    assert args.data_split in ['train','dev','test'], 'Incorrect data split.'
    assert args.dataset in args.encoder_model_path, f'Use inconsistent event_encoder to {args.dataset}'
    assert args.data_split in args.dataset_path, f'Use inconsistent dataset_path to {args.data_split}'
    
    
    if not os.path.exists(args.dataset_path):
        raise Exception('Please check dataset_path!')
    logging.info('Loading into data...')
    with open(args.dataset_path, 'rb') as f:
        data = cPickle.load(f)
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    #load into encoder model
    logging.info('Loading into event_encoder model...')
    with open(args.encoder_model_path, 'rb') as f_1:
        params = torch.load(f_1)#
        event_encoder = EncoderCosineRanker(device)
        event_encoder.load_state_dict(params)
        event_encoder = event_encoder.to(device).eval()
        event_encoder.requires_grad = False
    logging.info('event_encoder has been loaded.')

    #generate mention pairs
    logging.info(f'Retrieving nearest-${args.mention_pairs_num} event mention pairs')
    pairs=generate_mention_pairs(data,event_encoder,events=args.events_or_not,
                                 k=args.mention_pairs_num)
    #save pickle file for pairs
    if args.data_split == 'train':
        save_folder = os.path.join(args.out_dir, f'{args.dataset}/{args.data_split}/baseline')
    else:
        save_folder = os.path.join(args.out_dir, f'{args.dataset}/{args.data_split}')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_file_path = os.path.join(save_folder, f'{args.data_split}_pairs')
    with open(save_file_path, "wb") as p:
        pickle.dump(pairs, p)
    logging.info(f'Saved retrieved mention pairs')

if __name__ == '__main__':
    main()