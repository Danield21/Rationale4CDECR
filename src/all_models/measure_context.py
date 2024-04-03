import os
import sys
import json
import random
import logging
import argparse
import traceback
from tqdm import tqdm, trange
import _pickle as cPickle

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

parser = argparse.ArgumentParser(description='Training a regressor')

parser.add_argument('--config_path',
                    type=str,
                    help=' The path configuration json file')

args = parser.parse_args()

# Load json config file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

from classes import *


def generate_records_for_sent(sentence_id, sentences, events, window=3):
    mentions = 0
    in_sent_times = 0
    in_sent_locs = 0
    out_sent_times = 0
    out_sent_locs = 0
    sentence = sentences[sentence_id]
    sentence_mentions = sentence.gold_event_mentions if events else sentence.gold_entity_mentions
    lookback = max(0, sentence_id - window)
    lookforward = min(sentence_id + window, max(sentences.keys())) + 1
    sentences = ([sentences[_id] for _id in range(lookback, lookforward)],
                 sentence_id - lookback)
    mention_set = set([])
    coref_set = set([])
    for mention in sentence_mentions:
        mention_set.add(mention.gold_tag)
        mentions += 1
    for ent in sentence.gold_entity_mentions:
        if ent.mention_type == 'TIM':
            in_sent_times += 1
        elif ent.mention_type == 'LOC':
            in_sent_locs += 1
    for c_sentence in sentences[0]:
        if c_sentence == sentence:
            continue
        coref_mentions = c_sentence.gold_event_mentions if events else c_sentence.gold_entity_mentions
        for ment in coref_mentions:
            if ment.gold_tag in mention_set:
                coref_set.add(ment.gold_tag)
        for ent in c_sentence.gold_entity_mentions:
            if ent.mention_type == 'TIM':
                out_sent_times += 1
            elif ent.mention_type == 'LOC':
                out_sent_locs += 1
    return mentions, in_sent_times, in_sent_locs, out_sent_times, out_sent_locs, len(
        coref_set), len(mention_set) - len(coref_set)


def describe_dataset(data_set, events=True):
    docs = [
        document for topic in data_set.topics.values()
        for document in topic.docs.values()
    ]
    with_in_sent_time = 0
    with_in_sent_loc = 0
    with_out_sent_time = 0
    with_out_sent_loc = 0
    with_both_time = 0
    with_both_loc = 0
    with_no_time = 0
    with_no_loc = 0
    added_corefs = 0
    no_corefs = 0
    coref_dict = {}
    for doc in docs:
        sentences = doc.get_sentences()
        for sentence_id in sentences:
            coref_mentions = sentences[
                sentence_id].gold_event_mentions if events else sentences[
                    sentence_id].gold_entity_mentions
            for ment in coref_mentions:
                if ment.gold_tag not in coref_dict:
                    coref_dict[ment.gold_tag] = 1
                else:
                    coref_dict[ment.gold_tag] = coref_dict[ment.gold_tag] + 1
            mentions, in_sent_times, in_sent_locs, out_sent_times, out_sent_locs, out_sent_corefs, no_corefs_sent = generate_records_for_sent(
                sentence_id, sentences, events)
            if in_sent_times > 0 and out_sent_times > 0:
                with_both_time += mentions
            elif in_sent_times > 0:
                with_in_sent_time += mentions
            elif out_sent_times:
                with_out_sent_time += mentions
            else:
                with_no_time += mentions
            if in_sent_locs > 0 and out_sent_locs > 0:
                with_both_loc += mentions
            elif in_sent_locs > 0:
                with_in_sent_loc += mentions
            elif out_sent_locs:
                with_out_sent_loc += mentions
            else:
                with_no_loc += mentions
            added_corefs += out_sent_corefs
            no_corefs += no_corefs_sent
    return {
        "in_time": with_in_sent_time,
        "in_loc": with_in_sent_loc,
        "out_time": with_out_sent_time,
        "out_loc": with_out_sent_loc,
        "both_time": with_both_time,
        "both_loc": with_both_loc,
        "no_time": with_no_time,
        "no_loc": with_no_loc,
        "added_corefs": added_corefs,
        "no_corefs": no_corefs,
        "num_links":
        sum([(num * (num - 1)) // 2 for num in coref_dict.values()])
    }


def main():
    logging.info('Loading training and dev data...')
    with open(config_dict["train_path"], 'rb') as f:
        training_data = cPickle.load(f)
    with open(config_dict["dev_path"], 'rb') as f:
        dev_data = cPickle.load(f)
    with open(config_dict["test_path"], 'rb') as f:
        test_data = cPickle.load(f)

    logging.info('Data has been loaded.')

    print("Train----------------")
    print(describe_dataset(training_data))
    print("Dev----------------")
    print(describe_dataset(dev_data))
    print("Test----------------")
    print(describe_dataset(test_data))


if __name__ == '__main__':
    main()
