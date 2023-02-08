'''
code to measure the controllability of the neurologic decosing.

In order to use this code, you first need to generate the outputs
using the neurologic decoding. You also need to create two files; 
the input file with the seed events and a constraint file which 
each np extracted from a seed event is a line in this file.
Once you create these files (the code is also availebl here), you
can generate the outputs using neurologic decoding.

Below are the set of parameters used for neurologic decoding.

--min_tgt_length 50 --max_tgt_length 100 \
--bs 32 --beam_size 40 --length_penalty 0.2 --ngram_size 3 \
--prune_factor 50 --sat_tolerance 1 --beta 0 --early_stop 0.0 \

To compute controllability you can then use:
python controllability_metric_neurologic_decoding.py --seeds_file=PATH_TO_FILE_WITH_SEED_EVENTS \
--input_file=PATH_TO_FILE_WITH_SEED_EVENT --neurologic_output_dir=PATH_TO_NEUROLOGIC_OUTPUTS_FILE \
--constraint_file=PATH_TO_FILE_WITH_NPS_AS_CONSTRAINTS --control_type=role
'''
import random
import numpy as np
import argparse
import torch
# from transformers import *
import transformers
import logging
import sys
import torch.nn.functional as F
import ast
import pandas as pd
import copy
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
import spacy
# import allennlp
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json

nlp = spacy.load("en_core_web_sm")
coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

np_crap_list = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday','morning','evening','afternoon',
'noon','time','hour','hours','day','week','weeks','that','this','year','years']

def find_clusters(document):
    pred = coref_predictor.predict(document=document.lower())
    clusters = pred['clusters']
    tokens = pred['document']
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for c in cluster:
            t = ' '.join(tokens[c[0]:c[1]+1])
            t = t.replace(" '","'")
            new_cluster.append(' '.join(t.split()))
        
        new_clusters.append(new_cluster)

    final_cluster = []
    for cluster in new_clusters:
        final_cluster.append(list(set(cluster)))
    
    return final_cluster

logger = logging.getLogger(__name__) 

parser = argparse.ArgumentParser()
parser.add_argument("--neurologic_output_dir",type=str, default="data/neurologic_sample.txt")
parser.add_argument("--seeds_file",type=str, default="data/sample_seeds.csv")
parser.add_argument("--input_file",type=str, default="data/sample_input_to_nd.txt")
parser.add_argument("--constraint_file",type=str, default="data/sample_constraints.txt")
parser.add_argument("--control_type", type=str, default="role",help="it can be role or non-role")
args = parser.parse_args()


self_bleu = 0
case_counter = 0
cross_seq_self_bleu = 0
cross_seq_case_counter = 0
counter = 0
chain_length = 1
sequence_number = 5
encoder_no_repeat_ngram_size = 5
no_repeat_ngram_size = 3
num_beams = 1
do_sample = True
top_p = 0.9
data = pd.DataFrame()
total_avg = 0
total_distance_avg = 0
instance_counter = 0
infinity_counter = 0
infinity = 2
if args.control_type == 'role'
    role = True
else:
    role = False

## write identified nps in a separate file and create the required files for neurologic decoding.
# samples = pd.read_csv(args.seeds_file,encoding='utf-8')
# input_file = open('/home/koupaee/qa_schema/data/neurologic_decoding_input.txt','w')
# constraint_file = open('/home/koupaee/qa_schema/data/neurologic_decoding_constraint.json','w')

# for r in range(samples.shape[0]):
#     if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
#         continue
#     counter += 1
#     # if counter == 5:
#     #     break
#     reference_list = []
#     context_list = []
#     initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
#     initial_context = [' '.join(item) for item in initial_context_list]
#     initial_context = ' <TUP> '.join(initial_context)
#     context_list.append(' '.join(initial_context.split()))

#     doc = nlp(initial_context)
#     nps_list = []
#     for chunk in doc.noun_chunks:
#         nps_list.append(chunk.root.text)

#     nps_list = list(set(nps_list))
#     for np in nps_list:
#         constraint_list = []
#         if np in np_crap_list:
#             continue
#         constraint_list.append([np])
#         json.dump(constraint_list,constraint_file)
#         constraint_file.write('\n')
#         input_file.write(initial_context)
#         input_file.write('\n')

# constraint_file.close()
# input_file.close()


input_file = open(args.input_file,'r')
output_file = open(args.neurologic_output_dir,'r')
constraint_file = open(args.constraint_file,'r')
constraints = constraint_file.readlines()

inputs = input_file.readlines()
outputs = output_file.readlines()
c_counter = 0

if not role:
    for i,output in enumerate(outputs):
        counter += 1
        logger.info("counter {}".format(counter))
        output = output.replace('\n','')
        if output[-1]=='.':
            output = output[:-1]
        if ast.literal_eval(constraints[i])[0][0] not in output.split():
            infinity_counter += 1

    print(infinity_counter)

if role:
    for i,output in enumerate(outputs):
        counter += 1
        logger.info("counter {}".format(counter))
        output = output.replace('\n','')
        if output[-1]=='.':
            output = output[:-1]
            
        doc = nlp(inputs[i].replace('\n',''))
        nps_list = []
        nps_everything_list = []
        visited_chunks = []
        for chunk in doc.noun_chunks:
            np = chunk.root.text
            if np in np_crap_list or np in visited_chunks or np != ast.literal_eval(constraints[i])[0][0]:
                continue

            visited_chunks.append(np)
            found = False
            if chunk.root.dep_ == 'conj' and chunk.root.head.dep_ != 'conj':
                np_type = chunk.root.head.dep_
            elif chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'conj':
                np_to_be_checked = (chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_)
                while not found:
                    if nps_list != []:
                        check_list = copy.deepcopy(nps_list[::-1])
                        n = 0
                        while n < len(check_list):
                        # for n in range(len(check_list)):
                            if np_to_be_checked[2] == 'conj' and np_to_be_checked[4] != 'conj':
                                np_type = np_to_be_checked[4]
                                found = True
                                break
                            # if np_to_be_checked != check_list[n] and check_list[n][1] == np_to_be_checked[3] and check_list[n][2] != 'conj':
                            #     np_type = check_list[n][2]
                            #     found = True
                            #     break
                            elif np_to_be_checked != check_list[n] and check_list[n][1] == np_to_be_checked[3] and check_list[n][2] == 'conj':
                                logger.info("INP WHILE {}".format(np_to_be_checked))
                                logger.info("INP WHILE outputs {}".format(nps_list))
                                np_to_be_checked = check_list[n]
                                n = 0
                            else:
                                logger.info("ExcP WHILE {}".format(np_to_be_checked))
                                logger.info("ExcP WHILE {}".format(nps_list))
                                n+=1
                                # np_type = np_to_be_checked[n][2]
                                # found = True
                                # break
                        if not found:
                            logger.info("INP HERE")
                            np_type = np_to_be_checked[2]
                            found = True
                    else:
                        np_type = np_to_be_checked[-1]
                        found = True
            else: 
                np_type = chunk.root.dep_

            if np_type not in ['nsubj','nsubjpass','dobj','pobj']:
                continue

            instance_counter += 1
            nps_list.append((chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_))
            finished = False
            seq = inputs[i].replace('\n','') + '. ' + output
            clusters = find_clusters(seq)
            out_doc = nlp(output)
            out_nps_list = []
            for chunk in out_doc.noun_chunks:
                out_np = chunk.root.text

                if chunk.root.dep_ == 'conj' and chunk.root.head.dep_ != 'conj':
                    out_np_type = chunk.root.head.dep_
                elif chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'conj':
                    np_to_be_checked = (chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_)
                    loop_counter = 0
                    while not found:
                        if out_nps_list != []:
                            check_list = copy.deepcopy(out_nps_list[::-1])
                            n = 0
                            while n < len(check_list):
                                if loop_counter == 100:
                                    logger.info("NONE")
                                    found = True
                                    out_np_type = 'none'
                                    break
                                elif np_to_be_checked[2] == 'conj' and np_to_be_checked[4] != 'conj':
                                    out_np_type = np_to_be_checked[4]
                                    found = True
                                    break
                                elif np_to_be_checked != check_list[n] and check_list[n][1] == np_to_be_checked[3] and check_list[n][2] == 'conj':
                                    logger.info("IN WHILE {}".format(np_to_be_checked))
                                    logger.info("IN WHILE outputs {}".format(out_nps_list))
                                    np_to_be_checked = check_list[n]
                                    n = 0
                                    loop_counter += 1
                                else:
                                    logger.info("Exc WHILE {}".format(np_to_be_checked))
                                    logger.info("Exc WHILE {}".format(out_nps_list))
                                    n += 1
                                    # continue
                                    
                            if not found:
                                logger.info("HERE")
                                out_np_type = np_to_be_checked[2]
                                found = True

                                    
                        else:
                            out_np_type = np_to_be_checked[-1]
                            found = True
                else: 
                    out_np_type = chunk.root.dep_

                out_nps_list.append((chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_))
                # if ('obj' in np_type and 'obj' in out_np_type) or ('subj' in np_type and 'subj' in out_np_type):
                # if np_type == out_np_type:
                    # print(np, np_type)
                    # print(out_np, out_np_type)
                if (('obj' in np_type or np_type == 'nsubjpass') and ('obj' in out_np_type or out_np_type == 'nsubjpass')) or ((np_type == 'nsubj') and (out_np_type == 'nsubj')):
                    if np == out_np:
                        finished = True
                        break
                    else:
                        ## find the coreferring arguments
                        np_cluster = []
                        out_np_cluster = []
                        for cluster in clusters: 
                            for item in cluster:
                                if item == np:
                                    np_cluster.append(clusters.index(cluster))
                                if item == out_np:
                                    out_np_cluster.append(clusters.index(cluster))
                            
                        if list(set(np_cluster) & set(out_np_cluster)) != []:
                            finished = True
                            break


            if not finished:
                infinity_counter += 1

    print(infinity_counter)
    print(instance_counter)