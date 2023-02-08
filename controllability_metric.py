'''
code to measure the controllability of the baseline system.
For each event, we keep sampling the next events (up to a max number)
until the entity of interest (and its role) is generated in the output.

To use different models, specify the type in --model_type arg. Types can be: 
baseline (for ELM), entity (for EGELM) and qa (fro QGELM).

python controllability_metric.py --seeds_file=PATH_TO_FILE_WITH_SEED_EVENTS \
--write_dir=PATH_TO_FILE_TO_SAVE_OUTPUT --output_dir=PATH_TO_SAVED_MODELS \
--model_type=qa --control_type=role
'''
import random
import numpy as np
import argparse
import torch
# from transformers import * ## commented because of the environments
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

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
parser.add_argument("--output_dir",type=str, default="saved_models/")
parser.add_argument("--write_dir",type=str, default="data/sample_output.csv")
parser.add_argument("--seeds_file",type=str, default="data/sample_seeds.csv")
parser.add_argument("--generation_type", type=str, default="beam",help="it can be greedy, beam, sampling")
parser.add_argument("--model_type", type=str, default="baseline",help="it can be baseline, entity or qa")
parser.add_argument("--control_type", type=str, default="role",help="it can be role or non-role")
parser.add_argument("--seed", type=int, default=114)
parser.add_argument("--output_length", type=int, default=40)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

# load a pre-trained T5 model
model_class = T5ForConditionalGeneration 
tokenizer_class = T5Tokenizer

# Load a args.output_dir model and vocabulary that you have fine-tuned
tokenizer = tokenizer_class.from_pretrained(args.output_dir)
model = model_class.from_pretrained(args.output_dir)
model.to(device)
model.eval()

samples = pd.read_csv(args.seeds_file,encoding='utf-8')

self_bleu = 0
case_counter = 0
cross_seq_self_bleu = 0
cross_seq_case_counter = 0
counter = 0
chain_length = 1
sequence_number = 5
encoder_no_repeat_ngram_size = 5
no_repeat_ngram_size = 3
num_beams = 40
do_sample = False
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

chain = False

if args.model_type == 'baseline' and not role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue
        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
            
        doc = nlp(initial_context)
        nps_list = []
        for chunk in doc.noun_chunks:
            nps_list.append(chunk.root.text)

        nps_list = list(set(nps_list))
        for np in nps_list:
            if np in np_crap_list:
                continue
            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))

                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    doc = nlp(out)
                    out_nps_list = []
                    for chunk in doc.noun_chunks:
                        out_nps_list.append(chunk.root.text)
                    out_nps_list = list(set(out_nps_list))
                    
                    ## find the coreferring arguments
                    for out_np in out_nps_list:
                        np_cluster = []
                        out_np_cluster = []
                        for cluster in clusters: 
                            for item in cluster:
                                if item == np:
                                    np_cluster.append(clusters.index(cluster))
                                if item == out_np:
                                    out_np_cluster.append(clusters.index(cluster))
                        
                        if np == out_np or list(set(np_cluster) & set(out_np_cluster)) != []:
                            finished = True
                    
                    sample_numbers += 1
                print(np)
                print(out)
                print(seq)
            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)

if args.model_type == 'baseline' and role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue
        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
        logger.info("seed {}".format(context_list))
            
        doc = nlp(initial_context)
        nps_list = []
        nps_everything_list = []
        visited_chunks = []
        for chunk in doc.noun_chunks:
            np = chunk.root.text
            if np in np_crap_list or np in visited_chunks:
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


            nps_list.append((chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_))

            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    sample_numbers += 1
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    out_doc = nlp(out)
                    out_nps_list = []
                    for chunk in out_doc.noun_chunks:
                        if finished:
                            break
                        out_np = chunk.root.text
                        found = False

                        if chunk.root.dep_ == 'conj' and chunk.root.head.dep_ != 'conj':
                            out_np_type = chunk.root.head.dep_
                        elif chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'conj':
                            np_to_be_checked = (chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_)
                            loop_counter = 0
                            while not found:
                                if out_nps_list != []:
                                    # logger.info("ALL WHILE {}".format(np_to_be_checked))
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
                                        # if np_to_be_checked != check_list[n] and check_list[n][1] == np_to_be_checked[3] and check_list[n][2] != 'conj':
                                        #     out_np_type = check_list[n][2]
                                        #     found = True
                                        #     break
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

                        if ('obj' in np_type and 'obj' in out_np_type) or ('subj' in np_type and 'subj' in out_np_type):
                        # if np_type == out_np_type:
                        # if (('obj' in np_type or np_type == 'nsubjpass') and ('obj' in out_np_type or out_np_type == 'nsubjpass')) or ((np_type == 'nsubj') and (out_np_type == 'nsubj')):
                            if np == out_np:
                                finished = True
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
                    print(np)
                    print(out)
                    print(seq)

            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)

if args.model_type == 'qa' and not role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue

        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
            
        doc = nlp(initial_context)
        nps_list = []
        for chunk in doc.noun_chunks:
            nps_list.append(chunk.root.text)

        nps_list = list(set(nps_list))
        for np in nps_list:
            if np in np_crap_list:
                continue
            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                
                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    current_q = random.sample(['what happens to %s next?' % np,'what did %s do next?' % np],1)[0]
                    # print(current_q)

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    sample_numbers += 1
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    doc = nlp(out)
                    out_nps_list = []
                    for chunk in doc.noun_chunks:
                        out_nps_list.append(chunk.root.text)
                    out_nps_list = list(set(out_nps_list))
                    
                    ## find the coreferring arguments
                    for out_np in out_nps_list:
                        np_cluster = []
                        out_np_cluster = []
                        for cluster in clusters: 
                            for item in cluster:
                                if item == np:
                                    np_cluster.append(clusters.index(cluster))
                                if item == out_np:
                                    out_np_cluster.append(clusters.index(cluster))
                        
                        if np == out_np or list(set(np_cluster) & set(out_np_cluster)) != []:
                            finished = True
                print(np)
                print(out)
                print(seq)

            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)

if args.model_type == 'qa' and role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue

        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
            
        doc = nlp(initial_context)
        nps_list = []
        visited_chunks = []
        
        for chunk in doc.noun_chunks:
            np = chunk.root.text
            if np in np_crap_list or np in visited_chunks:
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

            print(np_type)
            if np_type not in ['nsubj','nsubjpass','dobj','pobj']:
                continue
            nps_list.append((chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_))

            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                
                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    #current_q = random.sample(['what happens to %s next?' % np,'what did %s do next?' % np],1)[0]
                    ## to ask question with the specific role in mind
                    if np_type == 'nsubj':
                        current_q = 'what did %s do next?' % np
                    elif 'obj' in np_type or np_type == 'nsubjpass':
                        current_q = "what happens to %s next?" % np
                    else:
                        current_q = 'what did %s do next?' % np
                    print(current_q)

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    sample_numbers += 1
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    out_doc = nlp(out)
                    out_nps_list = []

                    for chunk in out_doc.noun_chunks:
                        if finished:
                            break
                        out_np = chunk.root.text
                        found = False

                        if chunk.root.dep_ == 'conj' and chunk.root.head.dep_ != 'conj':
                            out_np_type = chunk.root.head.dep_
                        elif chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'conj':
                            np_to_be_checked = (chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_)
                            loop_counter = 0
                            while not found:
                                if out_nps_list != []:
                                    # logger.info("ALL WHILE {}".format(np_to_be_checked))
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

                        # if np_type == out_np_type:
                        if (('obj' in np_type or np_type == 'nsubjpass') and ('obj' in out_np_type or out_np_type == 'nsubjpass')) or ((np_type == 'nsubj') and (out_np_type == 'nsubj')):
                            if np == out_np:
                                finished = True
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

                print(np)
                print(out)
                print(seq)

            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)

if args.model_type == 'entity' and not role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue

        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
            
        doc = nlp(initial_context)
        nps_list = []
        for chunk in doc.noun_chunks:
            nps_list.append(chunk.root.text)

        nps_list = list(set(nps_list))
        for np in nps_list:
            if np in np_crap_list:
                continue
            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                
                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    current_q = np
                    # print(current_q)

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    sample_numbers += 1
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    doc = nlp(out)
                    out_nps_list = []
                    for chunk in doc.noun_chunks:
                        out_nps_list.append(chunk.root.text)
                    out_nps_list = list(set(out_nps_list))
                    
                    ## find the coreferring arguments
                    for out_np in out_nps_list:
                        np_cluster = []
                        out_np_cluster = []
                        for cluster in clusters: 
                            for item in cluster:
                                if item == np:
                                    np_cluster.append(clusters.index(cluster))
                                if item == out_np:
                                    out_np_cluster.append(clusters.index(cluster))
                        
                        if np == out_np or list(set(np_cluster) & set(out_np_cluster)) != []:
                            finished = True
                print(np)
                print(out)
                print(seq)

            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)


if args.model_type == 'entity' and role:
    all_outs_list = []
    all_seeds_list = []
    all_nps_list = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1:
            continue

        counter += 1
        logger.info("counter {}".format(counter))

        reference_list = []
        context_list = []
        initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
        initial_context = [' '.join(item) for item in initial_context_list]
        initial_context = ' <TUP> '.join(initial_context)
        context_list.append(' '.join(initial_context.split()))
            
        doc = nlp(initial_context)
        nps_list = []
        visited_chunks = []
        
        for chunk in doc.noun_chunks:
            np = chunk.root.text
            if np in np_crap_list or np in visited_chunks:
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

            print(np_type)
            if np_type not in ['nsubj','nsubjpass','dobj','pobj']:
                continue
            nps_list.append((chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_))

            instance_counter += 1
            finished = False
            sample_numbers = 0
            while not finished and sample_numbers != infinity:
                all_nps_list.append(np)
                all_seeds_list.append(initial_context)
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                
                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    current_q = np
                    print(current_q)

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    sample_list = []
                    sample_list.append(sequence)
                    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')

                    source_ids = sample_tokenized["input_ids"]
                    lngth = len(source_ids[0])
                    if lngth > 512:
                        truncated_source_ids = source_ids[0][lngth-512:]
                        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
                        begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
                        source_ids = truncated_source_ids[begin_idx+1:]
                        source_ids = torch.tensor(source_ids).unsqueeze(0)

                    input_ids = source_ids
                    input_ids = input_ids.to(dtype=torch.long)
                    input_ids = input_ids.to(device)

                    sample_output = model.generate(
                            input_ids, 
                            do_sample=do_sample, 
                            max_length=100, 
                            repetition_penalty=2.5,
                            num_beams=num_beams,
                            top_p=top_p,
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    all_outs_list.append(out)
                    sample_numbers += 1
                    context_list.append(out)
                    seq = '. '.join(context_list)
                    clusters = find_clusters(seq)

                    out_doc = nlp(out)
                    out_nps_list = []

                    for chunk in out_doc.noun_chunks:
                        if finished:
                            break
                        out_np = chunk.root.text
                        found = False

                        if chunk.root.dep_ == 'conj' and chunk.root.head.dep_ != 'conj':
                            out_np_type = chunk.root.head.dep_
                        elif chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'conj':
                            np_to_be_checked = (chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text,chunk.root.head.dep_)
                            loop_counter = 0
                            while not found:
                                if out_nps_list != []:
                                    # logger.info("ALL WHILE {}".format(np_to_be_checked))
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

                        # if np_type == out_np_type:
                        if (('obj' in np_type or np_type == 'nsubjpass') and ('obj' in out_np_type or out_np_type == 'nsubjpass')) or ((np_type == 'nsubj') and (out_np_type == 'nsubj')):
                            if np == out_np:
                                finished = True
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

                print(np)
                print(out)
                print(seq)

            print(sample_numbers) 
            if sample_numbers == infinity:
                infinity_counter += 1
            total_avg += sample_numbers

    print(total_avg/instance_counter)
    print(instance_counter)
    print(infinity_counter)

    data['context'] = all_seeds_list
    data['np'] = all_nps_list
    data['output'] = all_outs_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)