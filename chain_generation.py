'''
Generate chain of k events incrementally and compute the intra/inter diversity scores.
To run the code, you need to load a file which consists of seed events and specify the 
number of sequences as well as number of events within each sequecne.

To use different models, specify the type in --model_type arg. Types can be: 
baseline (for ELM), entity (for EGELM) and qa (fro QGELM).

python chain_generation.py --output_dir=PATH_TO_SAVED_MODELS --write_dir=PATH_TO_FILE_TO_SAVE_OUTPUT \
--SEEDS_FILE=PATH_TO_FILE_WITH_SEED_EVENTS --MODEL_TYPE=qa

'''
import random
import numpy as np
import argparse
import torch
from transformers import *
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

nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__) 

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str, default="saved_models/")
parser.add_argument("--write_dir",type=str, default="data/sample_output.csv")
parser.add_argument("--seeds_file",type=str, default="data/sample_seeds.csv")
parser.add_argument("--model_type", type=str, default="baseline",help="it can be baseline, entity or qa")
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
chain_length = 10
sequence_number = 5
encoder_no_repeat_ngram_size = 5
no_repeat_ngram_size = 3
num_beams = 1
do_sample = True
top_p = 0.9
data = pd.DataFrame()

ngram = 3
weight = tuple((1. / ngram for _ in range(ngram)))

crap_list = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday','morning','evening','afternoon',
'noon','time','hour','hours','day','week','weeks','that','this','year','years','co','-','days','sundays','mondays',
'tuesdays','wednesdays','thursdays','fridays','saturdays','mornings','evenings','afternoons','noons',
'times','other','another',"'",'any','none','.']

if args.model_type == 'baseline':
    all_response_list = []
    all_seeds_list = []
    visited_seeds = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1 or ast.literal_eval(samples.iloc[r]['context']) in visited_seeds:
            continue

        counter += 1
        reference_list = []

        visited_seeds.append(ast.literal_eval(samples.iloc[r]['context']))
        seq_cross_seq_self_bleu = 0
        seq_cross_seq_case_counter = 0
        for l in range(sequence_number):
            context_list = []
            initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
            initial_context = [' '.join(item) for item in initial_context_list]
            initial_context = ' <TUP> '.join(initial_context)
            context_list.append(' '.join(initial_context.split()))

            all_seeds_list.append(initial_context)

            
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
                        #top_k=0,
                        #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                        top_p=top_p,
                        # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                        # no_repeat_ngram_size = no_repeat_ngram_size,
                        # output_scores = True
                    )
                out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                context_list.append(out)

            ## intra-sequence self-bleu
            result = []
            if len(context_list) >= 2:
                for index in range(len(context_list)):

                    hypothesis = context_list[index].split()
                    new_other_list = context_list[:index] + context_list[index+1:]
                    other = []
                    for o in new_other_list:
                        other.append(o.split())
                    result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                        smoothing_function=SmoothingFunction().method1))
                        
                score = 0.0
                cnt = 0
                for i in result:
                    score += i
                    cnt += 1

                if cnt != 0:
                    cross_seq_self_bleu += score/cnt
                    cross_seq_case_counter += 1

            reference_list.append(' <TUP> '.join(context_list[1:]))
            all_response_list.append(' <TUP> '.join(context_list[1:]))
        
        ## inter-sequence self-bleu
        result = []
        if len(reference_list) >= 2:
            for index in range(len(reference_list)):
                hypothesis = reference_list[index].split()
                new_other_list = reference_list[:index] + reference_list[index+1:]
                other = []
                for o in new_other_list:
                    other.append(o.split())
                result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                            smoothing_function=SmoothingFunction().method1))
                
            score = 0.0
            cnt = 0
            for i in result:
                score += i
                cnt += 1

            if cnt != 0:
                self_bleu += score/cnt
                case_counter += 1

    print("inter-sequence self_bleu: ", self_bleu/case_counter)
    print("intra-sequence self_bleu: ", cross_seq_self_bleu/cross_seq_case_counter)

    data['seeds'] = all_seeds_list
    data['chains'] = all_response_list
    data.to_csv(args.write_dir,encoding='UTF-8',index=False)

elif args.model_type == 'qa':
    all_response_list = []
    all_seeds_list = []
    all_qs_list = [[] for i in range(chain_length)]
    all_outs_list = [[] for i in range(chain_length)]
    visited_seeds = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1 or ast.literal_eval(samples.iloc[r]['context']) in visited_seeds:
            continue

        counter += 1
        question_list = []
        event_to_be_asked = ast.literal_eval(samples.iloc[r]['event_to_be_asked'])
        event_to_be_asked = ' '.join(event_to_be_asked)
        doc = nlp(event_to_be_asked)
        for chunk in doc.noun_chunks:
            if str(chunk) not in crap_list:
                for question in ["what did %s do next?" % str(chunk.text), "what happens to %s next?" % str(chunk.text)]:
                    question_list.append(question)

        ## to add "what happens next?"
        # question_list.append("what happens next?")   
        q_number = len(question_list)

        reference_list = []
        visited_seeds.append(ast.literal_eval(samples.iloc[r]['context']))
        if q_number >= sequence_number:
            questions = random.sample(question_list, sequence_number)
            for l in range(sequence_number):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                current_q = questions[l]

                all_seeds_list.append(initial_context)


                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            for question in ["what did %s do next?" % str(chunk.text), "what happens to %s next?" % str(chunk.text)]:
                                new_question_list.append(question)
                    
                    ## to add "what happens next?"
                    # new_question_list.append("what happens next?")
                    if new_question_list == []:
                        new_question_list.append('what happens next?')
                    current_q = random.sample(new_question_list, 1)[0]
                
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))
            
            ## inter-sequence self-bleu
            result = []
            if len(reference_list) >= 2:
                for index in range(len(reference_list)):
                    hypothesis = reference_list[index].split()
                    new_other_list = reference_list[:index] + reference_list[index+1:]
                    other = []
                    for o in new_other_list:
                        other.append(o.split())
                    result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
                    
                score = 0.0
                cnt = 0
                for i in result:
                    score += i
                    cnt += 1

                if cnt != 0:
                    self_bleu += score/cnt
                    case_counter += 1        
        
        ## if number of generated questions is lower than number of sequences
        elif q_number < sequence_number:
            for l in range(len(question_list)):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                current_q = question_list[l]

                all_seeds_list.append(initial_context)

                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            for question in ["what did %s do next?" % str(chunk.text), "what happens to %s next?" % str(chunk.text)]:
                            # for question in ["what happens to %s next?" % str(chunk.text)]:
                                new_question_list.append(question)
                    
                    ## to add "what happens next?"
                    # new_question_list.append("what happens next?")
                    if new_question_list == []:
                        new_question_list.append('what happens next?')
                    current_q = random.sample(new_question_list, 1)[0]
                    
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))

            diff = sequence_number-len(question_list)
            for d in range(diff):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))

                all_seeds_list.append(initial_context)

                
                if question_list == []:
                        question_list.append('what happens next?')
                current_q = random.sample(question_list, 1)[0]

                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            for question in ["what did %s do next?" % str(chunk.text), "what happens to %s next?" % str(chunk.text)]:
                            # for question in ["what happens to %s next?" % str(chunk.text)]:
                                new_question_list.append(question)
                    
                    ## to add "what happens next?"
                    # new_question_list.append("what happens next?")
                    if new_question_list == []:
                        new_question_list.append('what happens next?')
                    current_q = random.sample(new_question_list, 1)[0]
                    
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))

            ## inter-sequence self-bleu
            result = []
            if len(reference_list) >= 2:
                for index in range(len(reference_list)):
                    hypothesis = reference_list[index].split()
                    new_other_list = reference_list[:index] + reference_list[index+1:]
                    other = []
                    for o in new_other_list:
                        other.append(o.split())
                    result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
                    
                score = 0.0
                cnt = 0
                for i in result:
                    score += i
                    cnt += 1

                if cnt != 0:
                    self_bleu += score/cnt
                    case_counter += 1       


    print("inter-sequence self_bleu: ",self_bleu/case_counter)
    print("intra-sequence self_bleu: ",cross_seq_self_bleu/cross_seq_case_counter)

    data['seeds'] = all_seeds_list
    data['chains'] = all_response_list
    for i in range(chain_length):
        data['q_%i'%i] = all_qs_list[i]
        data['out_%i'%i] = all_outs_list[i]

    data.to_csv(args.write_dir,encoding='UTF-8',index=False)
            
elif args.model_type == 'entity':
    all_response_list = []
    all_seeds_list = []
    all_qs_list = [[] for i in range(chain_length)]
    all_outs_list = [[] for i in range(chain_length)]
    visited_seeds = []

    for r in range(samples.shape[0]):
        if len(ast.literal_eval(samples.iloc[r]['context'])) != 1 or ast.literal_eval(samples.iloc[r]['context']) in visited_seeds:
            continue

        counter += 1
        question_list = []
        event_to_be_asked = ast.literal_eval(samples.iloc[r]['event_to_be_asked'])
        event_to_be_asked = ' '.join(event_to_be_asked)
        doc = nlp(event_to_be_asked)
        for chunk in doc.noun_chunks:
            if str(chunk) not in crap_list:
                question_list.append(str(chunk))

        ## to account for "what happens next?"
        question_list.append("none")   
        q_number = len(question_list)

        reference_list = []
        visited_seeds.append(ast.literal_eval(samples.iloc[r]['context']))
        if q_number >= sequence_number:
            questions = random.sample(question_list, sequence_number)
            for l in range(sequence_number):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                current_q = questions[l]

                all_seeds_list.append(initial_context)


                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())

                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            new_question_list.append(str(chunk))
                    
                    ## to account for "what happens next?"
                    new_question_list.append("none")
                    if new_question_list == []:
                        new_question_list.append("none")
                    current_q = random.sample(new_question_list, 1)[0]
                
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))
            
            ## inter-sequence self-bleu
            result = []
            if len(reference_list) >= 2:
                for index in range(len(reference_list)):
                    hypothesis = reference_list[index].split()
                    new_other_list = reference_list[:index] + reference_list[index+1:]
                    other = []
                    for o in new_other_list:
                        other.append(o.split())
                    result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
                    
                score = 0.0
                cnt = 0
                for i in result:
                    score += i
                    cnt += 1

                if cnt != 0:
                    self_bleu += score/cnt
                    case_counter += 1        
        
        ## if number of generated questions is lower than number of sequences
        elif q_number < sequence_number:
            for l in range(len(question_list)):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))
                current_q = question_list[l]

                all_seeds_list.append(initial_context)

                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            new_question_list.append(str(chunk))
                    
                    ## to account for "what happens next?"
                    new_question_list.append("none")
                    if new_question_list == []:
                        new_question_list.append("none")
                    current_q = random.sample(new_question_list, 1)[0]
                    
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))

            diff = sequence_number-len(question_list)
            for d in range(diff):
                context_list = []
                initial_context_list = ast.literal_eval(samples.iloc[r]['context'])    
                initial_context = [' '.join(item) for item in initial_context_list]
                initial_context = ' <TUP> '.join(initial_context)
                context_list.append(' '.join(initial_context.split()))

                all_seeds_list.append(initial_context)

                
                if question_list == []:
                        question_list.append('none')
                current_q = random.sample(question_list, 1)[0]

                for i in range(chain_length):
                    context = ' <TUP> '.join(context_list)
                    context = ' '.join(context.split())
                    
                    sequence = context + ' <SEP> ' + current_q
                    sequence = ' '.join(sequence.split())

                    all_qs_list[i].append(current_q)

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
                            #top_k=0,
                            #temperature=0.8 ##use temperature to decrease the sensitivity to low probability candidates
                            top_p=top_p,
                            # encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
                            # no_repeat_ngram_size = no_repeat_ngram_size,
                            # output_scores = True
                        )
                    out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                    context_list.append(out)
                    all_outs_list[i].append(out)

                    doc = nlp(out)
                    new_question_list = []
                    for chunk in doc.noun_chunks:
                        if str(chunk) not in crap_list:
                            new_question_list.append(str(chunk)
                    
                    ## to account for "what happens next?"
                    new_question_list.append("none")
                    if new_question_list == []:
                        new_question_list.append("none")
                    current_q = random.sample(new_question_list, 1)[0]
                    
                ## intra-sequence self-bleu
                result = []
                if len(context_list) >= 2:
                    for index in range(len(context_list)):
                        hypothesis = context_list[index].split()
                        new_other_list = context_list[:index] + context_list[index+1:]
                        other = []
                        for o in new_other_list:
                            other.append(o.split())
                        result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                            smoothing_function=SmoothingFunction().method1))
                            
                    score = 0.0
                    cnt = 0
                    for i in result:
                        score += i
                        cnt += 1

                    if cnt != 0:
                        cross_seq_self_bleu += score/cnt
                        cross_seq_case_counter += 1

                reference_list.append(' <TUP> '.join(context_list[1:]))
                all_response_list.append(' <TUP> '.join(context_list[1:]))

            ## inter-sequence self-bleu
            result = []
            if len(reference_list) >= 2:
                for index in range(len(reference_list)):
                    hypothesis = reference_list[index].split()
                    new_other_list = reference_list[:index] + reference_list[index+1:]
                    other = []
                    for o in new_other_list:
                        other.append(o.split())
                    result.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
                    
                score = 0.0
                cnt = 0
                for i in result:
                    score += i
                    cnt += 1

                if cnt != 0:
                    self_bleu += score/cnt
                    case_counter += 1       


    print("inter-sequence self_bleu: ",self_bleu/case_counter)
    print("intra-sequence self_bleu: ",cross_seq_self_bleu/cross_seq_case_counter)

    data['seeds'] = all_seeds_list
    data['chains'] = all_response_list
    for i in range(chain_length):
        data['q_%i'%i] = all_qs_list[i]
        data['out_%i'%i] = all_outs_list[i]

    data.to_csv(args.write_dir,encoding='UTF-8',index=False)
            

