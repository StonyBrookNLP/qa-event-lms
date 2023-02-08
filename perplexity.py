'''
To calculate the perplexity of the test set using different models
'''

import torch
from transformers import *
import numpy as np
import math
import random
from tqdm import tqdm, trange
import logging
import pandas as pd
import ast
import argparse

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str, default="saved_models/")
parser.add_argument("--perplexity_type", type=str, default="normal",help="it can be normal or marginalized")
parser.add_argument("--model_type", type=str, default="baseline",help="it can be baseline, entity or qa")
parser.add_argument("--test_file",type=str, default="data/test_sample.csv")


args = parser.parse_args()


def score(context,question,answer):
    if args.model_type == 'baseline':
        sequence = context
    else:
        sequence = context + " <SEP> " + question

    sequence = ' '.join(sequence.split())
    
    sample_list = []
    sample_list.append(sequence)

    answer_list = []
    answer_list.append(answer)

    sample_tokenized = tokenizer.batch_encode_plus(sample_list,return_tensors='pt')
    source_ids = sample_tokenized["input_ids"]
    lngth = len(source_ids[0])
    if lngth > 512:
        truncated_source_ids = source_ids[0][lngth-512:]
        tup_idx = tokenizer.convert_tokens_to_ids('<TUP>')
        begin_idx = np.where(truncated_source_ids.detach().cpu().numpy()==tup_idx)[0][0]
        source_ids = truncated_source_ids[begin_idx+1:]
        source_ids = torch.tensor(source_ids).unsqueeze(0)

    input_ids = source_ids
    input_ids = input_ids.to(dtype=torch.long)
    input_ids = input_ids.to(device)

    target_tokenized = tokenizer.batch_encode_plus(answer_list,return_tensors='pt')
    target_ids = target_tokenized["input_ids"]

    input_ids = input_ids.to(device)
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    target_ids = target_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=target_ids)

    loss, logits = outputs[:2]

    if args.perplexity_type == 'normal':
        return loss*target_ids.size(1), target_ids.size(1)
    else:
        return loss, target_ids.size(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

## T5 model
model_class = T5ForConditionalGeneration
tokenizer_class = T5Tokenizer

model_dir = args.output_dir
tokenizer = tokenizer_class.from_pretrained(model_dir)
model = model_class.from_pretrained(model_dir)
model.to(device)
model.eval()

samples = pd.read_csv(args.test_file,encoding='utf-8')
        
## to calculate the perplexity of text sequences
counter = 0
sample_scores = 0
word_count = 0

if args.perplexity_type == 'normal':
    for r in range(samples.shape[0]):
        context_list = ast.literal_eval(samples.iloc[r]['context'])
        context = [' '.join(item) for item in context_list]
        context = ' <TUP> '.join(context)
        context = ' '.join(context.split())
        
        if args.model_type == 'entity':
            if question == "what else happened?":
                question = "none"
            elif question.startswith("what else happened to"):
                question = question.replace("what else happened to","")
            elif question.startswith("what else did"):
                question = question.replace("what else did","")
                question = question.replace("do?","")

        else:
            question = ' '.join(samples.iloc[r]['question'].split())    
        
        question = ' '.join(question.split())
        answer = ast.literal_eval(samples.iloc[r]['answer'])
        answer = ' '.join(answer)

        s, size = score(context,question,answer)
        sample_scores += s.data
        word_count += size

    print("Perplexity is: ", math.exp(sample_scores/word_count))

elif args.perplexity_type == 'marginalized' and args.model_type != 'baseline':
    for r in range(samples.shape[0]):
        q_counter = 1
        context_list = ast.literal_eval(samples.iloc[r]['context'])
        context = [' '.join(item) for item in context_list]
        context = ' <TUP> '.join(context)
        context = ' '.join(context.split())

        if args.model_type == 'entity':
            if question == "what else happened?":
                question = "none"
            elif question.startswith("what else happened to"):
                question = question.replace("what else happened to","")
            elif question.startswith("what else did"):
                question = question.replace("what else did","")
                question = question.replace("do?","")

            question = ' '.join(question.split())
            original_question = question
            event_to_be_asked = ast.literal_eval(samples.iloc[r]['event_to_be_asked'])
            event_to_be_asked = ' '.join(event_to_be_asked)
            doc = nlp(event_to_be_asked)

            s_sample = 0
            s_current, size = score(context,question,answer)
            p = 0
            p += math.exp(s_current)

            if original_question != "what else happened?":
                s_current, size = score(context,"none",answer)
                p += math.exp(s_current)
                q_counter += 1

            for chunk in doc.noun_chunks:
                    if str(chunk) in crap_list:
                        continue
                    if str(chunk) != question:
                        print(str(chunk))
                        s_current, size = score(context,str(chunk),answer)
                        p += math.exp(s_current)
                        q_counter += 1
            
            p = p/q_counter
            new_l = math.log(p)*size
            sample_scores += new_l
            word_count += size

        # print("Perplexity is: ", math.exp(sample_scores/word_count))

        else:
            question = ' '.join(samples.iloc[r]['question'].split())    
            question = ' '.join(question.split())
            original_question = question
            answer = ast.literal_eval(samples.iloc[r]['answer'])
            answer = ' '.join(answer)

            event_to_be_asked = ast.literal_eval(samples.iloc[r]['event_to_be_asked'])
            event_to_be_asked = ' '.join(event_to_be_asked)
            doc = nlp(event_to_be_asked)

            s_sample = 0
            s_current, size = score(context,question,answer)
            p = 0
            p += math.exp(s_current)

            if question != "what happens next?":
                s_current, size = score(context,"what happens next?",answer)
                p += math.exp(s_current)
                q_counter += 1

            for chunk in doc.noun_chunks:
                if str(chunk) in crap_list:
                    continue
                for question in ["what did %s do next?" % str(chunk), "what happens to %s next?" % str(chunk)]:
                    if question != original_question:
                        s_current, size = score(context,question,answer)

                        p += math.exp(s_current)

                        q_counter += 1
            
            print(q_counter)
            p = p/q_counter
            new_l = math.log(p)*size
            sample_scores += new_l
            word_count += size

    print("Perplexity is: ", math.exp(sample_scores/word_count))

elif args.perplexity_type == 'marginalized' and args.model_type == 'baseline':
    print("Marginalized perplexity can only be used with qa or entity models")


