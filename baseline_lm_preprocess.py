'''
This following code reads a file in which each line represents an event sequence.
These event sequences are the concatenation of OpenIE events that are extracted from 
news articles. You can refer to data/sample_event_sequences.txt file to see how the
data looks like.

Each line in this file consists of multiple events separated by <TUP> token.
Each event consists of arg0, predicate, arg0, and the sentence it is extracted from, 
all separated by '|'. The first event in each line also contains the document id.

For baseline models, we create pairs of (context,event) where context contains all 
the events prior to the event. 
'''


import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str, default="data/sample_output.csv")
parser.add_argument("--sequence_file",type=str, default="data/sample_event_sequences.txt")

args = parser.parse_args()

context_list = []
answer_list = []
baseline_data = pd.DataFrame()

with open(args.sequence_file,'r') as f:
    sequences = f.readlines()

counter = 0
for seq in sequences:
    counter += 1
    print(counter)
    seq = seq.replace('\n','')
    events_tuples = []
    events = seq.split(' <TUP> ')
    doc_id = events[0].split('|')[0]
    non_rep_sents = []
    for event in events:
        subj = event.split('|')[-4].lower()
        verb = event.split('|')[-3].lower()
        obj = event.split('|')[-2].lower()
        sent = event.split('|')[-1].lower()

        ## all events per sent
        # e_text = ' '.join(subj.split())+' '+' '.join(verb.split())+' '+' '.join(obj.split())
        # e_text = ' '.join(e_text.split())
        # e_text = e_text.replace(" '","'")
        # events_tuples.append(e_text)
        # non_rep_sents.append(sent)

        ## for one event per sent
        if sent not in non_rep_sents:
            e_text = ' '.join(subj.split())+' '+' '.join(verb.split())+' '+' '.join(obj.split())
            e_text = ' '.join(e_text.split())
            e_text = e_text.replace(" '","'")
            events_tuples.append(e_text)
            non_rep_sents.append(sent)
    
    for i in range(1,len(events_tuples)):
        answer = events_tuples[i]
        context = ' <TUP> '.join(events_tuples[:i])
        context = ' '.join(context.split())
        context_list.append(context)
        answer_list.append(answer)

baseline_data['context'] = context_list
baseline_data['answer'] = answer_list

baseline_data.to_csv(args.output_dir,encoding='UTF-8',index=False)

