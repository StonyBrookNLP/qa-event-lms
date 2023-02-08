'''
This following code reads a file in which each line represents an event sequence.
These event sequences are the concatenation of OpenIE events that are extracted from 
news articles. You can refer to data/sample_event_sequences.txt file to see how the
data looks like.

Each line in this file consists of multiple events separated by <TUP> token.
Each event consists of arg0, predicate, arg0, and the sentence it is extracted from, 
all separated by '|'. The first event in each line also contains the document id.

For question/entity guided models, we create tuples of (context,question,answer) 
where question asks about an entity from the last event in the given context,
context contains all the events prior to the event in question and answer is any 
subsequent event that has the entity in question in the corresponding role. 

We also read the original new articles to use them to find the coreferring clusters.
We observed that using the original articles, instead of the event sequences results 
in higher accuracy of the coref resolution system.
'''


import copy
import pandas as pd
import spacy 
import os
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import datetime
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",type=str, default="data/sample_output.csv")
parser.add_argument("--sequence_file",type=str, default="data/sample_event_sequences.txt")

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

nlp = spacy.load("en_core_web_sm")

with open(args.sequence_file,'r') as f:
    sequences = f.readlines()

training_data = pd.DataFrame()
training_corpus = []
context_list = []
question_list = []
answer_list = []
event_to_be_asked_list = []

def find_clusters(id):
    ## we use doc_ids to easily search for the documents
    for root, dirs, files in os.walk('/home/koupaee/qa_schema/data/nyt_corpus_text/'):
        for file in files:
            if '_'.join(os.path.join(root,file).split('/')[-4::1]) == id:
                with open(os.path.join(root, file),'r') as f:
                    document = f.read()
                    break


    pred = predictor.predict(document=document.lower())
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

current_doc_id = sequences[0].split('|')[0]
clusters = find_clusters(current_doc_id)

counter = 0
for seq in sequences:
    counter += 1
    logger.info({"counter: ": counter})
    events_queue = []
    seq = seq.replace('\n','')
    events_tuples = []
    events = seq.split(' <TUP> ')
    doc_id = events[0].split('|')[0]
    if doc_id != current_doc_id:
        current_doc_id = doc_id
        clusters = find_clusters(current_doc_id)

    non_rep_sents = []
    for event in events:
        subj = event.split('|')[-4].lower()
        verb = event.split('|')[-3].lower()
        obj = event.split('|')[-2].lower()
        sent = event.split('|')[-1].lower()
        # events_tuples.append((doc_id,' '.join(subj.split()),' '.join(verb.split()),' '.join(obj.split()),' '.join(sent.split())))
        
        ## for one event per sent
        if sent not in non_rep_sents:
            events_tuples.append((doc_id,' '.join(subj.split()),' '.join(verb.split()),' '.join(obj.split()),' '.join(sent.split())))
            non_rep_sents.append(sent)

    events_queue.append(events_tuples[0])  ## add seed event to the queue
    context = []
    events_queue_list = []
    e_count = 0
    for e in events_tuples:
        e_count += 1
        # print("event %s: %s" %(e_count,e[0]))
        event_to_be_asked = e
        event_rep = ' '.join(event_to_be_asked[1:4]) ## openIE rep of an event
        event_rep = event_rep.replace(" '","'")
        event_rep = ' '+event_rep+' '

        if event_to_be_asked not in context:
            context.append(event_to_be_asked)
        idx = events_tuples.index(event_to_be_asked)
        
        event_nouns_list = []
        doc = nlp(event_to_be_asked[-1])
        for chunk in doc.noun_chunks:
            event_nouns_list.append(chunk)

        event_cond = False
        for noun in event_nouns_list:
            noun = str(noun)
            if ' '+noun+' ' not in event_rep:
                continue
            
            subj_cond = False
            obj_cond = False
            ## ask what the noun phrase does next
            for tt in range(idx+1,len(events_tuples)):
                if subj_cond == True and obj_cond == True:
                    break
                
                new_event_rep = ' '.join(events_tuples[tt][1:4])
                new_event_rep = new_event_rep.replace(" '","'")
                new_event_rep = ' '+new_event_rep+' '
                new_doc = nlp(events_tuples[tt][-1])
                    
                for chunk in new_doc.noun_chunks:
                    n_cluster = []
                    ss_cluster = []
                    nn_cluster = []
                    oo_cluster = []
                    if subj_cond == True and obj_cond == True:
                        break

                    if ' '+str(chunk)+' ' not in new_event_rep:
                        continue

                    if subj_cond == False and obj_cond == False:
                        if chunk.root.dep_ == 'nsubj' or (chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'nsubj'):
                            ss = chunk
                            ss = str(ss)

                            ## find the coreferring arguments
                            for cluster in clusters: 
                                if n_cluster != [] and ss_cluster != []:
                                    break
                                for item in cluster:
                                    if item == noun:
                                        n_cluster.append(clusters.index(cluster))

                                    if item == ss:
                                        ss_cluster.append(clusters.index(cluster))
                                
                            if list(set(n_cluster) & set(ss_cluster)) != []:
                                subj_cond = True
                                event_cond = True
                                question = "what else did %s do?" % noun
                                answer = events_tuples[tt]

                                question_list.append(question)
                                answer_list.append(answer)
                                context_list.append(copy.deepcopy(context))
                                event_to_be_asked_list.append(event_to_be_asked)


                    if obj_cond == False and subj_cond == False:
                        if chunk.root.dep_ == 'nsubjpass' or 'obj' in chunk.root.dep_ or (chunk.root.dep_ == 'conj' and chunk.root.head.dep_ == 'nsubjpass') \
                        or (chunk.root.dep_ == 'conj' and 'obj' in chunk.root.head.dep_ ):
                            oo = chunk
                            oo = str(oo)

                            ## find the coreferring arguments
                            for cluster in clusters: 
                                if nn_cluster != [] and ss_cluster != []:
                                    break
                                for item in cluster:
                                    if item == noun:
                                        nn_cluster.append(clusters.index(cluster))

                                    if item == oo:
                                        oo_cluster.append(clusters.index(cluster))
                                
                            if list(set(nn_cluster) & set(oo_cluster)) != []:
                                obj_cond = True
                                event_cond = True
                                question = "what else happened to %s?" % noun
                                answer = events_tuples[tt]

                                question_list.append(question)
                                answer_list.append(answer)
                                context_list.append(copy.deepcopy(context))
                                event_to_be_asked_list.append(event_to_be_asked)
                                

        if event_cond == False and idx != len(events_tuples)-1:# and events_queue == []:
            question = "what else happened?"
            answer = events_tuples[idx+1]
            question_list.append(question)
            answer_list.append(answer)
            context_list.append(copy.deepcopy(context))
            event_to_be_asked_list.append(event_to_be_asked)
            
        
new_context_list = []
new_context_list2 = []

for item in context_list:
    temp_list = []
    temp_list2 = []
    for c in item:
        temp_list.append((c[1],c[2],c[3]))
        temp_list2.append(c[4])
    new_context_list.append(temp_list)
    new_context_list2.append(temp_list2)

new_answer_list = []
for a in answer_list:
    new_answer_list.append((a[1],a[2],a[3]))

new_event_to_be_asked_list = []
for a in event_to_be_asked_list:
    new_event_to_be_asked_list.append((a[1],a[2],a[3]))

training_data['context'] = new_context_list
training_data['event_to_be_asked'] = new_event_to_be_asked_list
training_data['question'] = question_list
training_data['answer'] = new_answer_list

training_data.to_csv(args.output_dir,encoding='UTF-8',index=False)
df = pd.read_csv(args.output_dir)
df = df.drop_duplicates(subset=['context','question','answer'])
df.to_csv(args.output_dir,encoding='UTF-8',index=False)