'''
Creating a new T5 model for training the dataset

To use different models, specify the type in --model_type arg. Types can be: 
baseline (for ELM), entity (for EGELM) and qa (fro QGELM).

python t5.py --train_file=PATH_TO_TRAIN --dev_file=PATH_TO_DEV \
--save_path=PATH_TO_DIR_TO_SAVE_MODEL --model_type=baseline
'''

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sys
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import time
import os
# Importing the T5 modules from huggingface/transformers
import transformers
from transformers import *
import logging
#from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_file",type=str, default="data/train_sample.csv")
parser.add_argument("--dev_file",type=str, default="data/dev_sample.csv")
parser.add_argument("--save_path",type=str, default="data/models")
parser.add_argument("--model_type", type=str, default="baseline",help="it can be baseline, entity or qa")

args = parser.parse_args()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

if cuda.is_available():
    print('cuda available')
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    print('cuda not available')
    device = 'cpu'

sys.stdout.flush()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
       
        self.context = self.data.context
        self.question = self.data.question
        self.answer = self.data.answer
        self.event_to_be_asked = self.data.event_to_be_asked
        self.answer_cut = self.data.answer_cut

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        if args.model_type == 'baseline':
            context = str(self.context[index])
            context = ' '.join(context.split())
            answer = str(self.answer[index])
            answer = answer + " " + self.tokenizer.eos_token
            sequence = context

        elif args.model_type == 'qa'
            context_list = ast.literal_eval(self.context[index])
            context = [' '.join(item) for item in context_list]
            context = ' <TUP> '.join(context)
            context = ' '.join(context.split())

            question = str(self.question[index])
            question = ' '.join(question.split())
            answer = ast.literal_eval(self.answer[index])
            answer = ' '.join(answer) + " " + self.tokenizer.eos_token
            sequence = context + " <SEP> " + question
            sequence = ' '.join(sequence.split())

        elif if args.model_type == 'entity'
            ## added to train a system only conditioned on entities.
            if question == "what else happened?":
                question = "none"
            elif question.startswith("what else happened to"):
                question = question.replace("what else happened to","")
            elif question.startswith("what else did"):
                question = question.replace("what else did","")
                question = question.replace("do?","")
        
            question = ' '.join(question.split())
            answer = ast.literal_eval(self.answer[index])
            answer = ' '.join(answer) + " " + self.tokenizer.eos_token
            sequence = context + " <SEP> " + question
            sequence = ' '.join(sequence.split())

        source = self.tokenizer.batch_encode_plus([sequence], pad_to_max_length=True,return_tensors='pt',add_special_tokens=False) #max_length = 512, 
        target = self.tokenizer.batch_encode_plus([answer],truncation=True, max_length= 50, pad_to_max_length=True,return_tensors='pt',add_special_tokens=False) #pad_to_max_length=True,return_tensors='pt',add_special_tokens=False)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        if len(source_ids) > 512:
            truncated_source_ids = source_ids[len(source_ids)-512:]
            tup_idx = self.tokenizer.convert_tokens_to_ids('<TUP>')
            begin_idx = torch.where(truncated_source_ids==tup_idx)[0][0].detach().data
            source_ids = truncated_source_ids[begin_idx+1:]
            
            new_context = self.tokenizer.decode(source_ids,skip_special_tokens=True)
            source = self.tokenizer.batch_encode_plus([new_context], max_length=512, pad_to_max_length=True,return_tensors='pt',add_special_tokens=False)
            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
        
        else:
            source = self.tokenizer.batch_encode_plus([sequence], max_length=512, pad_to_max_length=True,return_tensors='pt',add_special_tokens=False)
            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
            

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            #'target_mask': target_mask.to(dtype=torch.long)
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer,scheduler):
    model.train()
    total_loss = 0
    nb_tr_steps = 0
    for ind,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        lm_labels = y.clone().detach()
        lm_labels[y == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
        loss = outputs.loss
        
        if ind%100 == 0:
            logger.info({"Training Loss": loss.item()})
            sys.stdout.flush()

        if ind%500==0:
            logger.info(f'Epoch: {epoch}, Loss:  {loss.item()}')
            sys.stdout.flush()
        total_loss +=loss.item() 
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        nb_tr_steps += 1
    logger.info(f'Epoch: {epoch}, Average Loss:  {total_loss/ind}')

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            
            lm_labels = y.clone().detach()
            lm_labels[y == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            #outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)

            mc_loss = outputs.loss
            mc_logits = outputs.logits
            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = ids.to("cpu").numpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)
        
            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
        
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        #train_loss = total_loss / nb_tr_steps
        result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}#, "train_loss": train_loss}
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        return eval_loss


def main():
    path = os.path.join(args.save_path,str(args.model_type))
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4   
    TRAIN_EPOCHS = 5  
    VAL_EPOCHS = 1
    LEARNING_RATE = 6.25e-5  
    SEED = 42              
    MODEL_NAME='t5-base'
    do_train=True
    do_test=True
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # T5 model
    model_class = T5ForConditionalGeneration
    tokenizer_class = T5Tokenizer
    pretrained_weights = 't5-base'


    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    tokenizer.add_tokens(['<TUP>','<SEP>'])

    # Importing and Pre-Processing the domain data
    df_train = pd.read_csv(args.train_file,encoding='utf-8')
    df_dev = pd.read_csv(args.dev_file,encoding='utf-8')


    train_dataset=df_train.reset_index(drop=True)
    val_dataset=df_dev.reset_index(drop=True)
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer)
    val_set = CustomDataset(val_dataset, tokenizer)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    ## T5
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    
    t_total = train_dataset.shape[0] // 1 * (TRAIN_EPOCHS-1)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=6.25e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    # Load optimizer and scheduler state
    # optimizer.load_state_dict(torch.load(os.path.join(pre_trained_model,'optimizer.pt')))
    # scheduler.load_state_dict(torch.load(os.path.join(pre_trained_model,'scheduler.pt')))
     
    # Training loop
    if do_train:
        logger.info('Initiating Fine-Tuning for the model on our dataset')
        t0 = time.time()

        # validate(0, tokenizer, model, device, val_loader)
        for epoch in range(1,TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer,scheduler)
            model_to_save = model.module if hasattr(model, "module") else model
            model.save_pretrained(os.path.join(path,str(epoch)))
            #tokenizer.save_vocabulary(os.path.join(path,str(epoch)))
            tokenizer.save_pretrained(os.path.join(path,str(epoch)))
            torch.save(optimizer.state_dict(),os.path.join(os.path.join(path,str(epoch)), 'optimizer.pt'))

            eval_loss = validate(epoch, tokenizer, model, device, val_loader)
            print("eval_loss",eval_loss)

        t1 = time.time()
        total = t1-t0
        print('time spent on training: ', total)

if __name__ == '__main__':
    main()
