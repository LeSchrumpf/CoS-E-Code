# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv

import numpy as np
import torch
#import apex

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class CQAExample(object):
    def __init__(self,
                 cqa_id,
                 question,
		 explanation,
                 choice_0,
                 choice_1,
                 choice_2,
                 label = None):
        self.cqa_id = cqa_id
        self.question = question
        self.explanation = explanation
        self.choices = [
            choice_0,
            choice_1,
            choice_2,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"cqa_id: {self.cqa_id}",
            f"question: {self.question}",
            f"explanation: {self.explanation}",
            f"choice_0: {self.choices[0]}",
            f"choice_1: {self.choices[1]}",
            f"choice_2: {self.choices[2]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

#Convert .csv files into something usable for the model. 
#Depending on what sort of explanation you want to do, you can set is_select to 0 or 1 using --do_select
def read_cqa_examples(input_file, is_select, is_training):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)
    if not is_select: #if NOT is_select, then CoS-E human explanations
        examples = [
           CQAExample(
            cqa_id = line[0],
            question = line[1],
            explanation = line[6], #adjusted to actually do the right thing
            choice_0 = line[2],
            choice_1 = line[3],
            choice_2 = line[4],
            label = int(line[5]) if is_training else None
        ) for line in lines[1:] # we skip the line with the column names
    ]
    else: #if is_select, then LM explanations
       examples = [
           CQAExample(
            cqa_id = line[0],
            question = line[1],
            explanation = line[7],
            choice_0 = line[2],
            choice_1 = line[3],
            choice_2 = line[4],
            label = int(line[5]) if is_training else None
        ) for line in lines[1:] # we skip the line with the column names
    ]
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training, is_flip):
    """Loads a data file into a list of `InputBatch`s."""

    # CQA is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given CQA example, we will create the 3
    # following inputs using LM expl:
    # - [CLS] question [SEP] expl [SEP] choice_0 [SEP]
    # - [CLS] question [SEP] expl [SEP] choice_1 [SEP]
    # - [CLS] question [SEP] expl [SEP] choice_2 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        question_tokens = tokenizer.tokenize(example.question)
        #start_choice_tokens = tokenizer.tokenize(example.start_choice)
        explanation_tokens = tokenizer.tokenize(example.explanation)

        choices_features = []
        for choice_index, choice in enumerate(example.choices):
            # We create a copy of the question tokens in order to be
            # able to shrink it according to choice_tokens
            question_tokens_choice = question_tokens[:]
            choice_tokens = tokenizer.tokenize(choice)
            explanation_tokens_choice = explanation_tokens[:]
            # Modifies `question_tokens_choice` and `choice_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(question_tokens + explanation_tokens, choice_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + explanation_tokens + ["[SEP]"] + choice_tokens + ["[SEP]"]
            #segment_ids = [0] * (len(question_tokens_choice) + len(explanation_tokens_choice) + 3) + [1] * (len(choice_tokens) + 1)
            if is_flip:
                segment_ids = [0] * (len(question_tokens) + 2) + [1] * (len(choice_tokens) + len(explanation_tokens) + 2)
            else:
                segment_ids = [0] * (len(question_tokens) + len(explanation_tokens) + 3) + [1] * (len(choice_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))
        #Also, show me four examples
        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info(f"cqa_id: {example.cqa_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example.cqa_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
#accuracy function
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

#extract one specific "field": Either input_ids, input_mask, segment_ids, so we can have those neatly seperated from each other
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]



# MAIN PROGRAMM

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=200, #50 for baseline experiments, 175 for non-baseline :D However, it does not work with these amounts to my knowledge
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    # Required for uncased models
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    

    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=6,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit") #fp16 is non-functional in this version of the code, as Apex doesn't want to work on my machine
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_flip",
                        default=False,
                        action='store_true',
                        help="Whether to flip explanation segment.") #This is something they attempted in an effort to improve the results, but it backfired. 
    parser.add_argument("--do_select",
                        action='store_true',
                        help="Whether to use selected text as explanation.") #Whether to use human explanations or LM explanations. If this is true, then LM
    args = parser.parse_args()

    #Whether to use GPU or not:
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    #load the BERT model as a tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    #load training examples and optimization steps
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_cqa_examples(os.path.join(args.data_dir, 'train.csv'), args.do_select, is_training = True)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
        num_choices=3)
    model.to(device)
    if args.local_rank != -1:
        print("I will not use it.")
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    # Comment by authors:
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        print("I won't use this either")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    global_step = 0

    #let us begin the training
    if args.do_train:
        #Do we want to have the explanations together with the question at the end?
        if args.do_flip:
            train_features = convert_examples_to_features(
            		train_examples, tokenizer, args.max_seq_length, True, True)
        #or with the answer choices in the front?
        else:
            train_features = convert_examples_to_features(
                        train_examples, tokenizer, args.max_seq_length, True, False)
        # Little bit of logging to know what's going on... (It never hurt nobody)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        #Loading the training data into the shape BERT requires, with lots of masks
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        #initialize dataloader
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        #put model into training mode 
        model.train()
        #calculating loss during training and using it to optimize the model
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                #backward step
                loss.backward()
                #depending on earlier settings, optimize at particular times
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    #optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
    
    # Save a trained model
    #comment all model saving lines except the one containing "args.data_dir" to load a pre-trained model. (If you dare!!!)
    #You must also comment 'loss': tr_loss/nb_tr_steps in results = {...} later
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    #output_model_file = os.path.join(args.data_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file)
    model = BertForMultipleChoice.from_pretrained(args.bert_model,
        state_dict=model_state_dict,
        num_choices=3)
    model.to(device)

    #Time for evaluation!
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_cqa_examples(os.path.join(args.data_dir, 'dev.csv'), args.do_select, is_training = True)
        #Check on which end the explanations should end up:
        #Either with the question at the end...
        if args.do_flip:
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args.max_seq_length, True, True)
        #...or the multiple choice answers at the front
        else:
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args.max_seq_length, True, False)
        #Loggin'
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        #Converts everything into the tensors needed for BERT
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        #Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            #loading everything in
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                #calculating loss at one instance
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                #loading raw predictions
                logits = model(input_ids, segment_ids, input_mask)
            #converting to numpy to actually be able to use maths with it
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            #calculate accuracy at one instance
            tmp_eval_accuracy = accuracy(logits, label_ids)
            #save overall loss
            eval_loss += tmp_eval_loss.mean().item()
            #save overall accuracy
            eval_accuracy += tmp_eval_accuracy
            #count every step and example
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        #finally, calculate overall loss and accuracy with the stuff we saved earlier
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        #print it
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps #this is the line you have to comment in case you want to load a model checkpoint
                  }
        #put it in a file
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
