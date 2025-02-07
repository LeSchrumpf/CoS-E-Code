import math
import os

from sacrebleu import corpus_bleu
import csv
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIAdam
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

#Enables logging during training, validation, testing. 
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#Calculating BLEU scores
def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score

#Perhaps a copied pre_processing function, because most of this isn't used. 
def pre_process_datasets(encoded_datasets, start_token, delimiter_token, answer_token, clf_token, end_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)my 
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        tensors, labels, prompts = [], [], []
        max_len = 0
        max_prompt = 0
        train=True
        for i, ex in enumerate(dataset):
            tensor = ex.natural_tensor(end_token)
            if len(tensor) > max_len:
                max_len = len(tensor)
            if len(ex.text) > max_prompt:
                max_prompt = len(ex.text)
            if ex.explanation is not None:
                train = True
                label = [-1] * len(ex.text) + tensor[len(ex.text):]
                labels.append(label)
            else: 
                train = False
            prompts.append(ex.text)
            tensors.append(tensor)
        tensors = [f + [0.] * (max_len - len(f)) for f in tensors]
        prompts = [f + [-1] * (max_prompt - len(f)) for f in prompts]
        labels = [f + [-1] * (max_len - len(f)) for f in labels]
        #if there is an explanation, we slam label on top of it
        if train:
            tensor_datasets.append([torch.tensor(tensors, dtype=torch.long)[:, :-1], torch.tensor(labels, dtype=torch.long)[:, 1:], torch.tensor(prompts, dtype=torch.long)])
        else:
            tensor_datasets.append([torch.tensor(tensors, dtype=torch.long)[:, :-1], torch.tensor(prompts, dtype=torch.long)])
    return tensor_datasets


class CommonsenseExample:

    def __init__(self, question, choice0, choice1, choice2, answer_text, explanation, explain_predict=True):
        #create a self.text to use for logging later
        if explain_predict:
            self.text = f'{question} The choices are {choice0}, {choice1}, or {choice2}. My commonsense tells me '
        else:
            self.text = f'{question} The choices are {choice0}, {choice1}, or {choice2}. The answer is {answer_text} because '
        #then grant thyself the appropriate arguments
        self.question = question
        self.choice0 = choice0
        self.choice1 = choice1
        self.choice2 = choice2
        self.answer_text = answer_text 
        self.explanation = explanation
    #turn a list into a CommonsenseExample class object
    @classmethod
    def from_list(cls, o, train=True, explain_predict=True):
        ex, ak, at = None, None, None
        q, c0, c1, c2 = o[1:5]
        if train:
            ak = o[5]
            ex = o[6]
            at = [c0, c1, c2][int(ak)]
        return CommonsenseExample(q, c0, c1, c2, at, ex, explain_predict=explain_predict)
    #turn yourself into a tokenized variant of yourself, in case you need it
    def tokenize(self, tokenizer):
        self.text = tokenizer.tokenize(self.text)
        self.question = tokenizer.tokenize(self.question)
        self.choice0 = tokenizer.tokenize(self.choice0)
        self.choice1 = tokenizer.tokenize(self.choice1)
        self.choice2 = tokenizer.tokenize(self.choice2)
        if self.explanation is not None:
            self.answer_text = tokenizer.tokenize(self.answer_text)
            self.explanation = tokenizer.tokenize(self.explanation)
        return self
    #turn yourself into a numerical version of yourself, in case you need it
    def numericalize(self, tokenizer):
        self.text = tokenizer.convert_tokens_to_ids(self.text)
        self.question = tokenizer.convert_tokens_to_ids(self.question)
        self.choice0 = tokenizer.convert_tokens_to_ids(self.choice0)
        self.choice1 = tokenizer.convert_tokens_to_ids(self.choice1)
        self.choice2 = tokenizer.convert_tokens_to_ids(self.choice2)
        if self.explanation is not None:
            self.answer_text = tokenizer.convert_tokens_to_ids(self.answer_text)
            self.explanation = tokenizer.convert_tokens_to_ids(self.explanation)
        return self

    def natural_tensor(self, end_token):
        return self.text  if self.explanation is None else self.text + self.explanation + [end_token]

    @classmethod
    def tokenize_list(cls, l, tokenizer):
        return [x.tokenize(tokenizer) for x in l]

    @classmethod
    def numericalize_list(cls, l, tokenizer):
        return [x.numericalize(tokenizer) for x in l]

    def __repr__(self):
        return f'question: {self.question}\nchoices: {[self.choice0, self.choice1, self.choice2]}\nanswer: {self.answer_text}\nexplanation: {self.explanation}'

#Reading and structuring the csv files, depending on the "mode" selected during argument parsing and turning them into a CommonsenseExample class object
#explain_predict does not appear to make any difference, however
def parse_cqa(root_path, explain_predict):
    splits = ['train.csv', 'dev.csv', 'test.csv']
    split_data = []
    #for train, dev and test, iterate
    for split in splits:
        path = os.path.join(root_path, split) #select train, dev, or test through "split" 
        with open(path, encoding='utf_8') as f:
            f = csv.reader(f)
            examples = []
            next(f) # skip the first line
            for line in tqdm(f):
                examples.append(CommonsenseExample.from_list(line, train=split != 'test', explain_predict=explain_predict))
        split_data.append(examples)
    return split_data

#create a sentence of "length" based on prompts, using the model's predictive power
def sample(model, prompts, length, device):
    preds = []
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts.tolist()):
            preds.append([])
            #remove padding tokens and move to GPU/CPU if available
            prompt = torch.tensor([p for p in prompt if p > -1], dtype=torch.long).to(device)
            for i in range(length):
                logits = model(prompt.unsqueeze(0)).squeeze(0)
                #select most likely token
                pred = torch.max(logits, dim=-1)[1][-1]
                preds[prompt_idx].append(pred.item())
                prompt = torch.cat([prompt, pred.unsqueeze(0)], dim=-1)
    return preds

# Added/modified functions attempting to add top-k instead of using torch.max() as a decoder during testing.
# This might fix the looping predictions problem

def top_k_logits(logits, k):
    """Keep only the top-k highest probability logits, set the rest to a very low value."""
    if k == 0:
        return logits  # No filtering
    values, _ = torch.topk(logits, k)  # Get the top-k largest values
    min_values = values[:, -1].unsqueeze(1)  # Get the smallest of the top-k values
    return torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)  # Suppress lower values

def sample_top_k(model, prompts, length, device, k=40):
    preds = []
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(prompts.tolist()):
            preds.append([])
            prompt = torch.tensor([p for p in prompt if p > -1], dtype=torch.long).to(device)
            
            for _ in range(length):
                logits = model(prompt.unsqueeze(0)).squeeze(0)
                logits = top_k_logits(logits, k)  # Apply Top-k sampling
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred = torch.multinomial(probs, num_samples=1)  # Randomly select from top-k

                pred = pred.squeeze()  # Ensure it's a scalar
                if pred.dim() > 0:  # If still not a scalar, take first element
                    pred = pred[0]
                preds[prompt_idx].append(pred.item())  # Store the token ID
                prompt = torch.cat([prompt, pred.unsqueeze(0)], dim=-1)  # Append new token
    return preds



#Main Programm
def run_model():
    parser = argparse.ArgumentParser()
    # REQUIRED
    parser.add_argument('--model_name', type=str, default='openai-gpt', #You have the option to either use the GPT model...
                        help='pretrained model name or path to local checkpoint') #...or to load an already trained model
    parser.add_argument('--data', type=str, default='/stage/examples/commonsenseqa/') #Directory of the data :)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Select at least one of:
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    
    # NOT REQUIRED
    # Hyperparameters
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=36)
    parser.add_argument('--eval_batch_size', type=int, default=60)
    parser.add_argument("--n_gen", type=int, default=20) #How long the generated output can get at maximum (in tokens)
    parser.add_argument('--max_grad_norm', type=int, default=1) #Gradient clipping to prevent exploding gradients problem
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--lr_schedule', type=str, 
                        default='warmup_linear') #Enables manipulation of the learning rate scheduling, so it can change during training.
    parser.add_argument('--warmup_proportion', type=float, default=0.002) 
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # Logging
    parser.add_argument("--n_train_print", type=int, default=10) #The console prints out every generated answer during training. 
    parser.add_argument('--num_eval_print', type=int, default=15) #These two determine how many answers are shown at the same time
    # Additional Settings
    parser.add_argument('--setting', type=str, default='explain_predict') #Whether the model uses "reasoning" or "rationalization"
    parser.add_argument('--eval_preds_prefix', type=str, default='preds') #added by me because otherwise we won't be able to write any of the results down
    parser.add_argument('--test_preds_prefix', type=str, default='test') #also added by me
    parser.add_argument('--seed', type=int, default=42) #random seed. Helps reproducing results, ideally. 
    args = parser.parse_args()
    print(args)
    
    #Using the same seed should ensure the same result every time we run this thing
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #Whether we use cuda or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    
    #Ensure that we are actually doing something with this script
    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval`  or do_test must be True.")

    #Create the directory where we save the model and such, if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #Load GPT as the tokenizer
    special_tokens = ['_start_</w>', 'or</w>', '_answer_</w>', '_classify_</w>', '_end_</w>']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)

    #loading the dataset
    datasets = parse_cqa(args.data, args.setting)
    #turn labels for A, B, C into corresponding numbers
    numericalized = [CommonsenseExample.numericalize_list(CommonsenseExample.tokenize_list(d, tokenizer), tokenizer) for d in datasets]

    tensor_datasets = pre_process_datasets(numericalized, *special_tokens_ids)

#    train_tensor_dataset, eval_tensor_dataset, test_tensor_dataset = tensor_datasets[0], tensor_datasets[1], tensor_datasets[2]
    #load tensors, dataset, sampler, etc. etc. for training, evaluation, testing
    if args.do_train:
        train_tensor_dataset = tensor_datasets[0]
        train_data = TensorDataset(*train_tensor_dataset)
        train_sampler = RandomSampler(train_data)
        if args.do_train:
            train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.do_eval:
        eval_tensor_dataset = tensor_datasets[1]
        eval_data = TensorDataset(*eval_tensor_dataset)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.do_test:
        test_tensor_dataset = tensor_datasets[-1]
        test_data = TensorDataset(*test_tensor_dataset)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)


    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)


    #Defining functions to convert logits into something readable
    def trim_unks(x):
        try: 
            unk_id = x.index('_end_</w>')
            return x[:unk_id]
        except:
            return x
    def detokenize(x):
        y = ''.join(trim_unks(x))
        y = y.replace('</w>', ' ')
        y = y.replace(' .', '.')
        y = y.replace(' ,', ',')
        y = y.replace(' ?', '?')
        y = y.replace(' !', '!')
        y = y.replace(' \' ', '\'')
        y = y.replace(' \'re', '\'re')
        y = y.replace(' \'s', '\'s')
        y = y.replace(' n\'t', 'n\'t')
        return y
    #Detokenizing an entire batch to make it more readable for your human eyes. 
    def detok_batch(x):
        if not isinstance(x, list):
            x = x.tolist()
        return [detokenize(tokenizer.convert_ids_to_tokens([z for z in y if z >= 0])) for y in x]
    



    #training loop!
    if args.do_train:
        best_eval = 0
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        #Put the model into training mode
        model.train()
        #for every training epoch...
        for _ in range(int(args.num_train_epochs)):
            #reset loss, perplexity, training examples
            tr_loss, train_ppl, n_train_examples = 0, 0, 0

            #reset training steps
            nb_tr_steps = 0

            #create progress bar
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            
            train_pred_strs, train_lab_strs = [], []

            #for each batch...
            for batch in enumerate(tqdm_bar):
                #load inputs and labels
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                #calculate loss
                loss = model(inputs, lm_labels=labels)
                #calculate perplexity
                train_ppl += loss.item() * inputs.size(0)
                n_train_examples += inputs.size(0)

                #back propagation and optimizer to improve model parameters
                loss.backward()
                optimizer.step()
                #update training loss, average loss
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                #one more training step done, so add that
                nb_tr_steps += 1


                #if printing out the training results is desired, do that as much as the user asked
                if args.n_train_print > 0:
                    with torch.no_grad():
                        preds = sample(model, batch[2], 10, device)
                    #detokenize, so its readable
                    pred_str = detok_batch(preds)
                    label_str = detok_batch(labels)
                    train_lab_strs.extend(label_str)
                    train_pred_strs.extend(pred_str)
                    input_str = detok_batch(inputs)
                    #and print it all into a beautiful format
                    for print_idx in range(min(args.n_train_print, inputs.size(0))):
                        print('INPT: ', input_str[print_idx])
                        print('GOLD: ', label_str[print_idx])
                        print('PRED: ', pred_str[print_idx])
                        print()
            #reset BLEU score
            train_bleu = None
            #saving the evaluation results (BLEU and perplexity) to print out later
            if args.n_train_print > 0:
                train_bleu = computeBLEU(train_pred_strs, [[x] for x in train_lab_strs])
                train_ppl = math.exp(train_ppl / n_train_examples) 


            #evaluation
            if args.do_eval:
                #put model into evaluation mode
                model.eval()
                #create loss, exact match, perplexity
                eval_loss, eval_em, eval_ppl = 0, 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                label_strs, prediction_strs = [], []
                for batch in eval_dataloader:
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)

                    with torch.no_grad():
                        loss = model(inputs, lm_labels=labels)
                        preds = sample(model, batch[2], args.n_gen, device)

                    eval_loss += loss.item()
                    eval_ppl += loss.item() * inputs.size(0)
                    nb_eval_examples += inputs.size(0)
                    nb_eval_steps += 1
                    pred_str = detok_batch(preds)
                    label_str = detok_batch(labels)
                    label_strs.extend(label_str)
                    prediction_strs.extend(pred_str)
                    input_str = detok_batch(inputs)
                    eval_em += sum([x == y for x, y in zip(pred_str, label_str)]) 
                    for print_idx in range(min(inputs.size(0), args.num_eval_print)):
                        print('INPT: ', input_str[print_idx])
                        print('GOLD: ', label_str[print_idx])
                        print('PRED: ', pred_str[print_idx])
                        print()

                eval_bleu = computeBLEU(prediction_strs, [[x] for x in label_strs])
                eval_ppl = math.exp(eval_ppl / nb_eval_examples) 
                eval_em = eval_em / nb_eval_examples
                eval_loss = eval_loss / nb_eval_steps
                train_loss = tr_loss/nb_tr_steps if args.do_train else None
                result = {'eval_loss': eval_loss,
                         'eval_em': eval_em,
                         'eval_bleu': eval_bleu,
                         'eval_ppl': eval_ppl,
                         'train_loss': train_loss, 
                         'train_bleu': train_bleu, 
                         'train_ppl': train_ppl}
        
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

                if eval_bleu > best_eval:
                    best_eval = eval_bleu 

                    # Save a trained model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
                    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                    config = model.config
                    torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_eval:
        # Load a trained model that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = OpenAIGPTLMHeadModel(model.config)
        model.load_state_dict(model_state_dict)


        # uncomment to try out the default not finue-tuned model
        #model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens), cache_dir=os.path.dirname(args.data))
        model.to(device)
        model.eval()
        
        eval_loss, eval_em, eval_ppl = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_strs, prediction_strs = [], []
        for batch in eval_dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            with torch.no_grad():
                loss = model(inputs, lm_labels=labels)
                preds = sample(model, batch[2], args.n_gen, device)

            eval_loss += loss.item()
            eval_ppl += loss.item() * inputs.size(0)
            nb_eval_examples += inputs.size(0)
            nb_eval_steps += 1
            pred_str = detok_batch(preds)
            label_str = detok_batch(labels)
            label_strs.extend(label_str)
            prediction_strs.extend(pred_str)
            input_str = detok_batch(inputs)
            eval_em += sum([x == y for x, y in zip(pred_str, label_str)]) 
            for print_idx in range(min(inputs.size(0), args.num_eval_print)):
                print('INPT: ', input_str[print_idx])
                print('GOLD: ', label_str[print_idx])
                print('PRED: ', pred_str[print_idx])
                print()

        eval_bleu = computeBLEU(prediction_strs, [[x] for x in label_strs])
        eval_ppl = math.exp(eval_ppl / nb_eval_examples) 
        eval_em = eval_em / nb_eval_examples
        eval_loss = eval_loss / nb_eval_steps
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                 'eval_em': eval_em,
                 'eval_bleu': eval_bleu,
                 'eval_ppl': eval_ppl,
                 'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Best Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_preds_file = os.path.join(args.output_dir, f"{args.eval_preds_prefix}_{args.setting}.txt")
        with open(output_preds_file, 'w') as writer:
            logger.info("Writing predictions")
            for p in prediction_strs:
                writer.write(p + '\n')


    if args.do_test:
        # Load a trained model that you have fine-tuned
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        model = OpenAIGPTLMHeadModel(model.config)
        model.load_state_dict(model_state_dict)


        model.to(device)
        model.eval()
        eval_loss, eval_em, eval_ppl = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_strs, prediction_strs = [], []
        for batch in test_dataloader:
            inputs = batch[0].to(device)

            with torch.no_grad():
                preds = sample(model, batch[1], args.n_gen, device)
                #uncomment to try out top_k
                #preds = sample_top_k(model, batch[1], args.n_gen, device, k=40)
            #turn predictions into a string
            pred_str = detok_batch(preds)
            prediction_strs.extend(pred_str)

        output_preds_file = os.path.join(args.output_dir, f"{args.test_preds_prefix}_{args.setting}.txt")
        with open(output_preds_file, 'w') as writer:
            logger.info("Writing predictions")
            for p in prediction_strs:
                writer.write(f'"{p.strip()}"\n')

if __name__ == '__main__':
    run_model()
