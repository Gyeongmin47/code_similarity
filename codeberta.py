import argparse

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
import time
import datetime
from transformers import *
import seaborn as sns

train = pd.read_csv("sample_train.csv")
test = pd.read_csv("test.csv")

code_folder = 'code'
problem_folders = os.listdir(code_folder)


parser = argparse.ArgumentParser()
parser.add_argument('--output-prefix', default=str)
parser.add_argument('--resume-from', default=str)

args = parser.parse_args()

if args.resum_from:
    # load model
    model_data = torch.load(args.resume_from)



def preprocess_script(script):
    '''
    간단한 전처리 함수
    주석 -> 삭제
    '    '-> tab 변환
    다중 개행 -> 한 번으로 변환
    '''
    with open(script,'r',encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n','')
            line = line.replace('    ','\t')
            if line == '':
                continue
            preproc_lines.append(line)
        preprocessed_script = '\n'.join(preproc_lines)
    return preprocessed_script

preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder,problem_folder))
    problem_num = scripts[0].split('_')[0]
    for script in scripts:
        script_file = os.path.join(code_folder,problem_folder,script)
        preprocessed_script = preprocess_script(script_file)

        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num]*len(scripts))

import pandas as pd
df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})

tokenizer = AutoTokenizer.from_pretrained("mrm8488/CodeBERTaPy")
df['tokens'] = df['code'].apply(tokenizer.tokenize)
df['len'] = df['tokens'].apply(len)
df.describe()

ndf = df[df['len'] <= 512].reset_index(drop=True)
ndf.describe()

ndf.head()

# train test split
from sklearn.model_selection import train_test_split

train_df, valid_df, train_label, valid_label = train_test_split(
        ndf,
        ndf['problem_num'],
        random_state=42,
        test_size=0.1,
        stratify=ndf['problem_num'],
    )

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# 참고: https://dacon.io/competitions/official/235900/codeshare/4907?page=1&dtype=recent
from rank_bm25 import BM25Okapi
from itertools import combinations
codes = train_df['code'].to_list()
problems = train_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = train_df[train_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(),2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
    ranking_idx = 0

    for solution_code in solution_codes:
        negative_solutions = []
        while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
            high_score_idx = negative_code_ranking[ranking_idx]

            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(train_df['code'].iloc[high_score_idx])
            ranking_idx += 1

        for negative_solution in negative_solutions:
            negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1]*len(pos_code1)
neg_label = [0]*len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={
    'code1':total_code1,
    'code2':total_code2,
    'similar':total_label
})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)

pair_data.to_csv('train_data.csv',index=False)


# repeat for validation
codes = valid_df['code'].to_list()
problems = valid_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = valid_df[valid_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(),2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1] # 내림차순
    ranking_idx = 0

    for solution_code in solution_codes:
        negative_solutions = []
        while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
            high_score_idx = negative_code_ranking[ranking_idx]

            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(valid_df['code'].iloc[high_score_idx])
            ranking_idx += 1

        for negative_solution in negative_solutions:
            negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

pos_code1 = list(map(lambda x:x[0],total_positive_pairs))
pos_code2 = list(map(lambda x:x[1],total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1]*len(pos_code1)
neg_label = [0]*len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={
    'code1':total_code1,
    'code2':total_code2,
    'similar':total_label
})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)

pair_data.to_csv('valid_data.csv',index=False)

# read saved train and validation data
train_data = pd.read_csv("train_data.csv")
valid_data = pd.read_csv("valid_data.csv")


c1 = train_data['code1'].values
c2 = train_data['code2'].values
similar = train_data['similar']

N = train_data.shape[0]
MAX_LEN = 512

input_ids = np.zeros((N, MAX_LEN),dtype=int)
attention_masks = np.zeros((N, MAX_LEN),dtype=int)
labels = np.zeros((N),dtype=int)

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = c1[i]
        cur_c2 = c2[i]
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids[i,] = encoded_input['input_ids']
        attention_masks[i,] = encoded_input['attention_mask']
        labels[i] = similar[i]
    except Exception as e:
        print(e)
        pass


# import telegram
# tel_token = "5059732158:AAE87TaReNbDKH3_Fy-CAYCUuIO2qiUyK2I"
# chat_id = 1720119057
# bot = telegram.Bot(token=tel_token)

# bot.sendMessage(chat_id=chat_id, text="train preprocessing done!")

c1 = valid_data['code1'].values
c2 = valid_data['code2'].values
similar = valid_data['similar']

N = valid_data.shape[0]
MAX_LEN = 512

valid_input_ids = np.zeros((N, MAX_LEN),dtype=int)
valid_attention_masks = np.zeros((N, MAX_LEN),dtype=int)
valid_labels = np.zeros((N),dtype=int)

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = c1[i]
        cur_c2 = c2[i]
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        valid_input_ids[i,] = encoded_input['input_ids']
        valid_attention_masks[i,] = encoded_input['attention_mask']
        valid_labels[i] = similar[i]
    except Exception as e:
        print(e)
        pass
input_ids = torch.tensor(input_ids, dtype=int)
attention_masks = torch.tensor(attention_masks, dtype=int)
labels = torch.tensor(labels, dtype=int)

valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
valid_labels = torch.tensor(valid_labels, dtype=int)


torch.save(input_ids, 'train_input_ids.pt')
torch.save(attention_masks, 'train_attention_masks.pt')
torch.save(labels, "train_labels.pt")

torch.save(valid_input_ids, "valid_input_ids.pt")
torch.save(valid_attention_masks, "valid_attention_masks.pt")
torch.save(valid_labels, "valid_labels.pt")

'''
# load saved models
input_ids = torch.load('train_input_ids.pt')
attention_masks = torch.load('train_attention_masks.pt')
labels = torch.load('train_labels.pt')

valid_input_ids = torch.load('valid_input_ids.pt')
valid_attention_masks = torch.load('valid_attention_masks.pt')
valid_labels = torch.load('valid_labels.pt')
'''

batch_size = 32
train_data = TensorDataset(input_ids, attention_masks, labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained("mrm8488/CodeBERTaPy")
model.cuda()

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 10

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


device = torch.device("cuda")
loss_f = nn.CrossEntropyLoss()
##### train #####
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
model.zero_grad()

for i in range(epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(i + 1, epochs))
    print('Training...')
    t0 = time.time()
    train_loss, train_accuracy = 0, 0
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", smoothing=0.05):
        if step % 10000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            print('  current average loss = {}'.format(train_loss / step))
            # bot.sendMessage(chat_id=chat_id, text = '  current average loss = {}'.format(train_loss / step))

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        train_accuracy += flat_accuracy(logits, label_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    avg_train_loss = train_loss / len(train_dataloader)
    avg_train_accuracy = train_accuracy / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    print("  Average training loss: {0:.8f}".format(avg_train_loss))
    print("  Average training accuracy: {0:.8f}".format(avg_train_accuracy))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))


    print("")
    print("Validating...")
    t0 = time.time()
    model.eval()
    val_loss, val_accuracy = 0, 0
    for step, batch in tqdm(enumerate(validation_dataloader), desc="Iteration", smoothing=0.05):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu()
        label_ids = b_labels.detach().cpu()
        val_loss += loss_f(logits, label_ids)

        logits = logits.numpy()
        label_ids = label_ids.numpy()
        val_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = val_accuracy / len(validation_dataloader)
    avg_val_loss = val_loss / len(validation_dataloader)
    val_accuracies.append(avg_val_accuracy)
    val_losses.append(avg_val_loss)
    print("  Average validation loss: {0:.8f}".format(avg_val_loss))
    print("  Average validation accuracy: {0:.8f}".format(avg_val_accuracy))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # bot.sendMessage(chat_id=chat_id, text="Epoch {} Done!".format(i+1))
    # bot.sendMessage(chat_id=chat_id, text="avg validation loss = {}".format(avg_val_loss))
    # bot.sendMessage(chat_id=chat_id, text="avg validation accuracy = {}%".format(avg_val_accuracy))

    if np.min(val_losses) == val_losses[-1]:
        print("saving current best checkpoint")
        torch.save(model.state_dict(), "codebertapy.pt")
