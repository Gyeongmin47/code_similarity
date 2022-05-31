import numpy as np
import os
import pandas as pd
from transformers import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import torch
import torch.nn as nn
from rank_bm25 import BM25Okapi
from itertools import combinations
import seaborn as sns
from setproctitle import setproctitle

setproctitle("Gyeongmin")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

code_folder = "./data/code"
problem_folders = os.listdir(code_folder)


def preprocess_script(script):
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
        preprocessed_script = re.sub(r'\"\"\"([^\"]*)\"\"\"', "", preprocessed_script) # get rid of triple quoted comments

    return preprocessed_script


cnt = 0
for i in range(300):
    cnt += len(os.listdir('./data/code/' + problem_folders[i]))

preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):

    scripts = os.listdir(os.path.join(code_folder, problem_folder))
    problem_num = scripts[0].split('_')[0]
    for script in scripts:
        script_file = os.path.join(code_folder, problem_folder, script)
        preprocessed_script = preprocess_script(script_file)
        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num]*len(scripts))

df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})


tokenizer = AutoTokenizer.from_pretrained("mrm8488/CodeBERTaPy")
df['tokens'] = df['code'].apply(tokenizer.tokenize)
df['len'] = df['tokens'].apply(len)
df.describe()

# load model
sim_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/CodeBERTaPy")
checkpoint = torch.load("./data/CodeBERTaPy/codebertapy.pt")
sim_model.load_state_dict(checkpoint)
sim_model.eval()
sim_model.cuda()

train_df, valid_df, train_label, valid_label = train_test_split(
    df,
    df['problem_num'],
    random_state = 42,
    test_size = 0.1,
    stratify = df['problem_num']
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)


# Get train Pairs

# needed for semantic search
device = torch.device('cuda')
softmax = nn.Softmax(dim=1)
n = 255  # number of groups
m = 50  # overlap

codes = train_df['code'].to_list()
problems = train_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs, total_negative_pairs = [], []

for problem in tqdm(problems[:1], position=0, leave=True):
    solution_codes = train_df[train_df['problem_num'] == problem]['code']
    positive_pairs = []
    negative_pairs = []
    solution_codes_indices = solution_codes.index.to_list()
    solution_codes = solution_codes.values
    for i in tqdm(range(len(solution_codes)), position=0, leave=True):
        positive_cnt = 0
        for j in range(i + 1, len(solution_codes)):
            c1 = solution_codes[i]
            c2 = solution_codes[j]
            input_ids_c1 = tokenizer(c1)['input_ids']
            input_ids_c2 = tokenizer(c2)['input_ids']
            if len(input_ids_c1) + len(input_ids_c2) <= 512:
                positive_pairs.append((c1, c2))
            else:
                splitted_c1 = [input_ids_c1[i:i + n] for i in range(0, len(input_ids_c1), n - m)]
                splitted_c2 = [input_ids_c2[i:i + n] for i in range(0, len(input_ids_c2), n - m)]
                max_score = -1e9
                max_pairs = ((-1, -1))
                for x in splitted_c1:
                    for y in splitted_c2:
                        x_str = str(tokenizer.decode(x))
                        y_str = str(tokenizer.decode(y))
                        encoded_inputs = tokenizer(x_str, y_str, return_tensors="pt", max_length=512,
                                                   padding="max_length", truncation=True).to(device)
                        with torch.no_grad():
                            output = sim_model(**encoded_inputs)
                            prob = softmax(output['logits'])
                            if prob[0][1].item() > max_score:
                                max_score = prob[0][1].item()
                                max_pairs = (x_str, y_str)
                positive_pairs.append(max_pairs)
            positive_cnt += 1
        ''' getting negative pairs '''
        c1 = solution_codes[i]
        current_tokenized_solution_code = tokenizer.tokenize(c1)


        negative_code_sources = bm25.get_scores(current_tokenized_solution_code)
        negative_code_ranking = negative_code_sources.argsort()[::-1]
        ranking_idx = 0
        negative_solutions = []
        while len(negative_solutions) < positive_cnt:
            high_score_idx = negative_code_ranking[ranking_idx]
            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(train_df['code'].iloc[high_score_idx])
            ranking_idx += 1
        input_ids_c1 = tokenizer(c1)['input_ids']

        for negative_solution in negative_solutions:
            neg_tokenized = tokenizer(negative_solution)['input_ids']
            if len(input_ids_c1) + len(neg_tokenized) <= 512:
                negative_pairs.append((c1, negative_solution))
            else:
                splitted_c1 = [input_ids_c1[i:i + n] for i in range(0, len(input_ids_c1), n - m)]
                splitted_c2 = [input_ids_c2[i:i + n] for i in range(0, len(input_ids_c2), n - m)]
                max_score = -1e9
                max_pairs = ((-1, -1))
                for x in splitted_c1:
                    for y in splitted_c2:
                        x_str = str(tokenizer.decode(x))
                        y_str = str(tokenizer.decode(y))
                        encoded_inputs = tokenizer(x_str, y_str, return_tensors="pt", max_length=512,
                                                   padding="max_length", truncation=True).to(device)
                        with torch.no_grad():
                            output = sim_model(**encoded_inputs)
                            prob = softmax(output['logits'])
                            if prob[0][1].item() > max_score:
                                max_score = prob[0][1].item()
                                max_pairs = (x_str, y_str)
                negative_pairs.append(max_pairs)

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)


pos_code1 = list(map(lambda x:x[0], total_positive_pairs))
pos_code2 = list(map(lambda x:x[1], total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1] * len(pos_code1)
neg_label = [0] * len(neg_code1)

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

pair_data.to_csv("./data/new_dataset/dacon_code_train_data.csv", index=False)

codes = valid_df['code'].to_list()

problems = valid_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs, total_negative_pairs = [], []

for problem in tqdm(problems[:1], position=0, leave=True):
    solution_codes = valid_df[valid_df['problem_num'] == problem]['code']
    positive_pairs = []
    negative_pairs = []
    solution_codes_indices = solution_codes.index.to_list()
    solution_codes = solution_codes.values
    for i in tqdm(range(len(solution_codes)), position=0, leave=True):
        positive_cnt = 0
        for j in range(i + 1, len(solution_codes)):
            c1 = solution_codes[i]
            c2 = solution_codes[j]
            input_ids_c1 = tokenizer(c1)['input_ids']
            input_ids_c2 = tokenizer(c2)['input_ids']
            if len(input_ids_c1) + len(input_ids_c2) <= 512:
                positive_pairs.append((c1, c2))
            else:
                splitted_c1 = [input_ids_c1[i:i + n] for i in range(0, len(input_ids_c1), n - m)]
                splitted_c2 = [input_ids_c2[i:i + n] for i in range(0, len(input_ids_c2), n - m)]
                max_score = -1e9
                max_pairs = ((-1, -1))
                for x in splitted_c1:
                    for y in splitted_c2:
                        x_str = str(tokenizer.decode(x))
                        y_str = str(tokenizer.decode(y))
                        encoded_inputs = tokenizer(x_str, y_str, return_tensors="pt", max_length=512,
                                                   padding="max_length", truncation=True).to(device)
                        with torch.no_grad():
                            output = sim_model(**encoded_inputs)
                            prob = softmax(output['logits'])
                            if prob[0][1].item() > max_score:
                                max_score = prob[0][1].item()
                                max_pairs = (x_str, y_str)
                positive_pairs.append(max_pairs)
            positive_cnt += 1
        ''' getting negative pairs '''
        c1 = solution_codes[i]
        current_tokenized_solution_code = tokenizer.tokenize(c1)
        negative_code_sources = bm25.get_scores(current_tokenized_solution_code)
        negative_code_ranking = negative_code_sources.argsort()[::-1]
        ranking_idx = 0
        negative_solutions = []
        while len(negative_solutions) < positive_cnt:
            high_score_idx = negative_code_ranking[ranking_idx]
            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(valid_df['code'].iloc[high_score_idx])
            ranking_idx += 1
        input_ids_c1 = tokenizer(c1)['input_ids']
        for negative_solution in negative_solutions:
            neg_tokenized = tokenizer(negative_solution)['input_ids']
            if len(input_ids_c1) + len(neg_tokenized) <= 512:
                negative_pairs.append((c1, negative_solution))
            else:
                splitted_c1 = [input_ids_c1[i:i + n] for i in range(0, len(input_ids_c1), n - m)]
                splitted_c2 = [input_ids_c2[i:i + n] for i in range(0, len(input_ids_c2), n - m)]
                max_score = -1e9
                max_pairs = ((-1, -1))
                for x in splitted_c1:
                    for y in splitted_c2:
                        x_str = str(tokenizer.decode(x))
                        y_str = str(tokenizer.decode(y))
                        encoded_inputs = tokenizer(x_str, y_str, return_tensors="pt", max_length=512,
                                                   padding="max_length", truncation=True).to(device)
                        with torch.no_grad():
                            output = sim_model(**encoded_inputs)
                            prob = softmax(output['logits'])
                            if prob[0][1].item() > max_score:
                                max_score = prob[0][1].item()
                                max_pairs = (x_str, y_str)
                negative_pairs.append(max_pairs)

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)


pos_code1 = list(map(lambda x:x[0], total_positive_pairs))
pos_code2 = list(map(lambda x:x[1], total_positive_pairs))

neg_code1 = list(map(lambda x:x[0],total_negative_pairs))
neg_code2 = list(map(lambda x:x[1],total_negative_pairs))

pos_label = [1] * len(pos_code1)
neg_label = [0] * len(neg_code1)

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

pair_data.to_csv("./data/new_dataset/dacon_code_valid_data.csv",index=False)