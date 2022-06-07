import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split
from transformers import *
from rank_bm25 import BM25Okapi, BM25L
from itertools import combinations

from setproctitle import setproctitle
setproctitle("Gyeongmin")

# define utility functions
def get_rid_of_empty(c):
    ret = []
    splitted = c.split('\n')
    for s in splitted:
        if len(s.strip()) > 0:
            ret.append(s)
    return '\n'.join(ret)


def preprocess_script(script):
    with open(script, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        preproc_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n', '')
            line = line.replace('    ', '\t')
            if line == '':
                continue
            preproc_lines.append(line)

        preprocessed_script = '\n'.join(preproc_lines)

        preprocessed_script = re.sub('\"\"\"([^\"]*)\"\"\"', "", preprocessed_script)
        preprocessed_script = re.sub('\'\'\'([^\"]*)\'\'\'', "", preprocessed_script)
        preprocessed_script = re.sub(r'\'\w+', '', preprocessed_script)
        preprocessed_script = re.sub(r'\w*\d+\w*', '', preprocessed_script)
        preprocessed_script = re.sub(r'\s{2,}', ' ', preprocessed_script)
        preprocessed_script = re.sub(r'\s[^\w\s]\s', '', preprocessed_script)

        splitted = preprocessed_script.split('\n')
        found_triple = False
        start_idx, end_idx = -1, -1
        for i in range(len(splitted)):
            if found_triple == False and '\'\'\'' in splitted[i]:
                found_triple = True
                start_idx = i
            elif found_triple == True and '\'\'\'' in splitted[i]:
                end_idx = i
        if start_idx != -1 and end_idx != -1:
            splitted = splitted[:start_idx] + splitted[end_idx + 1:]
        elif start_idx != -1 and end_idx == -1:
            splitted = splitted[start_idx + 1:]

        found_triple = False
        start_idx, end_idx = -1, -1
        for i in range(len(splitted)):
            if found_triple == False and '\"\"\"' in splitted[i]:
                found_triple = True
                start_idx = i
            elif found_triple == True and '\"\"\"' in splitted[i]:
                end_idx = i
        if start_idx != -1 and end_idx != -1:
            splitted = splitted[:start_idx] + splitted[end_idx + 1:]
        elif start_idx != -1 and end_idx == -1:
            splitted = splitted[start_idx + 1:]

        preprocessed_script = '\n'.join(splitted)

        preprocessed_script = get_rid_of_empty(preprocessed_script)

    return preprocessed_script


# used for preprocessing test data
def preprocess_test(script):
    lines = script.split('\n')
    preproc_lines = []
    for line in lines:
        if line.lstrip().startswith('#'):
            continue
        line = line.rstrip()
        if '#' in line:
            line = line[:line.index('#')]
        line = line.replace('\n', '')
        line = line.replace('    ', '\t')
        if line == '':
            continue
        preproc_lines.append(line)

    preprocessed_script = '\n'.join(preproc_lines)
    preprocessed_script = re.sub('\"\"\"([^\"]*)\"\"\"', "", preprocessed_script)
    preprocessed_script = re.sub('\'\'\'([^\"]*)\'\'\'', "", preprocessed_script)
    preprocessed_script = re.sub(r'\'\w+', '', preprocessed_script)
    preprocessed_script = re.sub(r'\w*\d+\w*', '', preprocessed_script)
    preprocessed_script = re.sub(r'\s{2,}', ' ', preprocessed_script)
    preprocessed_script = re.sub(r'\s[^\w\s]\s', '', preprocessed_script)

    splitted = preprocessed_script.split('\n')
    found_triple = False
    start_idx, end_idx = -1, -1
    for i in range(len(splitted)):
        if found_triple == False and '\'\'\'' in splitted[i]:
            found_triple = True
            start_idx = i
        elif found_triple == True and '\'\'\'' in splitted[i]:
            end_idx = i
    if start_idx != -1 and end_idx != -1:
        splitted = splitted[:start_idx] + splitted[end_idx + 1:]
    elif start_idx != -1 and end_idx == -1:
        splitted = splitted[start_idx + 1:]

    found_triple = False
    start_idx, end_idx = -1, -1
    for i in range(len(splitted)):
        if found_triple == False and '\"\"\"' in splitted[i]:
            found_triple = True
            start_idx = i
        elif found_triple == True and '\"\"\"' in splitted[i]:
            end_idx = i
    if start_idx != -1 and end_idx != -1:
        splitted = splitted[:start_idx] + splitted[end_idx + 1:]
    elif start_idx != -1 and end_idx == -1:
        splitted = splitted[start_idx + 1:]

    preprocessed_script = '\n'.join(splitted)
    preprocessed_script = get_rid_of_empty(preprocessed_script)
    return preprocessed_script


# get processed train data
code_folder = "data/code"
problem_folders = os.listdir(code_folder)
preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder, problem_folder))
    problem_num = scripts[0].split('_')[0]
    for script in scripts:
        script_file = os.path.join(code_folder, problem_folder, script)
        preprocessed_script = preprocess_script(script_file)
        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num] * len(scripts))

train_df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums})


# get processed test data
test_df = pd.read_csv("data/test.csv")
code1 = test_df['code1'].values
code2 = test_df['code2'].values

processed_code1 = []
processed_code2 = []

for i in tqdm(range(len(code1))):
    processed_c1 = preprocess_test(code1[i])
    processed_c2 = preprocess_test(code2[i])
    processed_code1.append(processed_c1)
    processed_code2.append(processed_c2)

processed_test = pd.DataFrame(list(zip(processed_code1, processed_code2)), columns=["code1", "code2"])


# get processed codenet code

code_folder = "data/Project_CodeNet_Python800"
problem_folders = os.listdir(code_folder)

preproc_scripts = []
problem_nums = []

for problem_folder in tqdm(problem_folders):
    scripts = os.listdir(os.path.join(code_folder,problem_folder))
    problem_num = int(problem_folder.split('p')[1])
    problem_num = 'problem' + str(problem_num)
    for script in scripts:
        script_file = os.path.join(code_folder,problem_folder,script)
        preprocessed_script = preprocess_script(script_file)
        preproc_scripts.append(preprocessed_script)
    problem_nums.extend([problem_num]*len(scripts))

codenet_df = pd.DataFrame(data = {'code':preproc_scripts, 'problem_num':problem_nums})

dacon_codes = np.concatenate([train_df['code'].values, test_df['code1'].values, test_df['code2'].values])

dacon_codes_set = set()

for i in tqdm(range(len(dacon_codes))):
    dacon_codes_set.add(dacon_codes[i])

usable_codes = []
usable_problem_nums = []

codenet_codes = codenet_df['code'].values
problem_nums = codenet_df['problem_num'].values

for i in tqdm(range(len(codenet_codes))):
    if codenet_codes[i] not in dacon_codes_set:
        usable_codes.append(codenet_codes[i])
        usable_problem_nums.append(problem_nums[i])


filtered_codenet_df = pd.DataFrame(data = {'code':usable_codes, 'problem_num':usable_problem_nums})

filtered_codenet_df.sample(frac=0.5, random_state=42)

# filtered_codenet_df.to_csv("data/Filtered_CodeNet_df.csv", index=False)


## 여기아래부터 기존 그대로

train_df, valid_df, train_label, valid_label = train_test_split(
    filtered_codenet_df,
    filtered_codenet_df['problem_num'],
    random_state = 888,
    test_size = 0.1,
    stratify = filtered_codenet_df['problem_num']
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
tokenizer.truncation_side = 'left'

codes = train_df['code'].to_list()
problems = train_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25L(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = train_df[train_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(), 2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1]  # 내림차순
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

pos_code1 = list(map(lambda x: x[0], total_positive_pairs))
pos_code2 = list(map(lambda x: x[1], total_positive_pairs))

neg_code1 = list(map(lambda x: x[0], total_negative_pairs))
neg_code2 = list(map(lambda x: x[1], total_negative_pairs))

pos_label = [1] * len(pos_code1)
neg_label = [0] * len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={'code1': total_code1, 'code2': total_code2, 'similar': total_label})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)

pair_data.to_csv('data/new_dataset_0607/graph_codenet_train_bm25L.csv', index=False)

codes = valid_df['code'].to_list()
problems = valid_df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25L(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = valid_df[valid_df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(), 2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1]  # 내림차순
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

pos_code1 = list(map(lambda x: x[0], total_positive_pairs))
pos_code2 = list(map(lambda x: x[1], total_positive_pairs))

neg_code1 = list(map(lambda x: x[0], total_negative_pairs))
neg_code2 = list(map(lambda x: x[1], total_negative_pairs))

pos_label = [1] * len(pos_code1)
neg_label = [0] * len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={'code1': total_code1, 'code2': total_code2, 'similar': total_label})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)

pair_data.to_csv('data/new_dataset_0607/graph_codenet_valid_bm25L.csv', index=False)