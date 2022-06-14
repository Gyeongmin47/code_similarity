from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from setproctitle import setproctitle
from sklearn.model_selection import train_test_split
from itertools import combinations
from rank_bm25 import BM25L

import torch
import torch.nn as nn
import random
import time
import datetime
import numpy as np
import pandas as pd
import os, re
import argparse



''' 아무 내용이 없는 줄은 버린다 '''
def get_rid_of_empty(c):
    ret = []
    splitted = c.split('\n')
    for s in splitted:
        if len(s.strip()) > 0:
            ret.append(s)
    return '\n'.join(ret)


''' 데이터 클리닝 함수 '''
def clean_data(script, data_type="dir"):
    if data_type == "dir":
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
    elif data_type == "file":
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

    ''' 극소수지만 데이터 몇개는 완성되지 않은 주석들이 있었습니다 '''
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


''' positive, negative 페어 생성 함수 '''
def get_pairs(input_df, tokenizer):
    codes = input_df['code'].to_list()
    problems = input_df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
    bm25 = BM25L(tokenized_corpus)

    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(problems):
        solution_codes = input_df[input_df['problem_num'] == problem]['code']
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
                    negative_solutions.append(input_df['code'].iloc[high_score_idx])
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
    return pair_data


""" 전체 데이터 전처리 함수 """
def data_preprocess(args):
    """ Data preprocess """
    # train, valid, test 에 대한 전처리 추가해주시면 감사하겠습니다 :)

    # 아래에서 호출은 다음과 같이 진행합니다.

    """
    dacon_train_data = pd.read_csv("./data/" + "new_dataset_0607/graph_dacon_train_bm25L.csv")
    dacon_valid_data = pd.read_csv("./data/" + "new_dataset_0607/graph_dacon_valid_bm25L.csv")
    codenet_train_data = pd.read_csv("./data/" + "new_dataset_0607/graph_codenet_train_bm25L.csv")
    codenet_valid_data = pd.read_csv("./data/" + "new_dataset_0607/graph_codenet_valid_bm25L.csv")
    test_data = pd.read_csv("./data/new_dataset_0604/processed_test.csv")
    """

    # 데이콘이 제공해준 학습 코드 데이터 데이터프레임 만들기
    # 베스이    code_folder = "code"  # 데이콘이 제공해준 학습 데이터 파일의 경로
    problem_folders = os.listdir(code_folder)
    preproc_scripts = []
    problem_nums = []
    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(code_folder, problem_folder))
        problem_num = scripts[0].split('_')[0]
        for script in scripts:
            script_file = os.path.join(code_folder, problem_folder, script)
            preprocessed_script = clean_data(script_file, data_type="dir")
            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num] * len(scripts))
    train_df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums})

    # 데이콘이 제공해준 테스트 코드 데이터 데이터프레임 만들기
    test_df = pd.read_csv("test.csv")
    code1 = test_df['code1'].values
    code2 = test_df['code2'].values
    processed_code1 = []
    processed_code2 = []
    for i in tqdm(range(len(code1))):
        processed_c1 = clean_data(code1[i], data_type="file")
        processed_c2 = clean_data(code2[i], data_type="file")
        processed_code1.append(processed_c1)
        processed_code2.append(processed_c2)
    processed_test = pd.DataFrame(list(zip(processed_code1, processed_code2)),
                                  columns=["code1", "code2"])

    # IBM의 CodeNet으로 추가 코드 학습/검증 데이터 데이터프레임 만들기
    code_folder = "Project_CodeNet_Python800"  # CodeNet 데이터 경로
    problem_folders = os.listdir(code_folder)
    preproc_scripts = []
    problem_nums = []
    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(code_folder, problem_folder))
        problem_num = int(problem_folder.split('p')[1])
        problem_num = 'problem' + str(problem_num)
        for script in scripts:
            script_file = os.path.join(code_folder, problem_folder, script)
            preprocessed_script = clean_data(script_file)
            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num] * len(scripts))
    codenet_df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums})

    # 추가 codenet 데이터에서 테스트셋과 겹치는 데이터가 있다는걸 관찰했다.
    # 1차 필터링을 진행한다.
    # codenet_df에서 test_df의 데이터와 겹치는 녀석들을 필터링해준다.
    # 1차 필터링은 단순 set (hash table)을 이용해서 거의 다 필터링해준다. 매우 꼼꼼하게 진행하기 위해 총 3단계 필터링을 거쳤다.
    dacon_codes = np.concatenate([train_df['code'].values,
                                  test_df['code1'].values, test_df['code2'].values])
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
    filtered_codenet_df = pd.DataFrame(data={'code': usable_codes,
                                             'problem_num': usable_problem_nums})

    # 데이터 사이즈가 매우 커서 이렇게 완성된 filtered_codenet_df에서 50%의 데이터만 이용해서 학습에 사용한다.
    filtered_codenet_df = filtered_codenet_df.sample(frac=0.5, random_state=42)

    # 2차 필터링을 진행해준다.
    # 1차때 trailing space 등의 이유로 set 방법으로 완전하게 걸러지지 않은 녀석들을 걸러주는게 목적이다.
    # 이걸 위해서 코드 문자열에 존재하는 newline들을 전부 이어붙히고 앞뒤로 존재하는 공백과 newline들을 없앤다.
    # 이후 이렇게 전처리된 test code와 codenet code 문자열들을 각각 set에 넣고 intersection을 통해 겹치는걸 찾는다.
    def simplify(x):
        return ''.join(x.split('\n')).rstrip(' ').strip()

    codenet_codes = filtered_codenet_df['code'].values
    codenet_problem_nums = filtered_codenet_df['problem_num'].values
    test_codes1 = test['code1'].values
    test_codes2 = test['code2'].values
    test_codes = np.concatenate([test_codes1, test_codes2])
    codenet_set = set()
    for i in tqdm(range(len(codenet_codes))):
        codenet_set.add(simplify(codenet_codes[i]))
    test_set = set()
    for i in tqdm(range(len(test_codes))):
        test_set.add(simplify(test_codes[i]))
    intersection = codenet_set.intersection(test_set)
    usable_codenet_filterd, usable_codenet_filtered_problems = [], []
    for i in tqdm(range(len(codenet_codes))):
        if simplify(codenet_codes[i]) not in intersection:
            usable_codenet_filtered.append(codenet_codes[i])
            usable_codenet_filtered_problems.append(codenet_problem_nums[i])
    filtered_codenet_df = pd.DataFrame(data={'code': usable_codenet_filtered,
                                             'problem_num': usable_codenet_filterd_problems})

    # 3차 필터링을 진행해준다.
    # 2차 필터링 이후로 test set에 포함된 데이터가 추가 데이터에서 거의 다 제거되었겠지만
    # 확실하게 전부 제거해주기 위해 exhaustive search를 최종적으로 남아있는 test셋의 흔적들을 없애버립니다.
    codenet_codes = filtered_codenet_df['code'].values
    problem_nums = filtered_codenet_df['problem_num'].values
    usable_codenet_filtered, usable_codenet_filtered_problems = [], []
    for i in tqdm(range(len(codenet_codes)), position=0, leave=True):
        usable = True
        if codenet_codes[i] in test_set:
            continue
        else:
            for s in test_set:
                if len(s) > 0 and len(codenet_codes[i]) > 0 and ((codenet_codes[i] in s) or (s in codenet_codes[i])):
                    usable = False
                    break
        if usable == True:
            usable_codenet_filterd.append(codenet_codes[i])
            usable_codenet_filtered_problems.append(problem_nums[i])

    filtered_codenet_df = pd.DataFrame(data={'code': usable_codenet_filtered,
                                             'problem_num': usable_codenet_filterd_problems})

    # 데이터 프레임을 만들었으니 이제 train/val split을 진행한다.
    # 이후에 positive, negative pairs를 생성한다.
    # 청소님의 코드를 참고해서 hard negative pair를 생성하는데 BM25대신 BM25L을 사용한다.
    # tokenizer는 왼쪽부터 truncation을 진행하게해서 truncation이 필요할때는 코드의 끝 부분들을 이용하게 만든다.
    dacon_train_df, dacon_valid_df, dacon_train_label, dacon_valid_label = train_test_split(
        train_df,
        train_df['problem_num'],
        random_state=args.seed,
        test_size=0.1,
        stratify=full_df['problem_num']
    )
    dacon_train_df = dacon_train_df.reset_index(drop=True)
    dacon_valid_df = dacon_valid_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.truncation_side = 'left'

    dacon_train_bm25L = get_pairs(dacon_train_df, tokenizer)
    dacon_valid_bm25L = get_pairs(dacon_valid_df, tokenizer)

    # 생성된 데이터 저장
    dacon_train_bm25L.to_csv("./data/" + "new_dataset_0607/graph_dacon_train_bm25L.csv", index=False)
    dacon_valid_bm25L.to_csv("./data/" + "new_dataset_0607/graph_dacon_valid_bm25L.csv", index=False)
    processed_test.to_csv("./data/new_dataset_0604/processed_test.csv", index=False)

    codenet_train_df, codenet_valid_df, codenet_train_label, codenet_valid_label = train_test_split(
        filtered_codenet_df,
        filtered_codenet_df['problem_num'],
        random_state=args.seed,
        test_size=0.1,
        stratify=full_df['problem_num']
    )
    codenet_train_df = codenet_train_df.reset_index(drop=True)
    codenet_valid_df = codenet_valid_df.reset_index(drop=True)

    codenet_train_bm25L = get_pairs(codenet_train_df, tokenizer)
    codenet_valid_bm25L = get_pairs(codenet_valid_df, tokenizer)
    # 생성된 데이터 저장
    codenet_train_bm25L.to_csv("./data/" + "new_dataset_0607/graph_codenet_train_bm25L.csv",
                               index=False)
    codenet_valid_bm25L.to_csv("./data/" + "new_dataset_0607/graph_codenet_valid_bm25L.csv",
                               index=False)


def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def train_model(args):

    set_seed(args)
    setproctitle(args.process_name)
    dacon_train_data = pd.read_csv("./data/" + "new_dataset_0607/graph_dacon_train_bm25L.csv")
    dacon_valid_data = pd.read_csv("./data/" + "new_dataset_0607/graph_dacon_valid_bm25L.csv")

    codenet_train_data = pd.read_csv("./data/" + "new_dataset_0607/graph_codenet_train_bm25L.csv")
    codenet_valid_data = pd.read_csv("./data/" + "new_dataset_0607/graph_codenet_valid_bm25L.csv")

    train_data = pd.concat([dacon_train_data, codenet_train_data], axis=0)
    valid_data = pd.concat([dacon_valid_data, codenet_valid_data], axis=0)

    # training
    c1 = train_data['code1'].values
    c2 = train_data['code2'].values
    similar = train_data['similar'].values

    N = train_data.shape[0]
    MAX_LEN = 512

    input_ids = np.zeros((N, MAX_LEN), dtype=int)
    attention_masks = np.zeros((N, MAX_LEN), dtype=int)
    labels = np.zeros((N), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            input_ids[i,] = encoded_input['input_ids']
            attention_masks[i,] = encoded_input['attention_mask']
            labels[i] = similar[i]
        except Exception as e:
            print(e)
            pass


    # validating
    c1 = valid_data['code1'].values
    c2 = valid_data['code2'].values
    similar = valid_data['similar'].values

    N = valid_data.shape[0]

    MAX_LEN = 512

    valid_input_ids = np.zeros((N, MAX_LEN), dtype=int)
    valid_attention_masks = np.zeros((N, MAX_LEN), dtype=int)
    valid_labels = np.zeros((N), dtype=int)

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            valid_input_ids[i,] = encoded_input['input_ids']
            valid_attention_masks[i,] = encoded_input['attention_mask']
            valid_labels[i] = similar[i]
        except Exception as e:
            print(e)
            pass

    if os.path.exists(args.dir_path):
        os.makedirs(args.dir_path, exist_ok=True)


    print("\n\nMake tensor\n\n")
    input_ids = torch.tensor(input_ids, dtype=int)
    attention_masks = torch.tensor(attention_masks, dtype=int)
    labels = torch.tensor(labels, dtype=int)

    valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
    valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
    valid_labels = torch.tensor(valid_labels, dtype=int)

    if args.save_tensor == True:
        torch.save(input_ids, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_input_ids_BM25L_0608.pt')
        torch.save(attention_masks, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_attention_masks_BM25L_0608.pt')
        torch.save(labels, "./data/" + args.dir_path + "/" + args.model_name + '_mixed_train_labels_BM25L_0608.pt')

        torch.save(valid_input_ids, "./data/" + args.dir_path + "/" + args.model_name + "_mixed_valid_input_ids_BM25L_0608.pt")
        torch.save(valid_attention_masks, "./data/" + args.dir_path + "/" + args.model_name + "mixed_valid_attention_masks_BM25L_0608.pt")
        torch.save(valid_labels, "./data/" + args.dir_path + "/" + args.model_name + "mixed_valid_labels_BM25L_0608.pt")


    # Setup training
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    train_data = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5)  # 아직 이게 정확하지 않음

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda")
    loss_f = nn.CrossEntropyLoss()

    # Train
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    model.zero_grad()
    for i in range(args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(i + 1, args.epochs))
        print('Training...')
        t0 = time.time()
        train_loss, train_accuracy = 0, 0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", smoothing=0.05):
            if step % 10000 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                print('  current average loss = {}'.format(
                    train_loss / step))  # bot.sendMessage(chat_id=chat_id, text = '  current average loss = {}'.format(train_loss / step))

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
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
                outputs = model(b_input_ids, attention_mask=b_input_mask)

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

        # if np.min(val_losses) == val_losses[-1]:
        print("saving current best checkpoint")
        torch.save(model.state_dict(), "./data/" + args.dir_path + "/" + str(i + 1) + "_mixed_" + args.model_name + "_BM25L_0608.pt")


def inference_model(args):
    test_data = pd.read_csv("./data/new_dataset_0604/processed_test.csv")

    c1 = test_data['code1'].values
    c2 = test_data['code2'].values

    N = test_data.shape[0]
    MAX_LEN = 512

    test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
    test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.truncation_side = "left"

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_c1 = str(c1[i])
            cur_c2 = str(c2[i])
            encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
                                      truncation=True)
            test_input_ids[i,] = encoded_input['input_ids']
            test_attention_masks[i,] = encoded_input['attention_mask']

        except Exception as e:
            print(e)
            pass

    test_input_ids = torch.tensor(test_input_ids, dtype=int)
    test_attention_masks = torch.tensor(test_attention_masks, dtype=int)

    if args.save_tensor == True:
        torch.save(test_input_ids, "./data/" + args.dir_path + "/" + "test_input_ids_0605.pt")
        torch.save(test_attention_masks, "./data/" + args.dir_path + "/" + "test_attention_masks_0605.pt")

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    PATH = "./data/" + args.dir_path + "/" + "1_mixed_" + args.model_name + "_BM25L_0608.pt"

    model.load_state_dict(torch.load(PATH))
    model.cuda()

    test_tensor = TensorDataset(test_input_ids, test_attention_masks)
    test_sampler = SequentialSampler(test_tensor)
    test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=args.test_batch_size)

    submission = pd.read_csv('./data/sample_submission.csv')
    device = torch.device("cuda")

    preds = np.array([])
    for step, batch in tqdm(enumerate(test_dataloader), desc="Iteration", smoothing=0.05):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu()
        _pred = logits.numpy()
        pred = np.argmax(_pred, axis=1).flatten()
        preds = np.append(preds, pred)

    submission['similar'] = preds
    submission.to_csv('./data/submission_' + args.model_name + '_0610.csv', index=False)


def model_ensemble():
    submission = pd.read_csv('./data/sample_submission.csv')

    submission_1 = pd.read_csv('./data/submission_graphcodebert_BM25L_0610.csv')
    submission_2 = pd.read_csv('./data/submission_CodeBERTaPy_BM25L_0610.csv')
    submission_3 = pd.read_csv('./data/submission_codebert_mlm_BM25L_0610.csv')

    sub_1 = submission_1['similar']
    sub_2 = submission_2['similar']
    sub_3 = submission_3['similar']

    ensemble_preds = (sub_1 + sub_2 + sub_3) / 3

    preds = np.where(ensemble_preds > 0.5, 1, 0)

    submission['similar'] = preds

    submission.to_csv('./data/submission_ensemble_0610_v2.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set arguments.")

    parser.add_argument("--seed", default="42", type=int, help="Random seed for initialization")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-5, type=float, help="The initial eps.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=None, help="batch_size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="test_batch_size")

    parser.add_argument("--no_cuda", default=False, type=bool, help="Say True if you don't want to use cuda.")
    parser.add_argument("--ensemble", default=False, type=bool, help="Ensemble.")
    parser.add_argument("--save_tensor", default=True, type=str, help="Save tensor.")
    parser.add_argument("--mode", default="train", type=str, help="When you train the model.")
    parser.add_argument("--dir_path", default="graphcodebert", type=str, help="Save model path.")
    parser.add_argument("--model_name", default="graphcodebert", type=str, help="Model name.")
    parser.add_argument("--process_name", default="code_similarity", type=str, help="process_name.")
    parser.add_argument("--checkpoint_path", default="microsoft/graphcodebert-base", type=str, help="Pre-trained Language Model.")

    args = parser.parse_args()

    if args.mode == "train":
        data_preprocess()
        train_model(args)
    else:
        inference_model(args)

    if args.ensemble == True:
        model_ensemble()

    # CUDA_VISIBLE_DEVICES=0 python code_submission.py --seed 42 --learning_rate 2e-5 --eps 1e-5 --epochs 3 --batch_size 32 --test_batch_size 1048 --save_tensor True --mode train --dir_path graphcodebert --model_name graphcodebert --process_name code_similarity --checkpoint_path microsoft/graphcodebert-base