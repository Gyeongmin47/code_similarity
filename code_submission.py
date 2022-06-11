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


# 민석님이 해주실 부분
def data_preprocess():
    pass

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