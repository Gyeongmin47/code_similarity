import pandas as pd
import numpy as np

from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from setproctitle import setproctitle

import argparse
import torch


setproctitle("Gyeongmin")

print("")
print("Validating...")

class INIT_TEST():
    def __init__(self, args):

        # dir_path = CodeBERTaPy, graphcodebert, codebert-mlm
        self.pretrained_model_name_and_path = {'CodeBERTaPy': 'mrm8488/CodeBERTaPy',
                                               'graphcodebert': 'microsoft/graphcodebert-base',
                                               'codebert-mlm': 'microsoft/codebert-base-mlm'}

        test_data = pd.read_csv("./data/test.csv")

        if args.save_feature:
            c1, c2 = self.load_data(test_data)
            for dir_path, model_name in self.pretrained_model_name_and_path.items():
                self.save_features(c1, c2, test_data, dir_path, model_name)
                self.model_CodeBERTaPy, self.model_graphcode, self.model_codebert_mlm = self.load_model(dir_path, model_name)
        else:
            # load saved tensors
            for dir_path, model_name in self.pretrained_model_name_and_path.items():
                self.load_features(dir_path, model_name)
                self.model_CodeBERTaPy, self.model_graphcode, self.model_codebert_mlm = self.load_model(dir_path, model_name)


        batch_size = 512

        test_tensor = TensorDataset(self.CodeBERTaPy_test_input_ids, self.CodeBERTaPy_test_attention_masks)
        test_sampler = SequentialSampler(test_tensor)
        self.CodeBERTaPy_test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)

        test_tensor = TensorDataset(self.graphcodebert_test_input_ids, self.graphcodebert_test_attention_masks)
        test_sampler = SequentialSampler(test_tensor)
        self.graphcodebert_test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)

        test_tensor = TensorDataset(self.codebert_mlm_test_input_ids, self.codebert_mlm_test_attention_masks)
        test_sampler = SequentialSampler(test_tensor)
        self.codebert_mlm_test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)


    def load_model(self, dir_path, model_name):

        print("==========================================================")
        print("================   load model: {}   ======================".format(model_name))
        print("==========================================================")

        if model_name == 'mrm8488/CodeBERTaPy':
            self.model_CodeBERTaPy = AutoModelForSequenceClassification.from_pretrained(model_name)
            PATH = './data/' + dir_path + '/' + 'codebertapy_2epoch_0524.pt'
            self.model_CodeBERTaPy.load_state_dict(torch.load(PATH))
            self.model_CodeBERTaPy.cuda()

        elif model_name == 'microsoft/graphcodebert-base':
            self.model_graphcode = AutoModelForSequenceClassification.from_pretrained(model_name)
            PATH = './data/' + dir_path + '/' + 'graphcodebert-base_1_0524.pt'
            self.model_graphcode.load_state_dict(torch.load(PATH))
            self.model_graphcode.cuda()

        elif model_name == 'microsoft/codebert-base-mlm':
            self.model_codebert_mlm = AutoModelForSequenceClassification.from_pretrained(model_name)
            PATH = './data/' + dir_path + '/' + 'codebert-base-mlm_1_0524.pt'
            self.model_codebert_mlm.load_state_dict(torch.load(PATH))
            self.model_codebert_mlm.cuda()

        return self.model_CodeBERTaPy, self.model_graphcode, self.model_codebert_mlm


    def load_data(self, data):
        c1 = data['code1'].values
        c2 = data['code2'].values

        return c1, c2

    def load_features(self, dir_path, model_name):
        print("=============================================================")
        print("=================   load features: {}   =====================".format(model_name))
        print("=============================================================")

        if model_name == 'mrm8488/CodeBERTaPy':
            self.CodeBERTaPy_test_input_ids = torch.load("./data/" + dir_path + '/' + "test_input_ids.pt")
            self.CodeBERTaPy_test_attention_masks = torch.load("./data/" + dir_path + '/' + "test_attention_masks.pt")

        elif model_name == 'microsoft/graphcodebert-base':
            self.graphcodebert_test_input_ids = torch.load("./data/" + dir_path + '/' + "test_input_ids.pt")
            self.graphcodebert_test_attention_masks = torch.load("./data/" + dir_path + '/' + "test_attention_masks.pt")

        elif model_name == 'microsoft/codebert-base-mlm':
            self.codebert_mlm_test_input_ids = torch.load("./data/" + dir_path + '/' + "test_input_ids.pt")
            self.codebert_mlm_test_attention_masks = torch.load("./data/" + dir_path + '/' + "test_attention_masks.pt")


    def save_features(self, c1, c2, test_data, dir_path, model_name):

        N = test_data.shape[0]
        MAX_LEN = 512

        self.test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
        self.test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i in tqdm(range(N), position=0, leave=True):
            try:
                cur_c1 = c1[i]
                cur_c2 = c2[i]
                encoded_input = self.tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512,
                                                 padding='max_length', truncation=True)
                self.test_input_ids[i,] = encoded_input['input_ids']
                self.test_attention_masks[i,] = encoded_input['attention_mask']

            except Exception as e:
                print(e)
                pass

        test_input_ids = torch.tensor(self.test_input_ids, dtype=int)
        test_attention_masks = torch.tensor(self.test_attention_masks, dtype=int)

        torch.save(test_input_ids, "./data/" + dir_path + '/' + "test_input_ids.pt")
        torch.save(test_attention_masks, "./data/" + dir_path + '/' + "test_attention_masks.pt")


    def do_test(self):

        submission = pd.read_csv('./data/sample_submission.csv')

        device = torch.device("cuda")

        preds = np.array([])

        global preds_1
        global preds_2
        global preds_3

        # for CodeBERTaPy
        for step, batch in tqdm(enumerate(self.CodeBERTaPy_test_dataloader), desc="Iteration", smoothing=0.05):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model_CodeBERTaPy(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu()

            _pred = logits.numpy()
            pred = np.argmax(_pred, axis=1).flatten()

            preds_1 = np.append(preds, pred)

        preds = np.array([])
        # for CodeBERTaPy
        for step, batch in tqdm(enumerate(self.graphcodebert_test_dataloader), desc="Iteration", smoothing=0.05):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model_CodeBERTaPy(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu()

            _pred = logits.numpy()
            pred = np.argmax(_pred, axis=1).flatten()

            preds_2 = np.append(preds, pred)

        preds = np.array([])
        # for CodeBERTaPy
        for step, batch in tqdm(enumerate(self.codebert_mlm_test_dataloader), desc="Iteration", smoothing=0.05):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model_CodeBERTaPy(b_input_ids, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu()

            _pred = logits.numpy()
            pred = np.argmax(_pred, axis=1).flatten()

            preds_3 = np.append(preds, pred)

        preds = (preds_1 + preds_2 + preds_3) / 3
        # preds = np.where(preds > 0.5, 1, 0)
        submission['similar'] = preds
        submission.to_csv('./data/submission_ensemble.csv', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Set arguments.")
    # parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, choices=['mrm8488/CodeBERTaPy',
    #                                                                                         'microsoft/graphcodebert-base',
    #                                                                                         'microsoft/codebert-base-mlm'])

    parser.add_argument('--save_feature', type=bool, default=False)

    args = parser.parse_args()

    init_test = INIT_TEST(args)
    init_test.do_test()