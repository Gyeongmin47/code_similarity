import pandas as pd
import numpy as np

from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from setproctitle import setproctitle

import argparse
import torch

setproctitle("Gyeongmin")

print("\nValidating...")

# load model
# checkpoint = 'mrm8488/CodeBERTaPy'
checkpoint = 'microsoft/graphcodebert-base'
# checkpoint = 'microsoft/codebert-base-mlm'


# test_data = pd.read_csv("./data/new_dataset_0604/processed_test.csv")
#
# c1 = test_data['code1'].values
# c2 = test_data['code2'].values
#
# N = test_data.shape[0]
# MAX_LEN = 512
#
# test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
# test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)
#
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.truncation_side = "left"
#
# for i in tqdm(range(N), position=0, leave=True):
#     try:
#         cur_c1 = str(c1[i])
#         cur_c2 = str(c2[i])
#         encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length',
#                                        truncation=True)
#         test_input_ids[i,] = encoded_input['input_ids']
#         test_attention_masks[i,] = encoded_input['attention_mask']
#
#     except Exception as e:
#         print(e)
#         pass
#
#
# test_input_ids = torch.tensor(test_input_ids, dtype=int)
# test_attention_masks = torch.tensor(test_attention_masks, dtype=int)

# torch.save(test_input_ids, "./data/CodeBERTaPy/test_input_ids_0605.pt")
# torch.save(test_attention_masks, "./data/CodeBERTaPy/test_attention_masks_0605.pt")

# torch.save(test_input_ids, "./data/graphcodebert/test_input_ids_0605.pt")
# torch.save(test_attention_masks, "./data/graphcodebert/test_attention_masks_0605.pt")

# torch.save(test_input_ids, "./data/codebert-mlm/test_input_ids_0605.pt")
# torch.save(test_attention_masks, "./data/codebert-mlm/test_attention_masks_0605.pt")


# load test_dataset

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# PATH = './data/CodeBERTaPy/2_codebertapy_0604.pt'
PATH = './data/graphcodebert/1_graphcodebert-base_BM25L_0605.pt'
# PATH = './data/codebert-mlm/1_codebert-base-mlm_BM25L_0605.pt'


model.load_state_dict(torch.load(PATH))
model.cuda()

# test_input_ids = torch.load("./data/CodeBERTaPy/test_input_ids.pt")
# test_attention_masks = torch.load("./data/CodeBERTaPy/test_attention_masks.pt")

test_input_ids = torch.load("./data/graphcodebert/test_input_ids_0605.pt")
test_attention_masks = torch.load("./data/graphcodebert/test_attention_masks_0605.pt")

# test_input_ids = torch.load("./data/codebert-mlm/test_input_ids_0605.pt")
# test_attention_masks = torch.load("./data/codebert-mlm/test_attention_masks_0605.pt")

batch_size = 2048

test_tensor = TensorDataset(test_input_ids, test_attention_masks)
test_sampler = SequentialSampler(test_tensor)
test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=batch_size)

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
# submission.to_csv('./data/submission_CodeBERTaPy_0605.csv', index=False)
submission.to_csv('./data/submission_graphcodebert_BM25L_0607.csv', index=False)
# submission.to_csv('./data/submission_codebert_mlm_BM25L_0607.csv', index=False)


