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


# load test_dataset



# load model
# checkpoint = 'mrm8488/CodeBERTaPy'
# checkpoint = 'microsoft/graphcodebert-base'
checkpoint = 'microsoft/codebert-base-mlm'

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
# PATH = './data/CodeBERTaPy/1_codebertapy_0531.pt'
# PATH = './data/graphcodebert/1_graphcodebert-base_0531.pt'
PATH = './data/codebert-mlm/2_codebert-base-mlm_0531.pt'

model.load_state_dict(torch.load(PATH))
model.cuda()

# test_input_ids = torch.load("./data/CodeBERTaPy/test_input_ids.pt")
# test_attention_masks = torch.load("./data/CodeBERTaPy/test_attention_masks.pt")

# test_input_ids = torch.load("./data/graphcodebert/test_input_ids.pt")
# test_attention_masks = torch.load("./data/graphcodebert/test_attention_masks.pt")

test_input_ids = torch.load("./data/codebert-mlm/test_input_ids.pt")
test_attention_masks = torch.load("./data/codebert-mlm/test_attention_masks.pt")


batch_size = 512

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
submission.to_csv('./data/submission_codebert-mlm_0602.csv', index=False)

