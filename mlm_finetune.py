from setproctitle import setproctitle
from transformers import *
from accelerate import Accelerator

from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import math
import collections, os, re
import pandas as pd
import numpy as np



setproctitle("Gyeongmin")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dir_path = 'graphcodebert/'     # CodeBERTaPy/, graphcodebert/ /codebert-mlm


# read saved train and validation data

# 경로 추후 수정 예정
# train_data = pd.read_csv("./data/" + dir_path + "train_data.csv")
# valid_data = pd.read_csv("./data/" + dir_path + "valid_data.csv")

lines = []
train_data = pd.read_csv("./data/sample_train.csv")

for _, row in train_data.iterrows():
    rows = row['code1'] + row['code2']
    lines.append(rows)




c1 = train_data['code1'].values
c2 = train_data['code2'].values
similar = train_data['similar']


N = train_data.shape[0]
MAX_LEN = 512

input_ids = np.zeros((N, MAX_LEN),dtype=int)
attention_masks = np.zeros((N, MAX_LEN),dtype=int)
labels = np.zeros((N),dtype=int)


model_checkpoint = "microsoft/graphcodebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# # model = AutoModelForSequenceClassification.from_pretrained("microsoft/graphcodebert-base")
# # checkpoint = torch.load("data/CodeBERTaPy/codebertapy.pt")
# # sim_model.load_state_dict(checkpoint)
# # model.cuda()

for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_c1 = c1[i]
        cur_c2 = c2[i]
        encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids[i,] = encoded_input['input_ids']
        attention_masks[i,] = encoded_input['attention_mask']
        if tokenizer.is_fast:
            encoded_input["word_ids"] = [encoded_input.word_ids(i) for i in range(len(encoded_input["input_ids"]))]
        # labels[i] = similar[i]

    except Exception as e:
        print(e)
        pass




# c1 = valid_data['code1'].values
# c2 = valid_data['code2'].values
# similar = valid_data['similar']

# N = valid_data.shape[0]
# MAX_LEN = 512

# valid_input_ids = np.zeros((N, MAX_LEN),dtype=int)
# valid_attention_masks = np.zeros((N, MAX_LEN),dtype=int)
# valid_labels = np.zeros((N),dtype=int)

# for i in tqdm(range(N), position=0, leave=True):
#     try:
#         cur_c1 = c1[i]
#         cur_c2 = c2[i]
#         encoded_input = tokenizer(cur_c1, cur_c2, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
#         valid_input_ids[i,] = encoded_input['input_ids']
#         valid_attention_masks[i,] = encoded_input['attention_mask']
#         valid_labels[i] = similar[i]
#     except Exception as e:
#         print(e)
#         pass


input_ids = torch.tensor(input_ids, dtype=int)
attention_masks = torch.tensor(attention_masks, dtype=int)
# labels = torch.tensor(labels, dtype=int)

# valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
# valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
# valid_labels = torch.tensor(valid_labels, dtype=int)

# torch.save(input_ids, "./data/" + dir_path + 'train_input_ids_0529.pt')
# torch.save(attention_masks, "./data/" + dir_path + 'train_attention_masks_0529.pt')
# torch.save(labels, "./data/" + dir_path + "train_labels_0529.pt")

# torch.save(valid_input_ids, "./data/" + dir_path + "valid_input_ids_0529.pt")
# torch.save(valid_attention_masks, "./data/" + dir_path + "valid_attention_masks_0529.pt")
# torch.save(valid_labels, "./data/" + dir_path + "valid_labels_0529.pt")


# input_ids = torch.load("./data/" + dir_path + 'train_input_ids.pt')
# attention_masks = torch.load("./data/" + dir_path + 'train_attention_masks.pt')
# labels = torch.load("./data/" + dir_path + 'train_labels.pt')

# valid_input_ids = torch.load("./data/" + dir_path + 'valid_input_ids.pt')
# valid_attention_masks = torch.load("./data/" + dir_path + 'valid_attention_masks.pt')
# valid_labels = torch.load("./data/" + dir_path + 'valid_labels.pt')


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)  # 나중에 이 확률은 좀 수정해야할 수도 있긴 하겠다.

# samples = [input_ids[i] for i in range(2)]

# for sample in samples:
#     print(sample)

# for chunk in data_collator(samples)["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")

# exit()



batch_size = 64
train_data = TensorDataset(input_ids, attention_masks, labels)
train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=data_collator)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, collate_fn=data_collator)



# validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
# validation_sampler = SequentialSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, collate_fn=data_collator)



# for chunk in data_collator(input_ids):
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        print("feature:", feature)
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)


samples_input_ids = [input_ids[i] for i in range(2)]
samples_attention_masks = [attention_masks[i] for i in range(2)]
samples_labels = [labels[i] for i in range(2)]

samples = [{'input_ids': samples_input_ids, 'attention_masks': samples_attention_masks, 'labels': samples_labels}]

for sample in samples:
    _ = sample.pop("word_ids")

batch = whole_word_masking_data_collator(samples)

for chunk in batch:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

exit()

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model.cuda()


optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, validation_dataloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch


total_steps = len(train_dataloader) * num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

device = torch.device("cuda")


progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):

    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(validation_data)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./data/" + dir_path + "checkpoint_0529_" + str(epoch), save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained("./data/" + dir_path + "checkpoint_0529_" + str(epoch))
        # repo.push_to_hub(
        #     commit_message=f"Training in progress epoch {epoch}", blocking=False
        # )