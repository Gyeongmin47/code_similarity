from setproctitle import setproctitle
from transformers import *
from dataclasses import dataclass
from pathlib import Path
import torch
import re, os, random
import pandas as pd
from torch.utils.data import Dataset

# utils.py: https://gist.github.com/Kitsunetic/833143d9cc89325c7e95bf3d3a0d4fcf
from utils import make_result_dir, seed_everything

setproctitle("Gyeongmin")

# model.cuda()

# text = "This is a great <mask>."    # tokenizer에서 사용하는 mask 모양: <mask>, [MASK] 아님
#
# inputs = tokenizer(text, return_tensors="pt")
#
# token_logits = model(**inputs).logits
#
# print("\n\ntokenizer:", tokenizer)
# # print("tokenizer.mask_token:", tokenizer.mask_token)
# print("\n\n\n======================================\n\n\n")
#
# # torch.where == (tensor([0]), tensor([5]))
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
#
# mask_token_logits = token_logits[0, mask_token_index, :]
#
# # 가장 큰 logits값을 가지는 [MASK] 후보를 선택
# top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
#
# for token in top_5_tokens:
#     print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

@dataclass
class Config:
    # exp_num: str = re.search(r"exp(\d{3}(_\d+)?)", Path(__file__).name).group(1)

    debug: bool = False
    seed: int = 8888
    result_dir_root = Path("checkpoints")
    result_dir: Path = None

    nlp_model_name: str = "microsoft/graphcodebert-base"

    if debug:
        epochs: int = 1
    else:
        epochs: int = 3
    save_step: int = 30000

    dataset_dir: str = "data"
    batch_size: int = 48
    mlm_probability: float = 0.15



class TokenizerDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config, debug=False) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        # df = pd.read_csv("./data/sample_train.csv")
        # self.lines = []
        # for _, row in df.iterrows():
        #     line = preprocessing(row.title)
        #     self.lines.append(line)

        print("Info: Loading start")

        self.lines = []
        # train_data = pd.read_csv("./data/sample_train.csv")
        train_data = pd.read_csv("./data/" + "new_dataset/dacon_code_train_data.csv")
        valid_data = pd.read_csv("./data/" + "new_dataset/dacon_code_valid_data.csv")

        for _, row in train_data.iterrows():
            rows = row['code1'] + row['code2']
            self.lines.append(rows)

        for _, row in valid_data.iterrows():
            rows = row['code1'] + row['code2']
            self.lines.append(rows)

        if debug:
            self.lines = random.sample(self.lines, 1000)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int):
        line = self.lines[idx]
        token = self.tokenizer.encode_plus(line, max_length=self.tokenizer.model_max_length, truncation=True)
        return token

def main(config:Config):
    seed_everything(config.seed)

    model_checkpoint = "microsoft/graphcodebert-base"

    config.nlp_model_name = model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=config.mlm_probability)
    ds_train = TokenizerDataset(tokenizer, config)

    training_args = TrainingArguments(output_dir=config.result_dir, overwrite_output_dir=True,
        num_train_epochs=config.epochs, per_device_train_batch_size=config.batch_size, save_steps=config.save_step,
        save_total_limit=1, logging_dir=config.result_dir / "log", )
    trainer = Trainer(model=model, args=training_args, data_collator=collator, train_dataset=ds_train,
        # eval_dataset=ds_val,
        # prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(config.result_dir)

    final_dir = config.result_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

config = Config(debug=False)
config.epochs = 2
config.result_dir = make_result_dir(config)
main(config)