import argparse
import os
import argparse

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraConfig
)
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    InputExample
from tqdm import tqdm
from pytorchtools import EarlyStopping


# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='P-tuning v2')
parser.add_argument('--plm', type=str, default="bert", help='choose plm: bert or roberta or ernie')
parser.add_argument('--type', type=str, default="ptuningv2", help='choose model: ptuningv1, ptuningv2, lora')
parser.add_argument('--soft_tokens', type=int, default=10, help='define the soft tokens')
parser.add_argument('--patience', type=int, default=8, help='define the soft tokens')
parser.add_argument('--num_epochs', type=int, default=15, help='define the soft tokens')
args = parser.parse_args()

if args.plm == 'bert':
    model_name_or_path = "bert-base-chinese"
elif args.plm == 'roberta':
    model_name_or_path = "uer/chinese_roberta_L-12_H-768"
elif args.plm == 'ernie':
    model_name_or_path = "nghuyong/ernie-3.0-base-zh"
elif args.plm == 'bert-large':
    model_name_or_path = "yechen/bert-large-chinese"

batch_size = 8
device = "cuda"
lr = 5e-5

if args.type=='ptuningv2':
    peft_type = PeftType.PREFIX_TUNING
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.soft_tokens,encoder_hidden_size=1024)
elif args.type=='lora':
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=64, lora_alpha=16, lora_dropout=0.1)
elif args.type=='ptuningv1':
    peft_type = PeftType.P_TUNING
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=args.soft_tokens,
                                      encoder_hidden_size=1024)

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load Dataset
train_path = 'datasets/preprocessed/train.json'
dev_path = 'datasets/preprocessed/dev.json'
test_path = 'datasets/preprocessed/test.json'
datasets = DatasetDict.from_json({'train':train_path, 'dev':dev_path, 'test':test_path})

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default
    outputs = tokenizer(examples["claim"], examples["evidences"], truncation=True, max_length=256)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["claimId", "claim", "evidences"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, return_tensors="pt")

# Instantiate dataloaders.
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
dev_dataloader = DataLoader(
    tokenized_datasets["dev"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=3)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(args.num_epochs):
    model.train()
    total_loss = 0
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    # ========================================
    #               Training
    # ========================================
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        train_loss = outputs.loss
        total_loss += train_loss.sum().item()
        train_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print("Epoch {}, train_loss {}".format(epoch, total_loss / len(train_dataloader)), flush=True)
    
    # ========================================
    #               Validation
    # ========================================
    model.eval()
    valid_y_pred = []
    valid_y_true = []
    total_val_loss = 0
    for step, batch in enumerate(tqdm(dev_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        labels = batch['labels']
        val_loss = outputs.loss
        total_val_loss += val_loss.sum().item()
        predictions = outputs.logits.softmax(dim=-1)
        predictions = torch.argmax(predictions, dim=-1)
        valid_y_true.extend(labels.cpu().tolist())
        valid_y_pred.extend(predictions.cpu().tolist())
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro')
    microf1 = f1
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro')
    if f1 > best_macrof1:
        best_microf1 = microf1
        best_macrof1 = f1
        torch.save(model.state_dict(), f"./checkpoint/model.ckpt")
    print("Epoch {}, valid f1 {}".format(epoch, f1), flush=True)
    print("Epoch {}, valid_loss {}".format(epoch, total_val_loss / len(dev_dataloader)), flush=True)
    
    # early_stopping(total_val_loss, model)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

# ========================================
#               Test
# ========================================
model.load_state_dict(torch.load(f"./checkpoint/model.ckpt"))
model = model.to(device)
model.eval()
test_y_pred = []
test_y_true = []
for step, batch in enumerate(tqdm(test_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.softmax(dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    labels = batch['labels']
    test_y_true.extend(labels.cpu().tolist())
    test_y_pred.extend(predictions.cpu().tolist())
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))