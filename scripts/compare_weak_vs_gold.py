import os, sys
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 保证在脚本所在目录
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())

# 1. 读 gold_small 或 gold_augmented（看你用哪个训练 gold 模型）
gold_path = "gold_augmented.csv"   # 如果你用 gold_small 训练，就改成 gold_small.csv
df = pd.read_csv(gold_path, encoding="utf-8")

TEXT_COL = "text"
LABEL_COL = "gold_label"
df = df[[TEXT_COL, LABEL_COL]].copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]
df = df[df[LABEL_COL].isin(["civil", "uncivil", "pseudo_civil"])]

label2id = {"civil": 0, "uncivil": 1, "pseudo_civil": 2}
id2label = {v: k for k, v in label2id.items()}

df["label_id"] = df[LABEL_COL].map(label2id)

# 2. 固定一个 train / eval 划分，作为对比基准
train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COL],
)

print("eval 集大小:", len(eval_df))
print("eval 标签分布:")
print(eval_df[LABEL_COL].value_counts())

eval_ds = Dataset.from_pandas(eval_df[[TEXT_COL, "label_id"]])

def make_ds(tokenizer, df_):
    ds = Dataset.from_pandas(df_[[TEXT_COL, "label_id"]])
    def tokenize_fn(batch):
        return tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label_id"])
    return ds

def eval_model(model_dir, name):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    ds = make_ds(tokenizer, eval_df)

    import torch
    all_preds = []
    all_labels = []
    for batch in torch.utils.data.DataLoader(ds, batch_size=16):
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        labels = batch["label_id"].numpy()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            preds = torch.argmax(logits, dim=-1).numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(labels))

    report = classification_report(
        all_labels,
        all_preds,
        target_names=list(label2id.keys()),
        digits=4,
        zero_division=0,
    )
    print(f"\n====== {name} on gold eval ======")
    print(report)

# 3. 对比两个模型：weak vs gold
eval_model("./bert_weak_model", "WEAK MODEL")
eval_model("./bert_gold_model", "GOLD MODEL")
