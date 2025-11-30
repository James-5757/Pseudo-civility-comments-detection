import os
import sys
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ===== 1. 保证工作目录是脚本所在位置 =====
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())

# ===== 2. 读取 gold_augmented.csv =====
gold_path = "gold_augmented.csv"
df = pd.read_csv(gold_path)
print("gold_augmented 列名:", df.columns)
print("原始样本数:", len(df))

TEXT_COL = "text"
LABEL_COL = "gold_label"

df = df[[TEXT_COL, LABEL_COL]].copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]

# 只保留三种标签
valid_labels = ["civil", "uncivil", "pseudo_civil"]
df = df[df[LABEL_COL].isin(valid_labels)].copy()

print("标签分布（原始）:")
print(df[LABEL_COL].value_counts())

# ===== 3. 类别平衡：简单过采样，把各类样本数拉齐 =====
label_counts = df[LABEL_COL].value_counts()
max_n = label_counts.max()
print("各类样本数:", label_counts.to_dict(), " → 将统一到:", max_n)

balanced_list = []
rng = np.random.default_rng(42)
for label, count in label_counts.items():
    sub = df[df[LABEL_COL] == label]
    if count < max_n:
        # 有放回采样补到 max_n
        idx = rng.integers(0, count, size=(max_n - count))
        extra = sub.iloc[idx]
        sub_balanced = pd.concat([sub, extra], ignore_index=True)
    else:
        sub_balanced = sub
    balanced_list.append(sub_balanced)

df_balanced = pd.concat(balanced_list, ignore_index=True)
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("平衡后标签分布:")
print(df_balanced[LABEL_COL].value_counts())

# ===== 4. 划分 train / eval =====
train_df, eval_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=42,
    stratify=df_balanced[LABEL_COL],
)

print("train / eval 大小:", len(train_df), len(eval_df))

# 标签映射
label2id = {"civil": 0, "uncivil": 1, "pseudo_civil": 2}
id2label = {v: k for k, v in label2id.items()}

train_df = train_df.copy()
eval_df = eval_df.copy()
train_df["labels"] = train_df[LABEL_COL].map(label2id)
eval_df["labels"] = eval_df[LABEL_COL].map(label2id)

train_ds = Dataset.from_pandas(train_df[[TEXT_COL, "labels"]])
eval_ds = Dataset.from_pandas(eval_df[[TEXT_COL, "labels"]])

# ===== 5. 加载已有的弱监督模型 bert_weak_model 作为初始化 =====
model_dir = "./bert_weak_model"  # 你之前训练好的模型目录
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# ===== 6. Tokenize =====
def tokenize_fn(batch):
    return tokenizer(
        batch[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
eval_ds = eval_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ===== 7. 训练参数（第二阶段——小学习率 / 少 epoch） =====
training_args = TrainingArguments(
    output_dir="./bert_gold_outputs",
    num_train_epochs=5,                 # 数据量不大，可以多跑几轮
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,                 # 比第一阶段更小一点
    logging_steps=10,
    save_strategy="no",                 # 不保存中间 checkpoint，简单一点
)

# ===== 8. 评估函数，返回各类 F1 =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        target_names=list(label2id.keys()),
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "civil_f1": report["civil"]["f1-score"],
        "uncivil_f1": report["uncivil"]["f1-score"],
        "pseudo_civil_f1": report["pseudo_civil"]["f1-score"],
        "accuracy": report["accuracy"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ===== 9. 开始第二阶段训练 =====
print("开始第二阶段微调（基于 gold_small）...")
trainer.train()

# 训练完后在 eval 集上评估一次
eval_results = trainer.evaluate()
print("=== 最终评估结果（gold_small eval） ===")
for k, v in eval_results.items():
    print(f"{k}: {v}")

# ===== 10. 保存新模型 =====
final_model_dir = "./bert_gold_model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("已保存第二阶段模型到:", final_model_dir)
