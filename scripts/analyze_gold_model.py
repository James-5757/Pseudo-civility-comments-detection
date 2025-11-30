import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 0. 基本设置
# =========================

# 保证脚本在自己的目录下运行
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())

# gold 数据路径（按你实际训练的情况改）
GOLD_PATH = "gold_augmented.csv"   # 如果用 gold_small 训的，就改成 "gold_small.csv"
MODEL_DIR = "./bert_gold_model"    # 第二阶段微调后的模型

TEXT_COL = "text"
LABEL_COL = "gold_label"

label2id = {"civil": 0, "uncivil": 1, "pseudo_civil": 2}
id2label = {v: k for k, v in label2id.items()}
label_names = ["civil", "uncivil", "pseudo_civil"]

# =========================
# 1. 读取 gold 数据并划分 train / eval（与训练脚本保持一致）
# =========================
df = pd.read_csv(GOLD_PATH, encoding="utf-8")
print("gold 数据列名:", df.columns.tolist())
df = df[[TEXT_COL, LABEL_COL]].copy()

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]
df = df[df[LABEL_COL].isin(label2id.keys())]

print("gold 标签分布（全部）:")
print(df[LABEL_COL].value_counts())

df["label_id"] = df[LABEL_COL].map(label2id)

# 与训练时相同：stratify + random_state=42
train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COL],
)

print("\nEval 集大小:", len(eval_df))
print("Eval 标签分布:")
print(eval_df[LABEL_COL].value_counts())

# =========================
# 2. 加载模型并对 eval 集做预测
# =========================
print("\n加载模型:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

eval_ds = Dataset.from_pandas(eval_df[[TEXT_COL, "label_id"]])

def tokenize_fn(batch):
    return tokenizer(
        batch[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

eval_ds = eval_ds.map(tokenize_fn, batched=True)
eval_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label_id"],
)

eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=16)

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label_id"].numpy()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)

        all_labels.extend(list(labels))
        all_preds.extend(list(preds))
        all_probs.extend(list(probs))

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

print("\n=== classification_report（gold model on gold eval） ===")
print(classification_report(
    all_labels,
    all_preds,
    target_names=label_names,
    digits=4,
    zero_division=0,
))

# =========================
# 3. 结果展示 1：混淆矩阵
# =========================
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
print("\n混淆矩阵（行=真实，列=预测）：")
print(cm)

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )

    # 在每个格子里写数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.show()

plot_confusion_matrix(cm, label_names, title="Confusion Matrix (Gold Model on Gold Eval)")

# =========================
# 4. 结果展示 2：每类 Precision / Recall 柱状图
# =========================
report_dict = classification_report(
    all_labels,
    all_preds,
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)

precisions = [report_dict[name]["precision"] for name in label_names]
recalls = [report_dict[name]["recall"] for name in label_names]

# Precision 柱状图
plt.figure(figsize=(6,4))
plt.bar(label_names, precisions)
plt.ylim(0, 1.05)
plt.ylabel("Precision")
plt.title("Per-class Precision (Gold Model)")
for i, p in enumerate(precisions):
    plt.text(i, p + 0.02, f"{p:.2f}", ha="center")
plt.tight_layout()
plt.show()

# Recall 柱状图
plt.figure(figsize=(6,4))
plt.bar(label_names, recalls)
plt.ylim(0, 1.05)
plt.ylabel("Recall")
plt.title("Per-class Recall (Gold Model)")
for i, r in enumerate(recalls):
    plt.text(i, r + 0.02, f"{r:.2f}", ha="center")
plt.tight_layout()
plt.show()

# =========================
# 5. 结果展示 3：置信度分布（每类的 predicted probability）
# =========================
# 对每个样本：取模型对 “预测类别” 的最大置信度
max_conf = all_probs.max(axis=1)

plt.figure(figsize=(6,4))
plt.hist(max_conf, bins=10, alpha=0.8)
plt.xlabel("Max predicted probability")
plt.ylabel("Count")
plt.title("Overall Confidence Distribution (Gold Model)")
plt.tight_layout()
plt.show()

# 也可以按真实类别，画 “对真实标签的概率分布”
plt.figure(figsize=(6,4))
for lab_id, name in enumerate(label_names):
    # 模型给“真实类”的置信度
    true_mask = (all_labels == lab_id)
    true_probs = all_probs[true_mask, lab_id]
    plt.hist(true_probs, bins=10, alpha=0.5, label=name)

plt.xlabel("P(true class)")
plt.ylabel("Count")
plt.title("Confidence for True Class by Label")
plt.legend()
plt.tight_layout()
plt.show()
