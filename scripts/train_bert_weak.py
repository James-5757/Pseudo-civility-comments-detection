import os, sys
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import classification_report

# 1. 工作目录
script_path = os.path.abspath(sys.argv[0])
os.chdir(os.path.dirname(script_path))

# 2. 读取弱监督训练集
df = pd.read_csv("train_weak_balanced.csv")
df["text"] = df["text"].astype(str)

# 把标签映射成数字
label2id = {"civil": 0, "uncivil": 1, "pseudo_civil": 2}
id2label = {v: k for k, v in label2id.items()}
df = df[df["label"].isin(label2id.keys())]
df["labels"] = df["label"].map(label2id)

# 简单划分 train / eval
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
eval_ds = Dataset.from_pandas(eval_df[["text", "labels"]])

# 3. 选择一个英文预训练模型
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
eval_ds = eval_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./bert_weak_outputs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
   #evaluation_strategy="epoch",
   #save_strategy="epoch",
    logging_steps=50,
   #load_best_model_at_end=True,
   #metric_for_best_model="eval_loss",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, target_names=list(label2id.keys()), output_dict=True)
    # 这里返回 macro F1
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "civil_f1": report["civil"]["f1-score"],
        "uncivil_f1": report["uncivil"]["f1-score"],
        "pseudo_civil_f1": report["pseudo_civil"]["f1-score"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 5. 开始训练
trainer.train()
# ✅ 手动评估 —— 不涉及 TrainingArguments 的构造，因此不会触发之前那个 TypeError
eval_results = trainer.evaluate()
print("Eval results:", eval_results)
trainer.save_model("./bert_weak_model")
tokenizer.save_pretrained("./bert_weak_model")

print("训练完成，模型保存在 bert_weak_model/ 目录。")
