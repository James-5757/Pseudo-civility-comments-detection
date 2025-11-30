import os, sys
import pandas as pd

# 1. 保证在脚本所在目录
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())

# 2. 读取人工标注的 gold_small.csv
gold = pd.read_csv("gold_small.csv")
gold["text"] = gold["text"].astype(str).str.strip()
gold = gold[gold["text"] != ""]
print("gold_small 标签分布:")
print(gold["gold_label"].value_counts())

# 3. 读取弱标注的 uncivil_candidates.csv
unciv = pd.read_csv("uncivil_candidates.csv")
unciv["text"] = unciv["text"].astype(str).str.strip()
unciv = unciv[unciv["text"] != ""]
print("uncivil_candidates 行数:", len(unciv))

# 只要 text 和 label 列，避免重复
unciv = unciv[["text"]].drop_duplicates()

# 4. 从候选里随机抽一些样本作为“弱监督 uncivil”
#    可以自己调这个数量，比如 50 或 100
N_EXTRA = 80
sample_n = min(N_EXTRA, len(unciv))
unciv_sample = unciv.sample(n=sample_n, random_state=42).copy()

# 给它们打上 gold_label = 'uncivil'，并标记来源方便以后分析
unciv_sample["gold_label"] = "uncivil"
unciv_sample["source"] = "uncivil_candidates_weak"

# 5. 把人工 gold 和弱 uncivil 合并
gold["source"] = "gold_manual"
gold_small_cols = ["text", "gold_label", "source"]

combined = pd.concat(
    [gold[gold_small_cols], unciv_sample[gold_small_cols]],
    ignore_index=True
)

print("合并后标签分布:")
print(combined["gold_label"].value_counts())
print("来源分布:")
print(combined["source"].value_counts())

# 6. 保存为新的 gold_augmented.csv
out_path = "gold_augmented.csv"
combined.to_csv(out_path, index=False, encoding="utf-8")
print("已保存扩展后的 gold 到:", out_path)
