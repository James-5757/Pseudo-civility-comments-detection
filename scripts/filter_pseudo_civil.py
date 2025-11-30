import os, sys

# === 强制进入脚本所在的目录 ===
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

print("当前工作目录：", os.getcwd())

import pandas as pd
import re

# 1. 读取 pseudo_civil_candidates.csv
#    确保这个脚本和 pseudo_civil_candidates.csv 在同一个文件夹下
input_path = "pseudo_civil_candidates.csv"
df = pd.read_csv(input_path)
print("原始伪文明候选数量:", len(df))
print("列名:", df.columns.tolist())

# 2. 定义词表：强脏话（要排除），轻度贬义（我们想保留）
strong_obscene_patterns = [
    r"\bfuck(ing)?\b", r"\bfucked\b", r"\bcocksucker\b", r"\bcock\b",
    r"\bcunt\b", r"\bmotherfucker\b", r"\bshit\b", r"\basshole\b",
    r"\bbitch(es)?\b", r"\bslut\b", r"\bwhore\b", r"\bfaggot\b",
    r"\bnigger\b", r"\bretard(ed)?\b", r"\bdick\b", r"\bpiss\b", r"\bcrap\b"
]

mild_neg_patterns = [
    r"\bnonsense\b", r"\bridiculous\b", r"\bignorant\b", r"\bstupid\b",
    r"\bclueless\b", r"\bmeaningless\b", r"\bwaste\b", r"\bgarbage\b",
    r"\btrash\b", r"\bpathetic\b", r"\bworst\b", r"\bterrible\b", r"\bawful\b",
    r"\binsulting\b", r"\boffensive\b", r"\bno sense\b", r"\bnot helpful\b",
    r"\bpointless\b", r"\buseless\b"
]

strong_re = re.compile("|".join(strong_obscene_patterns), re.IGNORECASE)
mild_re = re.compile("|".join(mild_neg_patterns), re.IGNORECASE)

# 3. 标记两种词：强脏话 + 轻度贬义
df["has_strong_obscene"] = df["text"].astype(str).apply(
    lambda t: bool(strong_re.search(t))
)
df["has_mild_negative"] = df["text"].astype(str).apply(
    lambda t: bool(mild_re.search(t))
)

# 4. 按规则筛选“更像伪文明”的句子：
#    - has_polite = True               （之前规则选过的礼貌句）
#    - has_mild_negative = True        （有轻度贬义/否定词）
#    - has_strong_obscene = False      （排除特别粗暴的脏话）
#    - 0.3 <= tox_score <= 0.8         （中度毒性，不是极端骂人）
filtered = df[
    (df["has_polite"] == True) &
    (df["has_mild_negative"] == True) &
    (df["has_strong_obscene"] == False) &
    (df["tox_score"] >= 0.3) &
    (df["tox_score"] <= 0.8)
].copy()

print("筛选后数量:", len(filtered))

# 5. 保存为新文件，方便你用 Excel 打开慢慢挑
output_path = "pseudo_civil_filtered_for_manual.csv"
filtered.to_csv(output_path, index=False, encoding="utf-8")
print("已保存到:", output_path)

# 6. 随机看几条样本，感受一下效果
print("\n示例几条：")
for i, row in filtered.sample(min(5, len(filtered)), random_state=42).iterrows():
    print("----")
    print(row["text"])
    print("tox_score:", row["tox_score"])
