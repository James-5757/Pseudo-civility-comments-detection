import os
import sys
import re
import pandas as pd

# 让脚本在自己所在目录运行（避免路径问题）
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())

# ----------------- 一些配置，可以按需要调整 ----------------- #

# 礼貌词/礼貌结构（用于在 toxic 语料中检测“礼貌外壳”）
POLITE_PATTERNS = [
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bthank u\b",
    r"\bplease\b",
    r"\bi appreciate\b",
    r"\bi appreciate your\b",
    r"\bwith all due respect\b",
    r"\bi (am|\'m) sorry\b",
    r"\bno offense\b",
    r"\bi respect your\b",
    r"\bi understand your\b",
    r"\bkindly\b",
]

# 粗暴攻击/脏话词（用来过滤掉明显不文明的 civil 候选）
TOXIC_WORDS = [
    r"\bidiot\b",
    r"\bstupid\b",
    r"\bdumb\b",
    r"\bmoron\b",
    r"\bshut up\b",
    r"\bfuck\b",
    r"\bshit\b",
    r"\basshole\b",
    r"\bbitch\b",
    r"\bretard(ed)?\b",
    r"\bkill yourself\b",
    r"\bhate\b",
]

# 阈值可以按需要微调
PSEUDO_TOX_THRESHOLD = 0.3   # 毒性高于这个 + 有礼貌词 → 伪文明候选
UNCIVIL_TOX_THRESHOLD = 0.7  # 毒性特别高 + 无礼貌词 → 不文明候选

MAX_CIVIL_SAMPLES = 3000     # civil 候选最多保留多少条
MAX_UNCIVIL_SAMPLES = 3000   # uncivil 候选最多保留多少条
MAX_PSEUDO_SAMPLES = 3000    # 伪文明候选最多保留多少条


def build_candidates():
    # -------------- 读取四个基础 CSV -------------- #
    polite_df = pd.read_csv("polite_df.csv")
    friends_df = pd.read_csv("friends_civil_df.csv")
    toxicity_df = pd.read_csv("toxicity_df.csv")
    attacks_df = pd.read_csv("attacks_df.csv")

    print("polite_df.csv:", polite_df.shape)
    print("friends_civil_df.csv:", friends_df.shape)
    print("toxicity_df.csv:", toxicity_df.shape)
    print("attacks_df.csv:", attacks_df.shape)

    # -------------- 1. 构造一个 combined toxic 表 -------------- #
    # 来自 jigsaw toxicity 的整体毒性分
    tox_part = toxicity_df[["text", "overall_toxicity"]].copy()
    tox_part.rename(columns={"overall_toxicity": "tox_score"}, inplace=True)
    tox_part["source"] = "wiki_toxicity"

    # 来自 personal attacks 的攻击分
    atk_part = attacks_df[["text", "attack_score"]].copy()
    atk_part.rename(columns={"attack_score": "tox_score"}, inplace=True)
    atk_part["source"] = "wiki_attacks"

    # 合在一起
    tox_all = pd.concat([tox_part, atk_part], ignore_index=True)
    tox_all["text"] = tox_all["text"].astype(str).str.strip()
    tox_all = tox_all[tox_all["text"] != ""]
    tox_all.drop_duplicates(subset=["text", "source"], inplace=True)
    print("合并后 toxic+attacks 行数:", tox_all.shape)

    # -------------- 2. 检测礼貌词 / 粗暴词 -------------- #
    polite_regex = re.compile("|".join(POLITE_PATTERNS), flags=re.IGNORECASE)
    toxicword_regex = re.compile("|".join(TOXIC_WORDS), flags=re.IGNORECASE)

    def has_polite_markers(text: str) -> bool:
        return bool(polite_regex.search(text))

    def has_toxic_words(text: str) -> bool:
        return bool(toxicword_regex.search(text))

    tox_all["has_polite"] = tox_all["text"].apply(has_polite_markers)

    # -------------- 3. 伪文明候选：礼貌词 + 毒性中高 -------------- #
    pseudo_mask = (tox_all["has_polite"]) & (tox_all["tox_score"] >= PSEUDO_TOX_THRESHOLD)
    pseudo_df = tox_all[pseudo_mask].copy()
    pseudo_df["label"] = "pseudo_civil"

    if len(pseudo_df) > MAX_PSEUDO_SAMPLES:
        pseudo_df = pseudo_df.sample(n=MAX_PSEUDO_SAMPLES, random_state=42)

    pseudo_df.to_csv("pseudo_civil_candidates.csv", index=False, encoding="utf-8")
    print("✔ 已保存 pseudo_civil_candidates.csv:", len(pseudo_df))

    # -------------- 4. 不文明候选：毒性很高 + 没有礼貌词 -------------- #
    uncivil_mask = (~tox_all["has_polite"]) & (tox_all["tox_score"] >= UNCIVIL_TOX_THRESHOLD)
    uncivil_df = tox_all[uncivil_mask].copy()
    uncivil_df["label"] = "uncivil"

    if len(uncivil_df) > MAX_UNCIVIL_SAMPLES:
        uncivil_df = uncivil_df.sample(n=MAX_UNCIVIL_SAMPLES, random_state=42)

    uncivil_df.to_csv("uncivil_candidates.csv", index=False, encoding="utf-8")
    print("✔ 已保存 uncivil_candidates.csv:", len(uncivil_df))

    # -------------- 5. 文明候选：Friends 对话里无粗口/攻击词 -------------- #
    friends_df["text"] = friends_df["text"].astype(str).str.strip()
    friends_df = friends_df[friends_df["text"] != ""]

    friends_df["has_toxic_word"] = friends_df["text"].apply(has_toxic_words)
    civil_df = friends_df[~friends_df["has_toxic_word"]].copy()
    civil_df = civil_df[["text", "speaker"]]
    civil_df["source"] = "friends"
    civil_df["label"] = "civil"

    if len(civil_df) > MAX_CIVIL_SAMPLES:
        civil_df = civil_df.sample(n=MAX_CIVIL_SAMPLES, random_state=42)

    civil_df.to_csv("civil_candidates.csv", index=False, encoding="utf-8")
    print("✔ 已保存 civil_candidates.csv:", len(civil_df))

    # -------------- 6. 合并三个弱标注集合，给后续标注/训练用 -------------- #
    # 为了统一结构，只保留 text + label + tox_score(如果有) + source
    pseudo_small = pseudo_df[["text", "label", "tox_score", "source"]]
    uncivil_small = uncivil_df[["text", "label", "tox_score", "source"]]
    civil_small = civil_df[["text", "label", "source"]].copy()
    civil_small["tox_score"] = 0.0

    all_weak = pd.concat([pseudo_small, uncivil_small, civil_small], ignore_index=True)
    all_weak.to_csv("weak_labeled_dataset.csv", index=False, encoding="utf-8")
    print("✔ 已保存 weak_labeled_dataset.csv:", len(all_weak))


if __name__ == "__main__":
    build_candidates()
