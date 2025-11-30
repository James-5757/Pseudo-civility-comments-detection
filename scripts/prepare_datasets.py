import os
import sys
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
print("当前工作目录:", os.getcwd())
import pandas as pd

# ================================================================
# 1. polite_df.csv  ←  politeness_train.csv
# ================================================================

def build_polite_df():
    df = pd.read_csv("politeness_train.csv")
    print("Politeness 列名:", df.columns)

    # 自动检测文本列名（常见：text、Utterance、sentence）
    possible_text_cols = ["text", "Utterance", "sentence", "utterance"]
    text_col = None
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError("找不到文本列，请告诉我 politeness_train.csv 里文本列叫什么名字。")

    # 自动检测礼貌标签列（politeness/label）
    possible_label_cols = ["politeness", "label", "polite", "class"]
    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("找不到礼貌标签列，你告诉我 politeness_train.csv 的标签列名，我帮你改。")

    polite_df = df[[text_col, label_col]].copy()
    polite_df.rename(columns={text_col: "text", label_col: "politeness_label"}, inplace=True)
    polite_df["id"] = range(len(polite_df))
    polite_df.to_csv("polite_df.csv", index=False, encoding="utf-8")
    print("✔ 已输出 polite_df.csv:", len(polite_df))


# ================================================================
# 2. friends_civil_df.csv  ←  friends_all_episodes_clean.csv
# ================================================================

def build_friends_df():
    df = pd.read_csv("friends_all_episodes_clean.csv")
    print("Friends 列名:", df.columns)

    # 只保留真正的对话行
    if "type" in df.columns:
        df = df[df["type"] == "dialogue"]

    # 文本列：dialogue_clean
    if "dialogue_clean" not in df.columns:
        raise ValueError("friends_all_episodes_clean.csv 里没有 dialogue_clean 列，请截个 columns 给我看。")
    text_col = "dialogue_clean"

    # 说话人列：speaker
    if "speaker" in df.columns:
        speaker_col = "speaker"
    else:
        df["speaker"] = "unknown"
        speaker_col = "speaker"

    friends_df = df[[text_col, speaker_col]].copy()
    friends_df.rename(columns={text_col: "text", speaker_col: "speaker"}, inplace=True)
    friends_df["id"] = range(len(friends_df))
    friends_df = friends_df[friends_df["text"].astype(str).str.strip() != ""]

    friends_df.to_csv("friends_civil_df.csv", index=False, encoding="utf-8")
    print("✔ 已输出 friends_civil_df.csv:", len(friends_df))



# ================================================================
# 3. toxicity_df.csv  ←  train.csv (Jigsaw Toxicity)
# ================================================================

def build_toxicity_df():
    df = pd.read_csv("train.csv")
    print("Toxicity 列名:", df.columns)

    if "comment_text" not in df.columns:
        raise ValueError("train.csv 没有 comment_text！请告诉我文本列名，我帮你改。")

    toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    toxic_cols = [c for c in toxic_cols if c in df.columns]

    if not toxic_cols:
        raise ValueError("train.csv 里找不到任何 toxicity 列，请告诉我实际列名我来改。")

    df["overall_toxicity"] = df[toxic_cols].max(axis=1)

    toxicity_df = df[["comment_text", "overall_toxicity"]].copy()
    toxicity_df.rename(columns={"comment_text": "text"}, inplace=True)
    toxicity_df["id"] = range(len(toxicity_df))
    toxicity_df.to_csv("toxicity_df.csv", index=False)
    print("✔ 已输出 toxicity_df.csv:", len(toxicity_df))


# ================================================================
# 4. attacks_df.csv  ←  toxicity_annotated_comments.tsv + toxicity_annotations.tsv
# ================================================================

def build_attacks_df():
    ann_comments = pd.read_csv("toxicity_annotated_comments.tsv", sep="\t")
    ann_labels = pd.read_csv("toxicity_annotations.tsv", sep="\t")

    print("annotated_comments 列名:", ann_comments.columns)
    print("annotations 列名:", ann_labels.columns)

    # 自动判断 ID 列名称（通常是 rev_id）
    if "rev_id" in ann_comments.columns:
        id_col = "rev_id"
    else:
        raise ValueError("annotated_comments.tsv 没有 rev_id！请上传列名截图我帮你改。")

    # 自动判断文本列
    if "comment" in ann_comments.columns:
        text_col = "comment"
    else:
        raise ValueError("annotated_comments.tsv 没有 comment 列，请截图我帮你改。")

    # 自动判断 attack 标签列
    if "attack" in ann_labels.columns:
        attack_col = "attack"
    elif "toxicity" in ann_labels.columns:
        attack_col = "toxicity"  # 有些版本叫这个
    else:
        raise ValueError("annotations.tsv 里找不到 attack/tocixity 列，请截图我改。")

    grouped = ann_labels.groupby(id_col)[attack_col].mean().reset_index()
    grouped.rename(columns={attack_col: "attack_score"}, inplace=True)

    merged = ann_comments.merge(grouped, on=id_col, how="left")
    attacks_df = merged[[id_col, text_col, "attack_score"]].copy()
    attacks_df.rename(columns={id_col: "id", text_col: "text"}, inplace=True)

    attacks_df.to_csv("attacks_df.csv", index=False)
    print("✔ 已输出 attacks_df.csv:", len(attacks_df))


# ================================================================
# main
# ================================================================

if __name__ == "__main__":
    build_polite_df()
    build_friends_df()
    build_toxicity_df()
    build_attacks_df()
