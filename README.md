# Pseudo-civility-comments-detection  
*A Weak Supervision + Gold Refinement Approach using BERT*

---

## 📘 Project Overview

Pseudo-civility refers to comments that **appear polite on the surface** (e.g., containing courteous expressions) but **carry negative pragmatic intent**.  
This project builds an **English pseudo-civility classifier** using:

- Multi-source weak labels  
- Rule-based pseudo-civil candidate mining  
- A small manually annotated gold dataset  
- Two-stage BERT fine-tuning  
- Evaluation analysis  

---

## 📚 Task Definition

**Pseudo-civil comments** are defined as:

> “Comments with polite linguistic surface forms (e.g., ‘thank you’, ‘with all due respect’)  
>  **but carrying negative pragmatic intent**, such as belittling intelligence, dismissing ideas,  
>  covert hostility, refined sarcasm, etc.”

We classify comments into **three categories**:

| Label | Meaning |
|-------|---------|
| `civil` | Genuinely polite or neutral comments, even when expressing disagreement |
| `uncivil` | Explicit insults, profanity, toxic hostility |
| `pseudo_civil` | Polite surface + negative semantic/pragmatic intent |

---

## 📦 Data Sources

We integrated **four public datasets** to build weak supervision signals:

### 1. **Wikipedia Toxicity (Jigsaw)**
- Provides explicit toxicity attributes.
- Used as the main source for *uncivil* weak labels.

### 2. **Wikipedia Personal Attacks**
- Contains crowdsourced scores for attack severity.
- Used to enrich *uncivil* and *pseudo-civil* candidates.

### 3. **Stanford Politeness Corpus**
- Used to detect polite markers and polite surface cues.

### 4. **Friends TV Dialogue Corpus**
- A large set of natural, mostly non-toxic dialog.
- Used as the source of *civil* weak labels.

---

## 🏗️ Weak Supervision Pipeline

We built a weakly-labeled dataset using heuristic rules:

### **Polite markers**  
Extracted via regex: thank you, thanks, please, I appreciate, with all due respect, kindly, sorry, no offense


### **Toxicity scores (0–1)**
Sourced from Wikipedia datasets.

### **Weak labeling rules**
| Condition | Label |
|----------|--------|
| `tox_score` ≤ 0.1 & no profanity | **civil** |
| `tox_score` ≥ 0.7 & profanity words | **uncivil** |
| polite markers present & 0.3 ≤ `tox_score` ≤ 0.8 | **pseudo_civil (weak)** |

Generated file:  
weak_labeled_dataset.csv


---

## 🔍 Pseudo-Civil Candidate Filtering

To refine pseudo-civil candidates, we applied:

- **Exclude heavy profanity**
- **Keep polite markers**
- **Keep mild negative semantic cues** such as:  
  `nonsense, ridiculous, ignorant, garbage, pathetic, no sense, useless`
- **Keep mid-range toxicity**

Generated file:
pseudo_civil_filtered_for_manual.csv


---

## ✍️ Manual Annotation (Gold Data)

From the filtered set, **~200 comments** were manually labeled into:

- `civil`
- `uncivil`
- `pseudo_civil`

Final gold file:
gold_small.csv

To balance the severely underrepresented `uncivil` class,  
we augmented the gold set with **80 high-confidence uncivil sentences** from the weak dataset:
gold_augmented.csv


---

## 📝 Annotation Guidelines (Final Version)

### **1. Civil**
A comment is labeled **civil** if:
- It conveys disagreement politely  
- It avoids attacking the interlocutor  
- It uses reasoning rather than belittlement  

**Examples**  
- “I understand your point, but I disagree.”  
- “Thank you for your explanation; I see it differently.”

---

### **2. Uncivil**
A comment is **uncivil** if:
- Contains profanity / insults  
- Targets the interlocutor’s identity, intelligence, or intent  
- Exhibits explicit hostility  

**Examples**  
- “You are an idiot.”  
- “This is the dumbest thing I’ve ever read.”

---

### **3. Pseudo-Civil**
A comment is **pseudo-civil** if:
- Contains polite surface cues  
- AND implicitly attacks competence / intelligence  
- AND contains dismissive or belittling implications  

**Examples**
- “Thank you for your brilliant idea, though it makes absolutely no sense.”  
- “With all due respect, no reasonable person would believe that.”

This category requires pragmatic judgment.

---

## 🤖 Model Architecture

We use **BERT-base-uncased** as a contextual embedding model.

### **Two-stage training pipeline**

#### **Stage 1 — Weak Supervision Training**
- Train BERT on `train_weak_balanced.csv`
- Learns coarse-grained distinctions (civil vs toxic)

#### **Stage 2 — Gold Refinement**
- Load `bert_weak_model`  
- Fine-tune on `gold_augmented.csv`  
- Use small learning rate `1e-5`  
- 5 epochs  

Purpose:
> Correct biases created by weak labels  
> and teach the model subtle pragmatic phenomena.

---

## 📊 Experimental Results
### **1. Analyze Gold model**
=== 最终评估结果（gold_small eval） ===
- eval_loss: 0.6456559300422668
- eval_macro_f1: 0.8291776838741032
- eval_civil_f1: 0.851063829787234
- eval_uncivil_f1: 0.9047619047619048
- eval_pseudo_civil_f1: 0.7817073170731707
- eval_accuracy: 0.8307692307692308
- eval_runtime: 4.3529
- eval_samples_per_second: 14.933
- eval_steps_per_second: 1.149
- epoch: 5.0

### **2. Weak model vs Gold model performance (on the same gold evaluation set)**

| Model | Macro F1 | Civil F1 | Uncivil F1 | Pseudo-Civil F1 | Accuracy |
|-------|----------|----------|-------------|------------------|----------|
| Weak Model | **0.51** | **0.00** | 0.88 | 0.67 | 0.625 |
| Gold Model | **0.83** | 0.80 | 0.90 | 0.78 | **0.821** |

### 🔍 Interpretation
- Weak model **fails completely** on civil (F1 = 0)  
  → It treats almost all polite comments as pseudo-civil or uncivil  
- Gold model greatly improves civil detection (0 → 0.80)  
- Pseudo-civil F1 increases from **0.67 → 0.78**  
- Macro-F1 jumps from **0.51 → 0.83**

## 🧭 Future Work

### 1. **Larger annotated pseudo-civil dataset**
- Multi-round annotation  
- Clearer guidelines  
- Better edge-case handling

### 2. **Semi-supervised learning**
- Self-training with model confidence  
- Noise-robust loss functions (e.g., symmetric CE)

### 3. **Sarcasm & irony modeling**
- Integrate sarcasm detection models  
- Use discourse act classification  
- Capture pragmatic conflicts explicitly

### 4. **Multilingual / cross-lingual transfer**
- Apply XLM-R / mBERT  
- Zero-shot pseudo-civil transfer to Chinese  
- Collect polite-but-hostile multilingual corpora

### 5. **Evaluation of robustness & calibration**
- Temperature scaling  
- Expected Calibration Error (ECE)  
- Adversarial rewrite test (manual paraphrases)

  ---

## 📁 Repository Structure (Recommended)
```bash
.
project/
│
├── data/
│ ├── weak_labeled_dataset.csv
│ ├── gold_small.csv
│ ├── gold_augmented.csv
│ ├── pseudo_civil_candidates.csv
│ ├── pseudo_civil_filtered_for_manual.csv
│
├── models/
│ ├── bert_weak_model/
│ ├── bert_gold_model/
│
├── scripts/
│ ├── prepare_datasets.py
│ ├── build_candidates.py
│ ├── filter_pseudo_civil.py
│ ├── make_gold_augmented.py
│ ├── train_bert_weak.py
│ ├── train_bert_gold.py
│ ├── compare_weak_vs_gold.py
│
├── README.md
└── requirements.txt
