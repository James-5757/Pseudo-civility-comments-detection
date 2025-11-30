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
