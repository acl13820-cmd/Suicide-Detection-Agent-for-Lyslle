# Non-Therapist Suicide Risk Detection Chat Agent

**Course**: QMSS Practicum (Fall 2025)  
**Client**: Lyslle  
**Project Type**: Prototype Design & Implementation

---

## Overview

This repository contains a prototype **non-therapist safety chat agent** designed to detect suicidal ideation in user-provided text. The project was developed for the QMSS Practicum course and focuses on **transparent, modular, and interpretable suicide-risk detection**, rather than therapeutic intervention or response generation.

The system is implemented as a **local Streamlit web application** and integrates multiple components—data preprocessing, large language model (LLM) analysis, and classical machine-learning classification—into a single, testable pipeline.

The primary goal of the prototype is **accurate and conservative detection of suicide risk**, prioritizing safety and minimizing false negatives.

---

## Repository Structure

- `app.py` — Streamlit app wrapping the detection agent  
- `*.ipynb` — Data description, preprocessing, and model training  
- `lr_best` — Exported Logistic Regression model  
- `myvec` — Exported TF-IDF vectorizer  
- `Lyslle Practicum Memo.docx` / — Design decisions and methodology  
- `README.md`


---

## Key Files

### Notebook (`.ipynb`)
- Documents:
  - Dataset sources and structure
  - Text preprocessing steps
  - Model training and evaluation
  - Hyperparameter tuning
- Serves as the primary record of experimentation and model selection.

### `app.py`
- Wraps the suicide detection pipeline into a **local Streamlit interface**
- Enables interactive testing of:
  - De-identification
  - Individual model outputs
  - Final risk classification and system action

### Exported Models
- **`lr_best`**: Trained Logistic Regression classifier
- **`myvec`**: TF-IDF vectorizer used during training and inference

### Memo
- Provides detailed documentation of:
  - Design rationale
  - Ethical considerations
  - Modeling decisions
  - System limitations

---

## System Architecture

The prototype follows a **three-stage detection pipeline**.

### 1. De-identification Layer
Before analysis, user input is cleaned to remove or mask:
- Names
- Phone numbers
- Dates
- Addresses
- Emails
- URLs

This step is intentionally lightweight: it reduces the risk of exposing sensitive information to external models while preserving emotional and psychological meaning.

---

## Detection Mechanisms

### Part A: LLM-Based Detection

- Uses **gpt-4o-mini**
- Prompted to return a structured **JSON output** containing:
  - Risk level: `High`, `Medium`, or `Low`
  - Category label
- Strengths:
  - Captures contextual and implicit signals
  - Performs well on long or ambiguous messages
  - Interprets tone and emotional trajectory beyond keywords

---

### Part B: Machine-Learning-Based Detection

- Trained on social media text from two Kaggle datasets:
  - *Suicide and Depression Detection* (232,074 messages)
  - Supplementary dataset (9,999 messages)
- Binary labels: `suicidal` / `non-suicidal`

#### Preprocessing
- Lowercasing
- Stopword removal
- Stemming (PorterStemmer)
- TF-IDF vectorization

#### Models Evaluated
- Logistic Regression
- Random Forest
- XGBoost

#### Performance Summary

**Logistic Regression**

| Class            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Non-suicidal     | 0.93      | 0.94   | 0.94     |
| Suicidal         | 0.94      | 0.93   | 0.93     |

**Random Forest**

| Class            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Non-suicidal     | 0.92      | 0.90   | 0.91     |
| Suicidal         | 0.90      | 0.92   | 0.91     |

**XGBoost**

| Class            | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| Non-suicidal     | 0.89      | 0.94   | 0.91     |
| Suicidal         | 0.93      | 0.89   | 0.91     |

#### Model Selection

Logistic Regression was selected for the current prototype because it:
- Performs strongly on high-dimensional sparse text data
- Is more stable against overfitting
- Allows inspection of feature weights for interpretability
- Provides a reliable baseline to complement LLM outputs

---

## Part C: Final Classification Logic

Rather than averaging predictions, the system applies a **conservative decision rule**:

> The final risk level is the **higher risk output** produced by either model.

This design reflects crisis-intervention principles, where **false negatives are more dangerous than false positives**.

The final output includes:
- Risk level
- Classification category
- Recommended system action (e.g., monitor, escalate to therapist)

---

## Design Principles

- **Safety-first decision making**
- **Transparency** in model behavior and outputs
- **Modularity**, allowing components to be swapped or improved independently
- **Local deployment** for testing and demonstration purposes

---

## Limitations and Future Work

Planned or potential improvements include:
- Expanded and more diverse training data
- More granular risk categories
- Improved de-identification methods
- Reinforcement-learning-based response generation
- Longitudinal user-state tracking

---

## Disclaimer

This prototype is **not a therapeutic tool** and is **not intended for real-world clinical deployment**. It is a research and design exploration focused on detection mechanisms, interpretability, and system integration.

---

## License

For academic and educational use only. Refer to individual dataset licenses for data usage constraints.
