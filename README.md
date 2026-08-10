# NLP Sarcasm & Tone Classifier

A lightweight, self-contained Natural Language Processing (NLP) repository demonstrating sarcasm detection using **TF-IDF Feature Extraction** and **Logistic Regression**.

---

## 📌 Project Overview

Traditional software requires explicit rules. Natural Language Processing teaches machines to infer context and tone from textual data:

1. **Text Vectorization (TF-IDF):** Converts raw string sentences into weighted numerical matrices, extracting single words (`great`) and word pairs (`oh fantastic`).
2. **Classification (Logistic Regression):** Learns which phrases signal sarcasm (positive weights) versus genuine appreciation (negative weights).

---

## 📁 Repository Structure

```text
nlp-sarcasm-detector/
├── .gitignore         # Ignores venv, caches, and output binaries
├── requirements.txt   # Required Python libraries
├── main.py            # Complete executable training & prediction script
└── README.md          # Project documentation