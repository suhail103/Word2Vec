# Natural Language Processing Lab 1: Word2Vec

This project implements the **Skip-gram Word2Vec model** to analyze how word embeddings stabilize during training.  
Using text from the *Harry Potter* series, we explore whether certain words stabilize earlier than others and what factors influence this behavior.

---

## 🚀 Project Overview
The main objective was to **train Word2Vec from scratch** and study the temporal convergence of word embeddings.  
The workflow includes:

1. **Data Preprocessing**
   - Load raw text and clean it by removing punctuation.
   - Lowercase and tokenize text.
   - Create a vocabulary and index mapping for unique words.

2. **Skip-gram Dataset Creation**
   - Use a **context window size of 2** to generate input-target pairs.
   - Each target word is paired with its context words (two on each side).

3. **Model Architecture**
   - **Embedding layer:** Learns dense vector representations for words.
   - **Linear output layer:** Predicts context words given a target word.
   - Gaussian weight initialization to encourage stable learning.

4. **Stability Analysis**
   - Save embeddings after each epoch.
   - Calculate **nearest-neighbor overlap** across epochs to determine when each word stabilizes.
   - Compare stabilization timing with word frequency using **Spearman correlation**.

---

## 📊 Key Findings
- Some words stabilized **much earlier** than others, confirming varied convergence behavior.
- A weak correlation (ρ = 0.159) was found between frequency and stabilization speed.
- Stabilization is more influenced by **context consistency and specificity** rather than just frequency.

---

## 🗂 Repository Structure
