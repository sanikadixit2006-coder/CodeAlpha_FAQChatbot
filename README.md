# 🤖 FAQ Chatbot — AI & Data Science

> **CodeAlpha Artificial Intelligence Internship Project**  
> Built with Python · NLTK · Scikit-learn · Tkinter

---

## 📌 Project Overview

This is a beginner-friendly **FAQ Chatbot** that answers questions related to **Artificial Intelligence, Machine Learning, NLP, Python, and Data Science**.

It uses **Natural Language Processing (NLP)** to understand user questions and finds the most relevant answer from a pre-built FAQ database using **TF-IDF vectorisation** and **Cosine Similarity** — no API keys or internet connection required.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 NLP Pipeline | Tokenisation, stop word removal, punctuation cleaning via NLTK |
| 📐 Smart Matching | TF-IDF + Cosine Similarity to find the best FAQ answer |
| ❓ Unknown Handling | Friendly error message when no good match is found |
| 🖥️ Clean GUI | Dark-themed Tkinter chat interface |
| 📄 20 FAQs | Covers AI, ML, Deep Learning, NLP, Python, Data Science |
| 💬 Enter key | Press Enter or click Send to ask a question |
| 🗑️ Clear chat | Reset the conversation with one click |

---

## 🗂️ Project Structure

```
faq_chatbot/
│
├── faq_chatbot.py      ← Main application (UI + NLP logic)
├── faq_data.json       ← FAQ questions and answers database
├── requirements.txt    ← Python library dependencies
└── README.md           ← This file
```

---

## ⚙️ How to Run in VS Code — Step by Step

### Step 1 — Prerequisites
Make sure you have the following installed:
- [Python 3.9+](https://www.python.org/downloads/) *(check: `python --version`)*
- [VS Code](https://code.visualstudio.com/)
- VS Code **Python extension** (install from the Extensions panel)

---

### Step 2 — Download / Clone the Project

**Option A — Download ZIP:**  
Click the green `Code` button on GitHub → `Download ZIP` → Extract the folder.

**Option B — Git clone:**
```bash
git clone https://github.com/YOUR_USERNAME/faq-chatbot.git
```

---

### Step 3 — Open the Project in VS Code

```
File → Open Folder → select the faq_chatbot folder
```

---

### Step 4 — Open the Integrated Terminal

```
Terminal → New Terminal   (or press Ctrl + ` )
```

---

### Step 5 — (Optional but Recommended) Create a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it — Windows:
venv\Scripts\activate

# Activate it — macOS/Linux:
source venv/bin/activate
```

---

### Step 6 — Install Dependencies

```bash
pip install -r requirements.txt
```

Expected output: packages for `nltk`, `scikit-learn`, and `numpy` will be downloaded and installed.

---

### Step 7 — Run the Chatbot

```bash
python faq_chatbot.py
```

The chatbot window will open. Type any question and press **Enter** or click **Send**. 🎉

---

## 💡 Example Questions to Try

```
What is Artificial Intelligence?
What is machine learning?
Explain deep learning
What is NLP?
What Python libraries are used in AI?
What is overfitting?
Tell me about cosine similarity
What skills do I need for data science?
What is a neural network?
What is tokenization?
```

---

## 🧠 How It Works — NLP Pipeline

```
User types question
        ↓
  Text Pre-processing (NLTK)
  • Lowercase
  • Tokenise into words
  • Remove punctuation
  • Remove stop words ("the", "is", "a"…)
        ↓
  TF-IDF Vectorisation (Scikit-learn)
  • Convert each text to a numeric vector
  • Words weighted by importance across documents
        ↓
  Cosine Similarity
  • Measure angle between user vector and each FAQ vector
  • Score: 1.0 = identical, 0.0 = completely different
        ↓
  Best Match selected
  • If score ≥ 0.15 → show the answer
  • If score < 0.15 → show "I don't understand" message
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `nltk` | Tokenisation, stop word removal |
| `scikit-learn` | TF-IDF vectoriser, cosine similarity |
| `numpy` | Numerical operations (used by scikit-learn) |
| `tkinter` | GUI — built into Python, no install needed |
| `json` | Load FAQ data — Python standard library |

---

## 🚀 Future Improvements

- [ ] Add more FAQs or load dynamically from a CSV
- [ ] Integrate a sentence transformer model (e.g., `sentence-transformers`) for better semantic matching
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Add conversation memory / multi-turn dialogue

---

## 👨‍💻 Author

**Your Name**  
Second Year — AI & Data Science Engineering  
CodeAlpha Artificial Intelligence Internship  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME/faq-chatbot)

---

## 📄 License

This project is open-source and free to use for educational purposes.