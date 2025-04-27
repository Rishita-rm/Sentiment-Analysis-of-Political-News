# Sentiment Analysis of Political News 📰🔍

This project focuses on analyzing the **sentiment of political news headlines and descriptions** using deep learning.  
We fine-tune a **RoBERTa transformer model** to classify news into three categories: **Positive**, **Negative**, and **Neutral**.

---

## 🚀 Features
- Fine-tuning of pre-trained **RoBERTa** model for political news sentiment classification.
- **5-Fold Stratified Cross-Validation** to ensure robust and unbiased evaluation.
- **Confusion Matrix** visualization to analyze model performance.
- **LIME Explainability** to interpret model predictions.
- End-to-end, easily reproducible Jupyter Notebook workflow.

---

## 🗂 Dataset
- A merged dataset containing **210,000+ political news articles** from **2012–2022**.
- Each record includes the news headline, short description, authors, and publication date.

---

## 🛠 Tech Stack
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [LIME (Local Interpretable Model-Agnostic Explanations)](https://lime-ml.readthedocs.io/en/latest/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## 📁 Project Structure
lua
Copy
Edit
Sentiment_Analysis_of_Political_News_.ipynb
|-- Data Loading and Preprocessing
|-- RoBERTa Model Setup and Fine-tuning
|-- 5-Fold Cross-Validation Training
|-- Model Evaluation (Accuracy, F1 Score)
|-- Confusion Matrix Visualization
|-- LIME-based Model Interpretability

---

📋 Installation and Setup
Clone the repository

bash
Copy
Edit
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
Install the dependencies

bash
Copy
Edit
pip install transformers torch scikit-learn pandas matplotlib seaborn lime
Run the Jupyter Notebook

Open Sentiment_Analysis_of_Political_News_.ipynb

Follow the notebook sequentially.

---

## 📊 Results
Accuracy and F1 Scores across 5 different folds.

Confusion Matrix plotted to visualize classification performance.

LIME Explainability to understand individual news predictions.

---

## 🙏 Acknowledgments
HuggingFace for pre-trained models.

Sources of the political news dataset.

Scikit-learn and LIME contributors.

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Screenshots (Optional)
(Add screenshots of confusion matrix plots, LIME explanations, etc., if you want to make your repo look even cooler.)
