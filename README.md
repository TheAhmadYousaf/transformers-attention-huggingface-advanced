# 🤗 Transformers, Attention & Hugging Face — Advanced Implementation

An advanced deep learning project implementing self-attention from scratch, exploring three Hugging Face pipelines, inspecting DistilBERT internals, and fine-tuning using the Trainer API. This is Project 2 in the Transformers series.

---

## 🎯 Objective
- Implement scaled dot-product self-attention manually in PyTorch
- Use Hugging Face pipelines for multiple NLP tasks
- Inspect tokenization and visualize attention weights
- Fine-tune DistilBERT using Trainer API with full evaluation metrics

---

## 📂 Project Structure
```
transformers-attention-huggingface-advanced/
│
├── Transformers_Attention_HuggingFace_Advanced.ipynb   # Main Notebook
├── movie_reviews_advanced.csv                           # Dataset (600 samples)
└── README.md
```

---

## 📊 Dataset
- **Source:** IMDb Movie Reviews / Synthetic
- **Size:** 600 samples
- **Labels:** Positive (1), Negative (0)
- **Split:** 80% train / 20% test

---

## 🔧 Tech Stack
- Python 3.10
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- Scikit-learn
- Matplotlib, Seaborn

---

## 🧠 Parts Covered

| Part | Marks | Description |
|------|-------|-------------|
| Part A | 5 | Self-attention from scratch — Q, K, V, softmax, weighted output, heatmap |
| Part B | 5 | Three pipelines — sentiment, zero-shot classification, text generation |
| Part C | 5 | AutoTokenizer + AutoModel, attention visualization, WordPiece explanation |
| Part D | 7 | Fine-tuning DistilBERT with Trainer API — Accuracy, F1, Precision, Recall |

---

## 📉 Results

| Metric | Pre-trained Pipeline | Fine-tuned DistilBERT |
|--------|---------------------|----------------------|
| Accuracy | 90.00% | 51.67% |
| F1 Score | 0.9474 | 0.6813 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9000 | 0.5167 |

> Note: Pre-trained pipeline outperformed fine-tuned here because the dataset was small (600 samples) and the pipeline was already trained on similar sentiment data. With 5000+ samples fine-tuned BERT would outperform the general pipeline.

---

## 💡 Key Concepts Covered
- Scaled dot-product self-attention: QKᵀ / √d_k
- Softmax role and scaling factor explanation
- Encoder-only vs Decoder-only vs Encoder-Decoder models
- WordPiece tokenization and subword units
- Multi-head attention visualization per layer
- Hugging Face Trainer API with warmup scheduler
- Transfer learning and fine-tuning

---

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**

Open notebook in Colab and run all cells.

**Option 2 — Local**
```bash
git clone https://github.com/TheAhmadYousaf/transformers-attention-huggingface-advanced.git
cd transformers-attention-huggingface-advanced
pip install transformers datasets torch scikit-learn seaborn matplotlib accelerate evaluate
jupyter notebook Transformers_Attention_HuggingFace_Advanced.ipynb
```

---

## 🔗 Related Projects
- [Project 1 — Transformers & Hugging Face Basic](https://github.com/TheAhmadYousaf/transformers-huggingface-bert-sentiment)


---

## 👤 Author
**Ahmad Yousaf**
BS Artificial Intelligence
