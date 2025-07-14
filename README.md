# 🇸🇦 Saudi Dialect Classifier using SA-BERT  
*A Fine-Tuned Transformer Model for Real-World Saudi Text Classification*

## Project Summary  
This project leverages the **SA-BERT-V1** transformer model to fine-tune on a real-world dataset of Saudi dialectal text. The model learns to classify text into categories such as finance, daily life, transportation, and more — making it ideal for NLP applications in Saudi Arabia like digital assistants, sentiment-aware services, or smart categorization tools.

With a robust fine-tuning pipeline and an interactive Gradio app, this project offers a complete solution from training to live prediction.

---

## Project Idea  
Saudi dialects contain rich linguistic patterns tied to various topics and domains — from driving violations to financial transactions to casual conversation. The core idea of this project is to:

> **Fine-tune the SA-BERT-V1 model on labeled Saudi dialectal text, enabling automatic classification of real-life user inputs into meaningful categories.**

---

## Project Goals  
- Fine-tune a domain-specific BERT model on Saudi dialect classification data  
- Build an efficient, reproducible pipeline using Hugging Face Transformers and PyTorch  
- Explore and visualize text statistics (length, word count, label distribution)  
- Evaluate performance using metrics like accuracy, F1-score, and confusion matrix  
- Deploy a ready-to-use **Gradio demo** for live testing with user-friendly input/output

---

## Dataset

### Dataset  
- **Source:** [`AI-Diploma/saudi-dialect-classification-train`](https://huggingface.co/datasets/AI-Diploma/saudi-dialect-classification-train)  
- **Type:** Arabic texts labeled by domain/category  
- **Examples include:** finance, driving, daily life, and others  
- **Preprocessing:** Tokenization, text length analysis, padding/truncation, and stratified splitting  

### Pretrained Model & Tokenizer  
- **Model:** [`Omartificial-Intelligence-Space/SA-BERT-V1`](https://huggingface.co/Omartificial-Intelligence-Space/SA-BERT-V1)  
- **Tokenizer:** Same as model source  
- **Architecture:** BERT encoder with added classification head (fully connected layer)

---

## Fine-Tuning Pipeline  
- Tokenized using `AutoTokenizer.from_pretrained("Omartificial-Intelligence-Space/SA-BERT-V1")`  
- Model: `AutoModelForSequenceClassification` with number of labels from the dataset  
- Loss: Cross-Entropy  
- Optimizer: AdamW  
- Scheduler: Linear with warmup  
- Training Strategy:
  - Learning rate: `2e-5`
  - Batch size: `16`
  - Epochs: Up to 5 with early stopping (patience=2)
  - Validation during training  
- Evaluation Metrics:
  - `accuracy_score`, `f1_score`
  - Full `classification_report`
  - `confusion_matrix` visualization

---

## 📚 Lessons Learned  
Choosing the **right pretrained model** (SA-BERT-V1) trained on Arabic gave a huge boost in understanding dialectal text.  
**Dataset preparation matters** — exploring label distribution and cleaning text are critical for generalization.  
Adding **early stopping** prevents overfitting and improves robustness.  
Evaluation with **F1 and confusion matrix** gives much better insights than accuracy alone.  
**Real-world examples** are essential for validating the model’s practical use.  
Gradio is an excellent tool for **fast deployment and demo testing**.
