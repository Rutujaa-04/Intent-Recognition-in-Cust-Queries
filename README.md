# Customer Query Intent Recognition 🎯

An end-to-end Natural Language Processing (NLP) system designed to accurately identify customer intentions from conversational queries (chat, email, and support tickets), route requests automatically, detect out-of-scope queries, and escalate low-confidence interactions to human support.

---

## 📌 Project Overview

Customer support centers handle thousands of customer queries daily. Manually routing these tickets causes delays and high operational costs. This project builds a multi-tier intent classification engine that:
1. **Normalizes conversational text** (contractions, masked entity IDs, currency amounts, emails).
2. **Classifies customer requests** across 10 critical business intents.
3. **Calibrates confidence probabilities** to prevent confident misclassifications.
4. **Applies an Out-Of-Scope / Low-Confidence Fallback guardrail** to safely escalate ambiguous tickets to human representatives.
5. **Provides interpretability** by surfacing the key token n-grams driving each prediction.
6. **Exposes a REST API (FastAPI) and an interactive CLI REPL** for real-time testing and production serving.

---

## 🏷️ Supported Intent Taxonomy

| Intent | Description | Example Query |
| :--- | :--- | :--- |
| `track_order` | Status & courier delivery updates | *"Where is my package #8912 right now?"* |
| `request_refund` | Reimbursements and return claims | *"Can I get a full refund for this damaged shirt?"* |
| `cancel_order` | Order cancellation before shipment | *"I made a mistake, please cancel order ORD-331"* |
| `change_shipping_address` | Address or delivery updates | *"Change my shipping destination to 456 Elm Street"* |
| `account_issue` | Password reset, 2FA, profile login | *"I forgot my password and cannot sign in"* |
| `payment_problem` | Card declines, double billing, gateway errors | *"Why is there a duplicate $89 charge on my card?"* |
| `product_inquiry` | Sizing, specs, warranty, compatibility | *"Is this wireless mouse compatible with macOS?"* |
| `subscription_management`| Upgrades, downgrades, plan cancellation | *"How do I stop auto-renewal on my pro plan?"* |
| `contact_human_agent` | Explicit requests for human supervisor | *"Let me talk to a real human person please"* |
| `feedback_or_complaint` | Service feedback or escalation | *"The courier was rude and delivery was 2 weeks late"* |

---

## 🏗️ Project Architecture

```
Intent Recognition in Cust Queries/
├── data/
│   ├── raw/
│   │   └── customer_queries.json     # Curated benchmark dataset
│   └── processed/
│       ├── train.csv                 # 70% Stratified training set
│       ├── val.csv                   # 15% Validation set
│       └── test.csv                  # 15% Held-out test set
├── models/
│   ├── tfidf_intent_model.joblib     # Serialized TF-IDF + Classifier pipeline
│   └── label_encoder.joblib          # Target label encoder
├── reports/
│   ├── confusion_matrix.png          # High-resolution confusion matrix heatmap
│   └── error_analysis.csv            # Detailed misclassification analysis
├── src/
│   ├── config.py                     # Project configuration & hyperparameter constants
│   ├── data/
│   │   ├── preprocessor.py           # Text cleaning, contraction expansion, entity masking
│   │   └── dataset_generator.py      # Stratified dataset generator and loader
│   ├── models/
│   │   ├── baseline.py               # TF-IDF + LogisticRegression / LinearSVC pipeline
│   │   └── intent_classifier.py      # Inference wrapper with confidence thresholding & explainability
│   ├── evaluation/
│   │   └── metrics.py                # F1, accuracy, confusion matrix, error diagnostics
│   └── api/
│       └── app.py                    # FastAPI REST API service
├── tests/
│   ├── test_preprocessor.py          # Unit tests for text normalization
│   └── test_pipeline.py              # Unit tests for classification logic
├── train.py                          # Training & evaluation CLI script
├── predict.py                        # Interactive console & one-shot CLI prediction
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Set Up the Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model & Generate Reports
```bash
python train.py
```
This will:
- Generate and stratify the customer query dataset.
- Train the calibrated NLP pipeline.
- Print precision, recall, and F1-score across all 10 intents.
- Save the confusion matrix to `reports/confusion_matrix.png`.
- Output error diagnostics to `reports/error_analysis.csv`.

### 3. Test Inferences via CLI

**One-shot prediction:**
```bash
python predict.py --text "Where is my parcel #99102?"
```

**Interactive REPL Console:**
```bash
python predict.py
```

### 4. Run the REST API Service
```bash
uvicorn src.api.app:app --reload --port 8000
```
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Single prediction endpoint: `POST http://localhost:8000/predict`
- Batch prediction endpoint: `POST http://localhost:8000/predict/batch`

---

## 📈 Roadmap & Next Steps

1. **Transformer Fine-Tuning**: Fine-tune `distilbert-base-uncased` or `sentence-transformers/all-MiniLM-L6-v2` for state-of-the-art semantic intent matching.
2. **Out-of-Distribution (OOD) / Open-Set Recognition**: Train a thresholded Mahalanobis distance or contrastive embedding space for detecting unknown customer queries.
3. **Multi-Intent Detection**: Handle complex compound queries (e.g. *"I want to return this damaged jacket AND cancel my subscription"*).
4. **Interactive Web UI**: Streamlit or Gradio demo for live customer support ticket triage.
