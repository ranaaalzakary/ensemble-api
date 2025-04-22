# Disable unwanted warnings to keep logs clean
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==== Import core libraries ====
import joblib  # For loading Random Forest and XGBoost models + vectorizers
import torch  # For using MobileBERT model with PyTorch
import numpy as np  # For handling numeric arrays
import pandas as pd  # For tabular feature handling
from bs4 import BeautifulSoup  # For stripping HTML tags from email content
from urllib.parse import urlparse  # For extracting domain from email content
import whois  # For extracting domain age (used as a feature)

# ==== Import Hugging Face transformer tools ====
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

# ==== Import FastAPI tools ====
from fastapi import FastAPI
from pydantic import BaseModel

# ==== Initialize FastAPI ====
app = FastAPI()

# ==== Root endpoint just to verify API is up ====
@app.get("/")
def root():
    return {"status": "API is running"}

# ==== Define expected request format ====
class EmailRequest(BaseModel):
    email: str

# ==== Load trained models ====
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# ==== Load vectorizers ====
tfidf = joblib.load("tfidf_vectorizer.pkl")
count = joblib.load("count_vectorizer.pkl")
xgb_tfidf = joblib.load("xgb_tfidf_vectorizer.pkl")
xgb_count = joblib.load("xgb_count_vectorizer.pkl")

# ==== Load fine-tuned MobileBERT from local path ====
bert_path = "./"  # Assuming files are in the same folder as this script
tokenizer = MobileBertTokenizer.from_pretrained(bert_path, local_files_only=True)
bert_model = MobileBertForSequenceClassification.from_pretrained(bert_path, local_files_only=True)
bert_model.eval()  # Set to evaluation mode

# ==== Feature extraction functions ====
def extract_html_features(text):
    soup = BeautifulSoup(text, "html.parser")
    return {
        "num_forms": len(soup.find_all("form")),
        "num_scripts": len(soup.find_all("script")),
        "num_links": len(soup.find_all("a")),
    }

def extract_lexical_features(text):
    return {
        "num_special_chars": sum(1 for c in text if not c.isalnum() and not c.isspace()),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "num_uppercase": sum(1 for c in text if c.isupper()),
    }

def extract_host_features(text):
    try:
        domain = urlparse(text).netloc
        info = whois.whois(domain)
        created = info.creation_date[0] if isinstance(info.creation_date, list) else info.creation_date
        age = (pd.Timestamp.now() - pd.to_datetime(created)).days if created else 0
    except Exception as e:
        print(f"WHOIS lookup failed: {e}")
        age = 0
    return {"domain_age": age}

# ==== Prediction endpoint ====
@app.post("/predict")
async def predict_email(data: EmailRequest):
    email_text = data.email.strip()

    # Tokenize email for MobileBERT
    tokens = tokenizer(email_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        output = bert_model(**tokens)
    probs_bert = torch.softmax(output.logits, dim=1).squeeze().numpy()
    bert_pred = int(np.argmax(probs_bert))
    bert_conf = float(np.max(probs_bert))

    # Extract handcrafted features
    html_feats = extract_html_features(email_text)
    lexical_feats = extract_lexical_features(email_text)
    host_feats = extract_host_features(email_text)

    # Vectorize for RF
    tfidf_vec = tfidf.transform([email_text]).toarray()
    count_vec = count.transform([email_text]).toarray()
    rf_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        tfidf_vec[0],
        count_vec[0]
    ])

    # Vectorize for XGB
    xgb_tfidf_vec = xgb_tfidf.transform([email_text]).toarray()
    xgb_count_vec = xgb_count.transform([email_text]).toarray()
    xgb_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        xgb_tfidf_vec[0],
        xgb_count_vec[0]
    ])

    # Predict using RF
    rf_probs = rf_model.predict_proba(pd.DataFrame([rf_features]))[0]
    rf_pred = int(np.argmax(rf_probs))
    rf_conf = float(np.max(rf_probs))

    # Predict using XGB
    xgb_probs = xgb_model.predict_proba(pd.DataFrame([xgb_features]))[0]
    xgb_pred = int(np.argmax(xgb_probs))
    xgb_conf = float(np.max(xgb_probs))

    # Weighted ensemble
    final_score = (
        bert_conf * 0.5 * bert_pred +
        rf_conf * 0.25 * rf_pred +
        xgb_conf * 0.25 * xgb_pred
    )
    final_pred = 1 if final_score >= 0.5 else 0

    print(f"[PREDICTION] Final: {final_pred} | Score: {final_score:.4f} | BERT: {bert_conf:.4f}, RF: {rf_conf:.4f}, XGB: {xgb_conf:.4f}")

    return {
        "final_prediction": "phishing" if final_pred == 1 else "safe",
        "final_score": round(final_score, 4),
        "bert_confidence": round(bert_conf, 4),
        "rf_confidence": round(rf_conf, 4),
        "xgb_confidence": round(xgb_conf, 4)
    }