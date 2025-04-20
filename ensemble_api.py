import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import torch
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import whois
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel

# === Load models ===
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

# === Load vectorizers ===
rf_count_vec = joblib.load("count_vectorizer.pkl")
rf_tfidf_vec = joblib.load("tfidf_vectorizer.pkl")
xgb_count_vec = joblib.load("xgb_count_vectorizer.pkl")
xgb_tfidf_vec = joblib.load("xgb_tfidf_vectorizer.pkl")

# === Load BERT ===
tokenizer = MobileBertTokenizer.from_pretrained("./")
bert_model = MobileBertForSequenceClassification.from_pretrained("./")
bert_model.eval()

# === Feature Extraction ===
def extract_html_features(text):
    soup = BeautifulSoup(text, "html.parser")
    return [
        len(soup.find_all("form")),
        len(soup.find_all("script")),
        len(soup.find_all("a")),
    ]

def extract_lexical_features(text):
    return [
        sum(1 for c in text if not c.isalnum() and not c.isspace()),
        sum(1 for c in text if c.isdigit()),
        sum(1 for c in text if c.isupper()),
    ]

def extract_host_features(text):
    try:
        domain = urlparse(text).netloc
        info = whois.whois(domain)
        created = info.creation_date[0] if isinstance(info.creation_date, list) else info.creation_date
        age = (pd.Timestamp.now() - pd.to_datetime(created)).days if created else 0
    except:
        age = 0
    return [age]

# === Predict with MobileBERT ===
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

# === FastAPI Setup ===
app = FastAPI()

class EmailRequest(BaseModel):
    email: str

@app.post("/predict")
async def predict_email(data: EmailRequest):
    email_text = data.email

    # Handcrafted features
    html_feats = extract_html_features(email_text)
    lexical_feats = extract_lexical_features(email_text)
    host_feats = extract_host_features(email_text)
    handcrafted = html_feats + lexical_feats + host_feats  # total ~7 features

    # Vectorized features
    rf_vec_tfidf = rf_tfidf_vec.transform([email_text]).toarray()
    rf_vec_count = rf_count_vec.transform([email_text]).toarray()
    rf_input = np.concatenate((handcrafted, rf_vec_tfidf[0], rf_vec_count[0]))

    xgb_vec_tfidf = xgb_tfidf_vec.transform([email_text]).toarray()
    xgb_vec_count = xgb_count_vec.transform([email_text]).toarray()
    xgb_input = np.concatenate((handcrafted, xgb_vec_tfidf[0], xgb_vec_count[0]))

    # RF + XGB predictions
    rf_probs = rf_model.predict_proba([rf_input])[0]
    xgb_probs = xgb_model.predict_proba([xgb_input])[0]
    rf_conf = float(np.max(rf_probs))
    xgb_conf = float(np.max(xgb_probs))
    rf_pred = int(np.argmax(rf_probs))
    xgb_pred = int(np.argmax(xgb_probs))

    # BERT prediction
    bert_conf = predict_bert(email_text)
    bert_pred = int(bert_conf >= 0.5)

    # Final decision
    final_score = (rf_conf * rf_pred + xgb_conf * xgb_pred + bert_conf * bert_pred) / 3
    final_pred = int(final_score >= 0.5)

    return {
        "prediction": final_pred,
        "confidence": round(final_score, 4),
        "rf_conf": round(rf_conf, 4),
        "xgb_conf": round(xgb_conf, 4),
        "bert_conf": round(bert_conf, 4)
    }
