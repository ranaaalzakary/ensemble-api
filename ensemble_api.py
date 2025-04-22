import warnings
# Suppress user warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)

# === Import necessary libraries ===
import joblib  # For loading trained ML models and vectorizers
import torch  # For using PyTorch-based MobileBERT
import numpy as np  # For array operations
import pandas as pd  # For data handling
from bs4 import BeautifulSoup  # For HTML parsing
from urllib.parse import urlparse  # For extracting domains
import whois  # For getting domain creation dates
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification  # For BERT
from fastapi import FastAPI  # For creating the API
from pydantic import BaseModel  # For request body validation

# === Initialize FastAPI app ===
app = FastAPI()

# Define the expected request structure
class EmailRequest(BaseModel):
    email: str

# === Load pre-trained models ===
rf_model = joblib.load("random_forest_model.pkl")  # Random Forest
xgb_model = joblib.load("xgboost_model.pkl")  # XGBoost

# === Load vectorizers for feature transformation ===
tfidf = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF for RF
count = joblib.load("count_vectorizer.pkl")  # Count for RF
xgb_tfidf = joblib.load("xgb_tfidf_vectorizer.pkl")  # TF-IDF for XGB
xgb_count = joblib.load("xgb_count_vectorizer.pkl")  # Count for XGB

# === Load fine-tuned MobileBERT model ===
bert_path = "./"
tokenizer = MobileBertTokenizer.from_pretrained(bert_path)
bert_model = MobileBertForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()  # Set BERT to evaluation mode

# === Feature extraction functions ===
def extract_html_features(text):
    # Extract number of forms, scripts, and links from HTML
    soup = BeautifulSoup(text, "html.parser")
    return {
        "num_forms": len(soup.find_all("form")),
        "num_scripts": len(soup.find_all("script")),
        "num_links": len(soup.find_all("a")),
    }

def extract_lexical_features(text):
    # Extract lexical features like number of special characters, digits, and uppercase letters
    return {
        "num_special_chars": sum(1 for c in text if not c.isalnum() and not c.isspace()),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "num_uppercase": sum(1 for c in text if c.isupper()),
    }

def extract_host_features(text):
    # Extract domain age using WHOIS data
    try:
        domain = urlparse(text).netloc
        info = whois.whois(domain)
        created = info.creation_date[0] if isinstance(info.creation_date, list) else info.creation_date
        age = (pd.Timestamp.now() - pd.to_datetime(created)).days if created else 0
    except:
        age = 0
    return {"domain_age": age}

# === Prediction endpoint ===
@app.post("/predict")
async def predict_email(data: EmailRequest):
    # Clean and prepare the input
    email_text = data.email.strip()

    # === MobileBERT prediction ===
    tokens = tokenizer(email_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        output = bert_model(**tokens)
    probs_bert = torch.softmax(output.logits, dim=1).squeeze().numpy()
    bert_pred = int(np.argmax(probs_bert))
    bert_conf = float(np.max(probs_bert))

    # === Extract handcrafted features ===
    html_feats = extract_html_features(email_text)
    lexical_feats = extract_lexical_features(email_text)
    host_feats = extract_host_features(email_text)

    # === Vectorize the input text ===
    tfidf_vec = tfidf.transform([email_text]).toarray()
    count_vec = count.transform([email_text]).toarray()
    xgb_tfidf_vec = xgb_tfidf.transform([email_text]).toarray()
    xgb_count_vec = xgb_count.transform([email_text]).toarray()

    # === Combine features for RF and XGB ===
    rf_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        tfidf_vec[0],
        count_vec[0]
    ])
    xgb_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        xgb_tfidf_vec[0],
        xgb_count_vec[0]
    ])

    # === Random Forest prediction ===
    rf_probs = rf_model.predict_proba(pd.DataFrame([rf_features]))[0]
    rf_pred = int(np.argmax(rf_probs))
    rf_conf = float(np.max(rf_probs))

    # === XGBoost prediction ===
    xgb_probs = xgb_model.predict_proba(pd.DataFrame([xgb_features]))[0]
    xgb_pred = int(np.argmax(xgb_probs))
    xgb_conf = float(np.max(xgb_probs))

    # === Ensemble weighted voting ===
    final_score = (
        bert_conf * 0.5 * bert_pred +
        rf_conf * 0.25 * rf_pred +
        xgb_conf * 0.25 * xgb_pred
    )
    final_pred = 1 if final_score >= 0.5 else 0

    # === Return final results ===
    return {
        "final_prediction": "phishing" if final_pred == 1 else "safe",
        "final_score": round(final_score, 4),
        "bert_confidence": round(bert_conf, 4),
        "rf_confidence": round(rf_conf, 4),
        "xgb_confidence": round(xgb_conf, 4)
    }