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
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification  # Tokenizer + BERT model

# ==== Import FastAPI tools ====
from fastapi import FastAPI  # Framework for building the API
from pydantic import BaseModel  # Validates incoming JSON requests

# ==== Initialize the FastAPI application ====
app = FastAPI()

# ==== Root endpoint just to verify if service is alive ====
@app.get("/")
def root():
    return {"status": "API is running"}  # Basic health check route

# ==== Define the input structure for the /predict endpoint ====
class EmailRequest(BaseModel):
    email: str  # Expecting a single email body as text

# ==== Load ML models trained earlier and saved as .pkl ====
rf_model = joblib.load("random_forest_model.pkl")  # Load Random Forest model
xgb_model = joblib.load("xgboost_model.pkl")  # Load XGBoost model

# ==== Load the vectorizers used to transform email text into numeric format ====
tfidf = joblib.load("tfidf_vectorizer.pkl")  # For RF model
count = joblib.load("count_vectorizer.pkl")  # For RF model
xgb_tfidf = joblib.load("xgb_tfidf_vectorizer.pkl")  # For XGB model
xgb_count = joblib.load("xgb_count_vectorizer.pkl")  # For XGB model

# ==== Load the fine-tuned MobileBERT model and its tokenizer ====
bert_path = "./mobilebert_combined_model"  # Folder containing BERT model files
tokenizer = MobileBertTokenizer.from_pretrained(bert_path)  # Load tokenizer
bert_model = MobileBertForSequenceClassification.from_pretrained(bert_path)  # Load model
bert_model.eval()  # Set model to evaluation mode (important for inference)

# ==== Extract HTML-based features like number of links, forms, and scripts ====
def extract_html_features(text):
    soup = BeautifulSoup(text, "html.parser")
    return {
        "num_forms": len(soup.find_all("form")),
        "num_scripts": len(soup.find_all("script")),
        "num_links": len(soup.find_all("a")),
    }

# ==== Extract lexical features like punctuation count, uppercase, and digits ====
def extract_lexical_features(text):
    return {
        "num_special_chars": sum(1 for c in text if not c.isalnum() and not c.isspace()),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "num_uppercase": sum(1 for c in text if c.isupper()),
    }

# ==== Extract WHOIS-based feature: domain age (in days) ====
def extract_host_features(text):
    try:
        domain = urlparse(text).netloc  # Extract domain
        info = whois.whois(domain)  # Query WHOIS for domain creation date
        created = info.creation_date[0] if isinstance(info.creation_date, list) else info.creation_date
        age = (pd.Timestamp.now() - pd.to_datetime(created)).days if created else 0  # Age in days
    except Exception as e:
        print(f"WHOIS lookup failed: {e}")
        age = 0
    return {"domain_age": age}

# ==== Main prediction endpoint ====
@app.post("/predict")
async def predict_email(data: EmailRequest):
    # Step 1: Clean email input text
    email_text = data.email.strip()

    # Step 2: Tokenize the email text for MobileBERT
    tokens = tokenizer(email_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    # Step 3: Get prediction and confidence from MobileBERT
    with torch.no_grad():
        output = bert_model(**tokens)
    probs_bert = torch.softmax(output.logits, dim=1).squeeze().numpy()  # Convert logits to probabilities
    bert_pred = int(np.argmax(probs_bert))  # Predicted class (0 or 1)
    bert_conf = float(np.max(probs_bert))  # Confidence score of predicted class

    # Step 4: Extract handcrafted features for RF/XGB
    html_feats = extract_html_features(email_text)
    lexical_feats = extract_lexical_features(email_text)
    host_feats = extract_host_features(email_text)

    # Step 5: Convert email text to vectorized features for each model
    tfidf_vec = tfidf.transform([email_text]).toarray()
    count_vec = count.transform([email_text]).toarray()
    xgb_tfidf_vec = xgb_tfidf.transform([email_text]).toarray()
    xgb_count_vec = xgb_count.transform([email_text]).toarray()

    # Step 6: Combine handcrafted + vectorized features for Random Forest
    rf_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        tfidf_vec[0],
        count_vec[0]
    ])

    # Step 7: Combine handcrafted + vectorized features for XGBoost
    xgb_features = np.concatenate([
        list(html_feats.values()),
        list(lexical_feats.values()),
        list(host_feats.values()),
        xgb_tfidf_vec[0],
        xgb_count_vec[0]
    ])

    # Step 8: Get prediction and confidence from RF
    rf_probs = rf_model.predict_proba(pd.DataFrame([rf_features]))[0]
    rf_pred = int(np.argmax(rf_probs))
    rf_conf = float(np.max(rf_probs))

    # Step 9: Get prediction and confidence from XGB
    xgb_probs = xgb_model.predict_proba(pd.DataFrame([xgb_features]))[0]
    xgb_pred = int(np.argmax(xgb_probs))
    xgb_conf = float(np.max(xgb_probs))

    # Step 10: Compute final score using weighted ensemble voting
    final_score = (
        bert_conf * 0.5 * bert_pred +  # BERT contributes 50%
        rf_conf * 0.25 * rf_pred +    # RF contributes 25%
        xgb_conf * 0.25 * xgb_pred    # XGB contributes 25%
    )
    final_pred = 1 if final_score >= 0.5 else 0  # Predict phishing if score ≥ 0.5

    # Optional: Log result to server console
    print(f"[PREDICTION] Final: {final_pred} | Score: {final_score:.4f} | BERT: {bert_conf:.4f}, RF: {rf_conf:.4f}, XGB: {xgb_conf:.4f}")

    # Step 11: Return only relevant values to the app
    return {
        "final_prediction": "phishing" if final_pred == 1 else "safe",
        "final_score": round(final_score, 4),
        "bert_confidence": round(bert_conf, 4),
        "rf_confidence": round(rf_conf, 4),
        "xgb_confidence": round(xgb_conf, 4)
    }