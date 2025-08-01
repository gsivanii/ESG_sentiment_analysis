import os
import re
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import spacy
nlp = spacy.load("en_core_web_sm")  # Load spaCy model for NER

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time


from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# Configuration flags
# ---------------------------
USE_DUMMY_IF_SCRAPE_FAILS = True
RUN_TRANSFORMERS = False       # set True if HuggingFace sentiment (downloads model)
RANDOM_STATE = 42

# ---------------------------
# Utilities
# ---------------------------
def extract_company_ner(text: str) -> str:
    """
    Extract company using Named Entity Recognition (NER).
    Returns first ORG found or "Unknown".
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text
    return "Unknown"

def extract_company_simple(text: str) -> str:
    matches = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?\b", text)
    return matches[0] if matches else "Unknown"

# ---------------------------
# Selenium scraping logic
# ---------------------------
def selenium_scrape_headlines() -> pd.DataFrame:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    urls = {
        "Bloomberg Green": "https://www.bloomberg.com/green",
        "ESG Today": "https://www.esgtoday.com/",
        "Reuters Sustainability": "https://www.reuters.com/sustainability/",
        "Yahoo Finance ESG": "https://finance.yahoo.com/topic/esg"
    }

    headlines = set()

    for name, url in urls.items():
        try:
            driver.get(url)
            time.sleep(5)

            elems = driver.find_elements(By.TAG_NAME, "h1") + driver.find_elements(By.TAG_NAME, "h2") + driver.find_elements(By.TAG_NAME, "h3")
            for elem in elems:
                txt = elem.text.strip()
                if len(txt) >= 10:
                    headlines.add(txt)
            print(f"[INFO] Scraped {len(headlines)} headlines from {name}")
        except Exception as e:
            print(f"[ERROR] Failed to scrape {name}: {e}")

    driver.quit()
    df = pd.DataFrame(sorted(headlines), columns=["headline"])
    print(df.head())
    return df

# Example usage:
df = selenium_scrape_headlines()
df["company"] = df["headline"].apply(extract_company_ner)


############################################ End driver

print(f"[INFO] Total headlines: {len(df)}")

print("[INFO] Labeling headlines using transformer sentiment pipeline...")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
df['label'] = df['headline'].apply(lambda x: sentiment_pipeline(x)[0]['label'].lower())

# OPTIONAL: Keep only positive and negative
df = df[df['label'].isin(['positive', 'negative'])].reset_index(drop=True)

# === Step 3: Train/test split ===
X = df['headline']
y = df['label']
if len(set(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === Step 4: Vectorize ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Step 5: Train classifier ===
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# === Step 6: Evaluate ===
y_pred = clf.predict(X_test_vec)
print("\n=== Logistic Regression (transformer-based sentiment labels) ===")
print(classification_report(y_test, y_pred)) 
