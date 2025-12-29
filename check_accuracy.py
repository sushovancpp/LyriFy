import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df = pd.read_csv("data/Spotify_Song_Dataset.csv")
df = df.dropna(subset=["text"])
df["clean_text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2
)
X = vectorizer.fit_transform(df["clean_text"])

def predict_top_k(snippet, k=10):
    snippet_vec = vectorizer.transform([clean_text(snippet)])
    scores = cosine_similarity(snippet_vec, X)[0]
    return scores.argsort()[-k:][::-1]

def check_accuracy(samples=200):
    top10_correct = 0
    sampled = df.sample(samples, random_state=42)

    for _, row in sampled.iterrows():
        lyrics = row["text"]
        snippet = lyrics[100:200]  # IMPORTANT FIX

        true_song = row["song"]
        indices = predict_top_k(snippet, 10)
        predicted = df.iloc[indices]["song"].values

        if true_song in predicted:
            top10_correct += 1

    return top10_correct / samples

acc = check_accuracy(200)
print(f"✅ Top-10 Accuracy : {acc:.2%}")
