import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# IMPROVED Text Cleaning
# -----------------------------
def clean_text(text):
    """Minimal cleaning to preserve lyric semantics"""
    text = str(text).lower()
    # Keep apostrophes and basic punctuation for context
    text = re.sub(r'[^a-z\s\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Load and Prepare Dataset
# -----------------------------
try:
    df = pd.read_csv("data/Spotify_Song_Dataset.csv")
    print(f"✅ Loaded {len(df)} songs")
except FileNotFoundError:
    print("❌ Dataset not found! Place Spotify_Song_Dataset.csv in data/ folder.")
    exit()

# Remove rows with missing lyrics
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 50]  # Remove very short lyrics
print(f"✅ After cleaning: {len(df)} songs with valid lyrics")

# Clean lyrics (minimal preprocessing)
df['clean_text'] = df['text'].apply(clean_text)

# Create unique song identifier
df['song_id'] = df['song'] + " - " + df['artist']

# -----------------------------
# Train/Test Split for Evaluation
# -----------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"✅ Training set: {len(train_df)} songs")
print(f"✅ Test set: {len(test_df)} songs")

# -----------------------------
# Improved Vectorization
# -----------------------------
# Use character n-grams to capture phrases and word order
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),  # Capture single words, bigrams, trigrams
    analyzer='word',
    min_df=2,  # Ignore very rare terms
    max_df=0.8  # Ignore very common terms
)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
print(f"✅ Vectorized with {X_train.shape[1]} features")

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(test_df, X_test, train_df, X_train, k=5):
    """Evaluate top-k accuracy on test set"""
    correct_at_k = {1: 0, 3: 0, 5: 0}
    
    for idx, row in test_df.iterrows():
        # Take random snippet from test song (simulate user input)
        lyrics = row['clean_text']
        words = lyrics.split()
        if len(words) < 20:
            continue
            
        # Extract random 10-15 word snippet
        start = np.random.randint(0, max(1, len(words) - 15))
        snippet = " ".join(words[start:start + 15])
        
        # Find matches
        snippet_vector = vectorizer.transform([snippet])
        similarities = cosine_similarity(snippet_vector, X_train)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Check if correct song is in top-k
        true_song_id = row['song_id']
        for rank, idx in enumerate(top_k_indices, 1):
            pred_song_id = train_df.iloc[idx]['song_id']
            if pred_song_id == true_song_id:
                if rank <= 1:
                    correct_at_k[1] += 1
                if rank <= 3:
                    correct_at_k[3] += 1
                if rank <= 5:
                    correct_at_k[5] += 1
                break
    
    total = len(test_df)
    print("\n📊 Model Accuracy on Test Set:")
    print(f"   Top-1 Accuracy: {correct_at_k[1]/total*100:.2f}%")
    print(f"   Top-3 Accuracy: {correct_at_k[3]/total*100:.2f}%")
    print(f"   Top-5 Accuracy: {correct_at_k[5]/total*100:.2f}%")

# Run evaluation
print("\n⏳ Evaluating model (this may take a minute)...")
evaluate_model(test_df, X_test, train_df, X_train, k=5)

# -----------------------------
# Improved Prediction Function
# -----------------------------
def predict_song(snippet, top_n=5):
    """Return top N matches with confidence scores"""
    snippet_clean = clean_text(snippet)
    
    if len(snippet_clean.split()) < 3:
        return None, "Snippet too short (need at least 3 words)"
    
    snippet_vector = vectorizer.transform([snippet_clean])
    similarity_scores = cosine_similarity(snippet_vector, X_train)[0]
    
    # Get top N matches
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        song = train_df.iloc[idx]['song']
        artist = train_df.iloc[idx]['artist']
        score = similarity_scores[idx]
        results.append({
            'song': song,
            'artist': artist,
            'confidence': score
        })
    
    return results, None

# -----------------------------
# Main Program (Interactive Mode)
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎵 IMPROVED Spotify Lyric Search 🎵")
    print("="*50)
    print("Enter a lyrics snippet to identify the song")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("🎶 Enter lyrics: ")
        
        if user_input.lower() == "exit":
            print("\n👋 Goodbye!")
            break
        
        if len(user_input.strip()) < 10:
            print("❌ Please enter at least 10 characters.\n")
            continue
        
        results, error = predict_song(user_input, top_n=5)
        
        if error:
            print(f"❌ {error}\n")
            continue
        
        print("\n" + "="*50)
        print("✅ TOP 5 PREDICTIONS:")
        print("="*50)
        
        for i, result in enumerate(results, 1):
            confidence_pct = result['confidence'] * 100
            bars = "█" * int(confidence_pct / 5)
            
            print(f"\n{i}. 🎶 {result['song']}")
            print(f"   🎤 {result['artist']}")
            print(f"   📊 Confidence: [{bars:<20}] {confidence_pct:.1f}%")
        
        print("\n" + "="*50 + "\n")