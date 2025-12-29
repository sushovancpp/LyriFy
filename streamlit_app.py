import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Spotify Lyric Search",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        background-color: #1DB954;
        height: 8px;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Text Cleaning Function
# -----------------------------
@st.cache_data
def clean_text(text):
    """Minimal cleaning to preserve lyric semantics"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Load and Process Dataset
# -----------------------------
@st.cache_data
def load_data():
    """Load and preprocess the Spotify dataset"""
    try:
        df = pd.read_csv("data/Spotify_Song_Dataset.csv")
        
        # Remove rows with missing lyrics
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.len() > 50]
        
        # Clean lyrics
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Create unique identifier
        df['song_id'] = df['song'] + " - " + df['artist']
        
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please place 'Spotify_Song_Dataset.csv' in the 'data/' folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

# -----------------------------
# Build TF-IDF Model
# -----------------------------
@st.cache_resource
def build_model(df):
    """Build and cache the TF-IDF vectorizer and matrix"""
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        analyzer='word',
        min_df=2,
        max_df=0.8
    )
    
    X = vectorizer.fit_transform(df['clean_text'])
    return vectorizer, X

# -----------------------------
# Prediction Function
# -----------------------------
def predict_song(snippet, vectorizer, X, df, top_n=5):
    """Return top N matches with confidence scores"""
    snippet_clean = clean_text(snippet)
    
    if len(snippet_clean.split()) < 3:
        return None, "⚠️ Snippet too short. Please enter at least 3 words."
    
    snippet_vector = vectorizer.transform([snippet_clean])
    similarity_scores = cosine_similarity(snippet_vector, X)[0]
    
    # Get top N matches
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        song = df.iloc[idx]['song']
        artist = df.iloc[idx]['artist']
        score = similarity_scores[idx]
        original_lyrics = df.iloc[idx]['text']
        
        results.append({
            'song': song,
            'artist': artist,
            'confidence': score,
            'lyrics': original_lyrics
        })
    
    return results, None

# -----------------------------
# Visualization Functions
# -----------------------------
def create_confidence_chart(results):
    """Create a horizontal bar chart for confidence scores"""
    songs = [f"{r['song'][:30]}..." if len(r['song']) > 30 else r['song'] for r in results]
    confidences = [r['confidence'] * 100 for r in results]
    
    fig = go.Figure(go.Bar(
        x=confidences,
        y=songs,
        orientation='h',
        marker=dict(
            color=confidences,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f"{c:.1f}%" for c in confidences],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# -----------------------------
# Main App
# -----------------------------
def main():
    # Header
    st.markdown('<p class="main-header">🎵 Spotify Lyric Search</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Identify songs from lyrics snippets using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("🔄 Loading dataset and building model..."):
        df = load_data()
        vectorizer, X = build_model(df)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.metric("Total Songs", len(df))
        st.metric("Model Features", X.shape[1])
        
        st.divider()
        
        st.header("⚙️ Settings")
        top_n = st.slider("Number of predictions", min_value=1, max_value=10, value=5)
        show_lyrics = st.checkbox("Show full lyrics", value=False)
        
        st.divider()
        
        st.header("💡 Tips")
        st.info("""
        - Enter at least 10-15 words for better accuracy
        - Use unique phrases from the song
        - Check multiple results if unsure
        - Confidence >50% is usually reliable
        """)
        
        st.divider()
        
        st.header("📝 Example Snippets")
        examples = [
            "shake it off shake it off",
            "just a small town girl living in a lonely world",
            "we will we will rock you",
            "I'm on the highway to hell"
        ]
        
        for example in examples:
            if st.button(f"'{example[:30]}...'", key=example):
                st.session_state.lyrics_input = example
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎤 Enter Lyrics Snippet")
        
        # Text input
        lyrics_input = st.text_area(
            "Type or paste lyrics here:",
            value=st.session_state.get('lyrics_input', ''),
            height=150,
            placeholder="Example: 'cause the players gonna play play play play play...",
            key="lyrics_textarea"
        )
        
        # Search button
        search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
        with search_col2:
            search_button = st.button("🔍 Search Song", type="primary", use_container_width=True)
        
        # Clear button
        with search_col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.lyrics_input = ""
                st.rerun()
    
    with col2:
        st.subheader("📈 Quick Stats")
        if 'last_search_time' in st.session_state:
            st.metric("Last Search Time", f"{st.session_state.last_search_time:.3f}s")
        if 'last_confidence' in st.session_state:
            st.metric("Top Match Confidence", f"{st.session_state.last_confidence:.1f}%")
    
    # Process search
    if search_button and lyrics_input.strip():
        start_time = time.time()
        
        with st.spinner("🎵 Searching for matching songs..."):
            results, error = predict_song(lyrics_input, vectorizer, X, df, top_n=top_n)
        
        search_time = time.time() - start_time
        st.session_state.last_search_time = search_time
        
        if error:
            st.error(error)
        elif results:
            st.session_state.last_confidence = results[0]['confidence'] * 100
            
            # Display results
            st.divider()
            st.subheader("🎯 Search Results")
            
            # Show confidence chart
            st.plotly_chart(create_confidence_chart(results), use_container_width=True)
            
            st.divider()
            
            # Display each result
            for i, result in enumerate(results, 1):
                confidence_pct = result['confidence'] * 100
                
                # Color code based on confidence
                if confidence_pct >= 70:
                    emoji = "🟢"
                    color = "#1DB954"
                elif confidence_pct >= 40:
                    emoji = "🟡"
                    color = "#FFA500"
                else:
                    emoji = "🔴"
                    color = "#FF4444"
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {emoji} {i}. {result['song']}")
                        st.markdown(f"**Artist:** {result['artist']}")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence_pct:.1f}%")
                    
                    # Progress bar
                    st.progress(confidence_pct / 100)
                    
                    # Show lyrics preview if enabled
                    if show_lyrics:
                        with st.expander("📄 View Full Lyrics"):
                            st.text(result['lyrics'][:500] + "..." if len(result['lyrics']) > 500 else result['lyrics'])
                    
                    st.divider()
            
            # Success message
            st.success(f"✅ Found {len(results)} matching songs in {search_time:.3f} seconds!")
    
    elif search_button:
        st.warning("⚠️ Please enter some lyrics to search!")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Built with Streamlit 🎈 | Powered by TF-IDF & Scikit-learn</p>
            <p>Dataset: Spotify 50k+ Songs</p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()