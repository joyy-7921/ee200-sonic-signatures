import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from fingerprinting import build_database, save_database, load_database, match_query

st.set_page_config(page_title="Sonic Signatures", layout="wide", page_icon="🎵")

# Custom CSS for Premium Glassmorphism UI
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    
    /* Hide top header bar */
    header {visibility: hidden;}

    /* Glassmorphism for containers and sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
        background: rgba(139, 92, 246, 0.05);
    }

    /* Vibrant Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #8b5cf6 0%, #d946ef 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(217, 70, 239, 0.6) !important;
    }

    /* Titles and headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Success/Error Messages */
    div[data-testid="stCodeBlock"] {
        background: rgba(0,0,0,0.4) !important;
    }
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34d399 !important;
    }
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #f87171 !important;
    }
</style>
""", unsafe_allow_html=True)


DB_FILE = "song_db.pkl"
SONGS_DIR = "songs"

@st.cache_resource
def get_db():
    if os.path.exists(DB_FILE):
        return load_database(DB_FILE)
    else:
        st.warning("Database not found. Please build the database first.")
        return {}

st.title("🎵 Sonic Signatures - Audio Fingerprinting App")

db = get_db()
st.sidebar.write(f"Database contains hashes for {len(set([item[0] for items in db.values() for item in items])) if db else 0} songs.")

if st.sidebar.button("Rebuild Database"):
    with st.spinner("Building database... This may take a couple minutes."):
        new_db = build_database(SONGS_DIR)
        save_database(new_db, DB_FILE)
        get_db.clear()  # Clear the cache so it reloads the newly built database
        st.success("Database built successfully! Please refresh or click the button again.")
        st.rerun() # Automatically refresh the page

mode = st.sidebar.radio("Select Mode", ["Single-Clip Mode", "Batch Mode"])

if mode == "Single-Clip Mode":
    st.header("Identify a Single Clip")
    uploaded_file = st.file_uploader("Upload an audio clip (WAV/MP3)", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        if not db:
            st.error("Database is empty. Please rebuild it from the sidebar first.")
        else:
            if st.button("Identify Song"):
                # Save temp file
                temp_path = os.path.join("temp_query" + os.path.splitext(uploaded_file.name)[1])
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Analyzing..."):
                    best_song, scores, Sxx_log, peaks, offset_hist = match_query(temp_path, db)
                    
                st.subheader("Prediction")
                if best_song == "No match found":
                    st.error(best_song)
                else:
                    st.success(f"Matched Song: **{best_song}**")
                    
                st.subheader("Intermediate Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Spectrogram & Constellation**")
                    if Sxx_log is not None and peaks is not None:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(Sxx_log, aspect='auto', origin='lower', cmap='viridis')
                        
                        # Plot peaks
                        time_indices, freq_indices = zip(*peaks)
                        ax.scatter(time_indices, freq_indices, c='red', s=10, marker='x')
                        
                        ax.set_ylabel('Frequency Index')
                        ax.set_xlabel('Time Index')
                        st.pyplot(fig)
                        
                with col2:
                    st.write("**Offset Histogram**")
                    if offset_hist is not None:
                        unique_offsets, counts = offset_hist
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Use vlines/scatter instead of bar for better visibility across huge ranges
                        ax.vlines(unique_offsets, 0, counts, color='blue', alpha=0.5, linewidth=2)
                        ax.scatter(unique_offsets, counts, color='blue', s=10)
                        
                        # Highlight the maximum peak
                        max_idx = np.argmax(counts)
                        ax.scatter(unique_offsets[max_idx], counts[max_idx], color='red', s=50, label='True Offset Match')
                        
                        ax.set_xlabel('Time Offset (t_song - t_query)')
                        ax.set_ylabel('Number of Matching Hashes')
                        ax.set_title(f'Histogram for {best_song}')
                        ax.legend()
                        st.pyplot(fig)
                        
                if os.path.exists(temp_path):
                    os.remove(temp_path)

elif mode == "Batch Mode":
    st.header("Batch Process Queries")
    uploaded_files = st.file_uploader("Upload multiple query clips", type=['wav', 'mp3', 'ogg'], accept_multiple_files=True)
    
    if uploaded_files and db:
        if st.button("Process Batch"):
            results = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                temp_path = os.path.join("temp_batch_" + uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                best_song, _, _, _, _ = match_query(temp_path, db)
                
                prediction = best_song if best_song != "No match found" else "UNKNOWN"
                results.append({"filename": uploaded_file.name, "prediction": prediction})
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Batch processing complete!")
            
            # Write to results.csv
            df = pd.DataFrame(results)
            csv_path = "results.csv"
            df.to_csv(csv_path, index=False)
            
            st.success(f"Results saved to {csv_path}")
            st.dataframe(df)
            
            # Provide download link
            with open(csv_path, "rb") as f:
                st.download_button("Download results.csv", f, file_name="results.csv", mime="text/csv")
