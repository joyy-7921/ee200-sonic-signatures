import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from fingerprinting import build_database, save_database, load_database, match_query

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", page_icon="🎵")

# Custom CSS for Cyan Theme matching Demo Video
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
    }
    
    /* Header styling */
    .st-emotion-cache-10trblm {
        color: #06b6d4; /* Cyan Title */
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #06b6d4 !important;
        color: #06b6d4 !important;
    }
    
    /* Cyan Buttons */
    .stButton > button {
        background-color: #06b6d4 !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #22d3ee !important;
        transform: translateY(-1px);
    }
    
    /* Hide default sidebar toggle if empty */
    [data-testid="collapsedControl"] { display: none; }
</style>
''', unsafe_allow_html=True)

DB_FILE = "song_db.pkl"
SONGS_DIR = "songs"
QUERIES_DIR = "queries"

@st.cache_resource
def get_db():
    if os.path.exists(DB_FILE):
        return load_database(DB_FILE)
    return {}

db = get_db()

# Main Header matches demo
st.title("EE200: Audio Fingerprinting")
st.markdown("<p style='color: #94a3b8; font-weight: 600; letter-spacing: 1px; font-size: 0.9rem;'>SIGNALS, SYSTEMS & NETWORKS</p>", unsafe_allow_html=True)

tab_lib, tab_id, tab_batch = st.tabs(["♦ LIBRARY", "☉ IDENTIFY", "☷ BATCH"])

with tab_lib:
    st.subheader("In the Database")
    st.info("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
            
    if db:
        song_names = set(name for items in db.values() for name, _ in items)
        st.metric("Total Indexed Songs", len(song_names))
        
        cols = st.columns(3)
        for i, s_name in enumerate(sorted(list(song_names))):
            with cols[i % 3]:
                st.markdown(f'''
                <div style="background: #1e293b; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #06b6d4;">
                    <h4 style="margin:0; color:#e2e8f0;">{s_name}</h4>
                    <p style="margin:0; font-size:0.8rem; color:#94a3b8;">Indexed and hashed</p>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.warning("Database empty.")

with tab_id:
    st.markdown("### Upload a clip to identify")
    uploaded_file = st.file_uploader("Upload a clip (WAV/MP3/FLAC/OGG)", type=['wav', 'mp3', 'ogg', 'flac'])
    
    st.markdown("#### Or try a sample clip:")
    sample_clips = []
    if os.path.exists(QUERIES_DIR):
        sample_clips = sorted([f for f in os.listdir(QUERIES_DIR) if f.endswith('.wav')])[:5]
    
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    if sample_clips:
        for clip in sample_clips:
            st.markdown(f"**{clip}**")
            c1, c2 = st.columns([4, 1])
            with c1:
                st.audio(os.path.join(QUERIES_DIR, clip))
            with c2:
                if st.button("Try", key=f"btn_{clip}", use_container_width=True):
                    st.session_state.selected_sample = os.path.join(QUERIES_DIR, clip)
    else:
        st.info("Sample clips are not available in the cloud deployment to save space. Please upload your own clip.")
                
    target_path = None
    if uploaded_file is not None:
        target_path = "temp_query" + os.path.splitext(uploaded_file.name)[1]
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.selected_sample = None  # Clear sample if file uploaded
    elif st.session_state.selected_sample is not None:
        target_path = st.session_state.selected_sample
        st.info(f"Using sample clip: **{os.path.basename(target_path)}**")
        
    if target_path and db:
        if st.button("Identify Song", type="primary"):
            with st.spinner("Analyzing audio features..."):
                best_song, scores, Sxx_log, peaks, offset_hist = match_query(target_path, db)
                
            st.markdown("---")
            if best_song == "No match found":
                st.error("No match found in the database.")
            else:
                top_score = scores[best_song]
                st.success(f"### Matched: {best_song} \n**Candidate Score:** {top_score} hashes matched")
                
                st.markdown("### Feature Extraction & Matching")
                
                st.write("**Top Candidates Score Distribution**")
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                if sorted_scores:
                    s_names, s_vals = zip(*sorted_scores)
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
                    ax_bar.barh(s_names[::-1], s_vals[::-1], color='#06b6d4', alpha=0.8)
                    ax_bar.set_xlabel('Matching Hashes')
                    fig_bar.patch.set_facecolor('#0e1117')
                    ax_bar.set_facecolor('#0e1117')
                    ax_bar.tick_params(colors='white')
                    ax_bar.xaxis.label.set_color('white')
                    ax_bar.yaxis.label.set_color('white')
                    for spine in ax_bar.spines.values():
                        spine.set_edgecolor('#334155')
                    st.pyplot(fig_bar)

                c1, c2 = st.columns(2)
                
                with c1:
                    st.write("**Query Spectrogram & Constellation**")
                    if Sxx_log is not None and peaks is not None:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.imshow(Sxx_log, aspect='auto', origin='lower', cmap='magma')
                        t_idx, f_idx = zip(*peaks)
                        ax.scatter(t_idx, f_idx, c='#06b6d4', s=15, marker='o') # Cyan dots
                        ax.set_ylabel('Frequency')
                        ax.set_xlabel('Time')
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        st.pyplot(fig)
                        
                with c2:
                    st.write("**Offset Agreement Scatter (Reconstruction)**")
                    if offset_hist is not None:
                        unique_offsets, counts = offset_hist
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.scatter(unique_offsets, counts, color='#06b6d4', alpha=0.6, s=20)
                        max_idx = np.argmax(counts)
                        ax.scatter(unique_offsets[max_idx], counts[max_idx], color='#ef4444', s=80, label='Alignment Spike')
                        ax.set_xlabel('Time Offset')
                        ax.set_ylabel('Matches')
                        ax.legend()
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        st.pyplot(fig)
                        
            if uploaded_file is not None and os.path.exists(target_path):
                os.remove(target_path)

with tab_batch:
    st.markdown("### Batch Process Queries")
    batch_files = st.file_uploader("Upload multiple query clips", type=['wav', 'mp3', 'ogg'], accept_multiple_files=True)
    
    if batch_files and db:
        if st.button("Run batch", type="primary"):
            results = []
            pbar = st.progress(0)
            
            for i, uf in enumerate(batch_files):
                tp = "temp_batch_" + uf.name
                with open(tp, "wb") as f:
                    f.write(uf.getbuffer())
                b_song, _, _, _, _ = match_query(tp, db)
                pred = b_song if b_song != "No match found" else "UNKNOWN"
                results.append({"filename": uf.name, "prediction": pred})
                os.remove(tp)
                pbar.progress((i + 1) / len(batch_files))
                
            df = pd.DataFrame(results)
            df.to_csv("results.csv", index=False)
            
            # Styled dataframe
            if hasattr(df.style, "map"):
                st.dataframe(df.style.map(lambda x: "color: #06b6d4;" if x != "UNKNOWN" else "", subset=['prediction']))
            else:
                st.dataframe(df.style.applymap(lambda x: "color: #06b6d4;" if x != "UNKNOWN" else "", subset=['prediction']))
            
            with open("results.csv", "rb") as f:
                st.download_button("Download results.csv", f, file_name="results.csv", mime="text/csv")
