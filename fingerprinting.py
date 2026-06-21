import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
import librosa
import os
import pickle

# Fingerprinting parameters
FS = 22050
NPERSEG = 1024
NOVERLAP = 512
PEAK_NEIGHBORHOOD_SIZE = 20
MIN_AMPLITUDE_DB_DROP = -50  # Keep peaks within 50dB of max
FAN_VALUE = 15

def get_spectrogram(y, fs=FS):
    """Computes the spectrogram of an audio signal."""
    f, t, Sxx = signal.spectrogram(y, fs, nperseg=NPERSEG, noverlap=NOVERLAP)
    # Log scale and normalize max to 0 dB
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    Sxx_log = Sxx_log - np.max(Sxx_log)
    return f, t, Sxx_log

def get_constellation_peaks(Sxx_log):
    """Finds local maxima (peaks) in the spectrogram."""
    # Create a local maximum filter
    local_max = maximum_filter(Sxx_log, size=PEAK_NEIGHBORHOOD_SIZE) == Sxx_log
    
    # Combine conditions: must be a local max and above min amplitude relative to max
    peaks = local_max & (Sxx_log > MIN_AMPLITUDE_DB_DROP)
    
    # Get coordinates of the peaks (frequency_idx, time_idx)
    freq_indices, time_indices = np.where(peaks)
    
    # Return as list of (time_idx, freq_idx) sorted by time
    constellation = list(zip(time_indices, freq_indices))
    constellation.sort(key=lambda x: x[0])
    
    return constellation

def generate_hashes(constellation):
    """Generates hashes from the constellation peaks."""
    hashes = []
    # For each peak, pair it with subsequent peaks (up to FAN_VALUE)
    for i in range(len(constellation)):
        for j in range(1, FAN_VALUE):
            if (i + j) < len(constellation):
                t1, f1 = constellation[i]
                t2, f2 = constellation[i + j]
                
                t_delta = t2 - t1
                
                # We only want forward pairs (t_delta > 0) to avoid duplicates/mess
                if t_delta > 0 and t_delta < 200:
                    # Hash is a tuple of (f1, f2, t_delta)
                    hash_val = f"{f1}_{f2}_{t_delta}"
                    # Store (hash_val, absolute_time_of_t1)
                    hashes.append((hash_val, t1))
    return hashes

def build_database(songs_dir):
    """Builds a database of hashes from a directory of audio files."""
    database = {} # hash_val -> [(song_name, offset_time), ...]
    
    if not os.path.exists(songs_dir):
        print(f"Directory {songs_dir} not found.")
        return database
        
    for filename in os.listdir(songs_dir):
        if filename.endswith(('.wav', '.mp3', '.ogg')):
            filepath = os.path.join(songs_dir, filename)
            song_name = os.path.splitext(filename)[0]
            
            try:
                # Load audio
                y, sr = librosa.load(filepath, sr=FS, mono=True)
                
                # Get constellation
                f, t, Sxx_log = get_spectrogram(y, sr)
                peaks = get_constellation_peaks(Sxx_log)
                
                # Generate hashes
                hashes = generate_hashes(peaks)
                
                # Store in database
                for hash_val, t1 in hashes:
                    if hash_val not in database:
                        database[hash_val] = []
                    database[hash_val].append((song_name, t1))
                    
                print(f"Indexed {song_name} with {len(hashes)} hashes.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return database

def save_database(database, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(database, f)

def load_database(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return {}

def match_query(query_filepath, database):
    """Matches a query audio clip against the database."""
    try:
        y, sr = librosa.load(query_filepath, sr=FS, mono=True)
        f, t, Sxx_log = get_spectrogram(y, sr)
        peaks = get_constellation_peaks(Sxx_log)
        hashes = generate_hashes(peaks)
    except Exception as e:
        print(f"Error processing query {query_filepath}: {e}")
        return None, {}, None, None, None
        
    # Find matches
    matches_per_song = {} # song_name -> list of relative time offsets (t_song - t_query)
    
    for hash_val, t_query in hashes:
        if hash_val in database:
            for song_name, t_song in database[hash_val]:
                offset = t_song - t_query
                
                if song_name not in matches_per_song:
                    matches_per_song[song_name] = []
                matches_per_song[song_name].append(offset)
                
    # Score songs based on the largest peak in the offset histogram
    scores = {} # song_name -> max_count_at_single_offset
    offset_histograms = {}
    
    for song_name, offsets in matches_per_song.items():
        if len(offsets) == 0:
            continue
            
        # Count occurrences of each offset
        unique_offsets, counts = np.unique(offsets, return_counts=True)
        max_count_idx = np.argmax(counts)
        scores[song_name] = counts[max_count_idx]
        offset_histograms[song_name] = (unique_offsets, counts)
        
    if not scores:
        return "No match found", {}, Sxx_log, peaks, None
        
    # Best match
    best_song = max(scores.items(), key=lambda x: x[1])[0]
    
    return best_song, scores, Sxx_log, peaks, offset_histograms.get(best_song)
