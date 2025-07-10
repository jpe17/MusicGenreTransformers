from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
# Remove SocketIO for now to avoid compatibility issues
# from flask_socketio import SocketIO, emit
import torch
import librosa
import librosa.display  # Add this for spectrogram plotting
import numpy as np
from model import WhisperAudioClassifier
from dataloader import MusicGenreDataset
import torchvision.transforms.v2 as transforms
import whisper
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from scipy import signal
import tempfile
import os
import wave
import json
from datetime import datetime
import threading
import time
from pydub import AudioSegment
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'musicai_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Global variables for model and processing
model = None
class_names = None
device = None

def load_model():
    """Load the trained model and setup"""
    global model, class_names, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on device: {device}")
    
    # Load class names
    temp_dataset = MusicGenreDataset(split='train')
    class_names = temp_dataset.class_names
    num_classes = temp_dataset.num_classes
    del temp_dataset
    
    # Load model
    model = WhisperAudioClassifier(num_classes=num_classes, device=device)
    try:
        model.load_state_dict(torch.load('music_genre_classifier.pth', map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Warning: Model file not found. Please train the model first.")
        return False
    return True

def analyze_mood_energy(audio_data, sr):
    """Analyze mood and energy from audio features"""
    # Extract audio features for mood analysis
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
    
    # Rhythm features
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    tempo = float(tempo) if hasattr(tempo, 'item') else float(tempo)
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    
    # Energy analysis
    energy = np.sum(np.square(audio_data))
    rms_energy = librosa.feature.rms(y=audio_data)[0]
    
    # Simple heuristic mood classification
    avg_spectral_centroid = float(np.mean(spectral_centroids))
    avg_zero_crossing = float(np.mean(zero_crossing_rate))
    avg_rms = float(np.mean(rms_energy))
    
    # Classify mood based on features
    if avg_spectral_centroid > 2000 and tempo > 120 and avg_rms > 0.02:
        mood = "Energetic"
        energy_level = min(100, (avg_rms * 1000 + tempo/2))
    elif avg_spectral_centroid < 1500 and tempo < 90:
        mood = "Calm"
        energy_level = max(10, avg_rms * 500)
    elif tempo > 130 and avg_zero_crossing > 0.1:
        mood = "Happy"
        energy_level = min(100, tempo * 0.7)
    elif avg_spectral_centroid < 1200 and avg_rms < 0.015:
        mood = "Melancholic"
        energy_level = max(5, avg_rms * 300)
    else:
        mood = "Neutral"
        energy_level = avg_rms * 700
    
    return {
        'mood': mood,
        'energy_level': float(min(100, max(0, energy_level))),
        'tempo': tempo,
        'spectral_centroid': avg_spectral_centroid,
        'zero_crossing_rate': avg_zero_crossing
    }

def detect_key_scale(audio_data, sr):
    """Detect musical key and scale"""
    # Use chromagram for key detection
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Key mapping
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_index = np.argmax(chroma_mean)
    detected_key = keys[key_index]
    
    # Simple major/minor detection based on chroma pattern
    major_pattern = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # Major scale pattern
    minor_pattern = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # Minor scale pattern
    
    # Roll the patterns to match detected key
    rolled_major = np.roll(major_pattern, key_index)
    rolled_minor = np.roll(minor_pattern, key_index)
    
    # Calculate correlation with major and minor patterns
    major_corr = np.corrcoef(chroma_mean, rolled_major)[0, 1]
    minor_corr = np.corrcoef(chroma_mean, rolled_minor)[0, 1]
    
    # Handle NaN values
    if np.isnan(major_corr):
        major_corr = 0.0
    if np.isnan(minor_corr):
        minor_corr = 0.0
    
    scale = "Major" if major_corr > minor_corr else "Minor"
    confidence = float(max(major_corr, minor_corr)) if max(major_corr, minor_corr) > 0 else 0.5
    
    return {
        'key': detected_key,
        'scale': scale,
        'confidence': confidence
    }

def assess_audio_quality(audio_data, sr):
    """Assess audio quality metrics"""
    # Signal-to-noise ratio estimation
    noise_floor = float(np.percentile(np.abs(audio_data), 10))
    signal_peak = float(np.max(np.abs(audio_data)))
    snr_estimate = float(20 * np.log10(signal_peak / (noise_floor + 1e-10)))
    
    # Dynamic range
    rms = float(np.sqrt(np.mean(audio_data**2)))
    peak = float(np.max(np.abs(audio_data)))
    dynamic_range = float(20 * np.log10(peak / (rms + 1e-10)))
    
    # Frequency response analysis
    freqs, psd = signal.welch(audio_data, sr, nperseg=1024)
    
    # Quality score (0-100)
    quality_score = float(min(100, max(0, (snr_estimate + 30) * 1.5)))
    
    return {
        'quality_score': quality_score,
        'snr_estimate': snr_estimate,
        'dynamic_range': dynamic_range,
        'sample_rate': int(sr),
        'bit_depth_estimate': 16 if quality_score > 70 else 8
    }

def preprocess_audio_chunk(audio_data, sr=16000):
    """Preprocess audio for model prediction"""
    # Resample if needed
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure 30 seconds
    target_length = sr * 30
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
    
    # Process with Whisper
    audio_data = audio_data.astype(np.float32)
    log_mel_spectrogram = whisper.audio.log_mel_spectrogram(audio_data, n_mels=80)
    
    # Convert to tensor
    mel_np = log_mel_spectrogram.numpy()
    normalized = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min())
    img = (normalized * 255).astype(np.uint8)
    img = np.stack([img, img, img], axis=-1)
    
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((80, 1500), antialias=True),
    ])
    
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor

def create_spectrogram_image(audio_data, sr):
    """Create clean spectrogram visualization without axes or labels"""
    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=80)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure with no frame, axes, or labels
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Remove all axes, labels, and borders
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Display spectrogram with beautiful colormap
    img = ax.imshow(mel_spec_db, 
                   aspect='auto', 
                   origin='lower',
                   cmap='plasma',  # Beautiful purple-pink-yellow colormap
                   interpolation='bilinear')
    
    # Remove all padding and margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=120, bbox_inches='tight', 
                pad_inches=0, facecolor='black', edgecolor='none')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files with proper headers"""
    try:
        response = send_from_directory('static', filename, as_attachment=False)
        response.headers['Accept-Ranges'] = 'bytes'
        # Set content type based on file extension
        if filename.endswith('.mp3'):
            response.headers['Content-Type'] = 'audio/mpeg'
        else:
            response.headers['Content-Type'] = 'audio/wav'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        return jsonify({'error': 'Audio file not found'}), 404

@app.route('/cleanup/<audio_id>')
def cleanup_audio(audio_id):
    """Clean up temporary audio files"""
    try:
        # Clean up both wav and mp3 possibilities
        for ext in ['.wav', '.mp3']:
            playback_filename = f"temp_audio_{audio_id}{ext}"
            playback_path = os.path.join('static', playback_filename)
            if os.path.exists(playback_path):
                os.unlink(playback_path)
                print(f"Cleaned up {playback_path}")
        return jsonify({'status': 'cleaned'})
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return jsonify({'status': 'error'})

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Main audio analysis endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporary file
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        if not file_extension:
            file_extension = '.wav'  # Default for recordings
            
        # Create a unique filename for this session
        audio_id = str(int(time.time() * 1000))  # Use timestamp as unique ID
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            audio_file.save(tmp_file.name)
            
            # --- Audio Conversion to MP3 ---
            playback_filename = f"temp_audio_{audio_id}.mp3" # Save as MP3
            playback_path = os.path.join('static', playback_filename)
            os.makedirs('static', exist_ok=True)

            try:
                # Load the audio file of any format
                audio = AudioSegment.from_file(tmp_file.name)
                # Export as MP3 for maximum browser compatibility
                audio.export(playback_path, format="mp3")
                print(f"Successfully converted {audio_file.filename} to {playback_path}")
            except Exception as e:
                print(f"Audio conversion failed: {e}")
                # Fallback: just copy the file if conversion fails
                import shutil
                shutil.copy2(tmp_file.name, playback_path)

            # Load audio with error handling for analysis
            try:
                audio_data, sr = librosa.load(tmp_file.name, sr=None, mono=True)
            except Exception as e:
                # Try different formats if loading fails
                try:
                    audio_data, sr = librosa.load(tmp_file.name, sr=22050, mono=True)
                except Exception as e2:
                    os.unlink(tmp_file.name)
                    if os.path.exists(playback_path):
                        os.unlink(playback_path)
                    return jsonify({'error': f'Unable to load audio file: {str(e2)}'}), 400
            
        # Clean up original temp file
        os.unlink(tmp_file.name)
        
        if len(audio_data) < sr:  # Less than 1 second
            return jsonify({'error': 'Audio file too short (minimum 1 second)'}), 400
        
        # Run all analyses
        results = {}
        
        # 1. Genre Classification
        if model is not None:
            try:
                tensor = preprocess_audio_chunk(audio_data, sr)
                tensor = tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    # Get the predictions for the first (and only) sample in the batch
                    sample_probs = probabilities[0]  # Shape: [10]
                    confidence, predicted_index = torch.max(sample_probs, 0)
                    
                    predicted_genre = class_names[predicted_index.item()]
                    
                    # Get top 3 predictions
                    top_probs, top_indices = torch.topk(sample_probs, 3)
                    top_predictions = [
                        {'genre': class_names[idx.item()], 'confidence': float(prob.item())}
                        for prob, idx in zip(top_probs, top_indices)
                    ]
                    
                    results['genre'] = {
                        'predicted': predicted_genre,
                        'confidence': float(confidence.item()),
                        'top_predictions': top_predictions
                    }
            except Exception as e:
                results['genre'] = {'error': str(e)}
        else:
            results['genre'] = {'error': 'Model not loaded'}
        
        # 2. Mood and Energy Analysis
        try:
            mood_results = analyze_mood_energy(audio_data, sr)
            results['mood'] = mood_results
        except Exception as e:
            results['mood'] = {'error': str(e)}
        
        # 3. Key and Scale Detection
        try:
            key_results = detect_key_scale(audio_data, sr)
            results['key'] = key_results
        except Exception as e:
            results['key'] = {'error': str(e)}
        
        # 4. Audio Quality Assessment
        try:
            quality_results = assess_audio_quality(audio_data, sr)
            results['quality'] = quality_results
        except Exception as e:
            results['quality'] = {'error': str(e)}
        
        # 5. Create spectrogram
        try:
            spectrogram_img = create_spectrogram_image(audio_data[:sr*30], sr)  # First 30 seconds
            results['spectrogram'] = spectrogram_img
        except Exception as e:
            results['spectrogram'] = None
        
        # 6. Basic audio info
        results['audio_info'] = {
            'duration': float(len(audio_data) / sr),
            'sample_rate': int(sr),
            'channels': 1,  # We force mono
            'file_size_mb': len(audio_data) * 4 / (1024 * 1024)  # Approximate
        }
        
        # 7. Audio file for playback
        results['audio_url'] = f'/audio/{playback_filename}'
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# WebSocket handlers removed for compatibility - using simple REST API

if __name__ == '__main__':
    print("🎵 Starting MusicAI Pro...")
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  Model not loaded - genre classification will be disabled")
    
    print("🚀 Starting web server on http://localhost:5100")
    print("📱 Open your browser and navigate to http://localhost:5100")
    app.run(debug=True, host='0.0.0.0', port=5100, threaded=True) 