import torch
import librosa
import numpy as np
from model import WhisperAudioClassifier
from dataloader import MusicGenreDataset
import argparse
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import whisper

def preprocess_audio(audio_path, sample_rate=16000, n_mels=80, chunk_duration=30, save_spectrogram=False):
    """
    Loads a long audio file, splits it into 30-second chunks, and processes
    each chunk into a mel spectrogram tensor using Whisper's preprocessing.
    Returns a batch of spectrogram tensors.
    """
    # 1. Load audio file, resampling to 16000 Hz (Whisper's requirement) and ensuring it's mono
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # 2. Split into chunks of `chunk_duration`
    chunk_length = sample_rate * chunk_duration
    chunks = [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)]

    # Ensure the last chunk is padded to be exactly `chunk_duration` long
    if len(chunks[-1]) < chunk_length:
        chunks[-1] = np.pad(chunks[-1], (0, chunk_length - len(chunks[-1])), 'constant')
    
    # Filter out any potential empty or very short chunks if the audio is less than 1s
    min_length = sample_rate * 1
    chunks = [c for c in chunks if len(c) >= min_length]
    if not chunks:
        raise ValueError("Audio file is too short to process (must be at least 1s long).")

    print(f"Audio split into {len(chunks)} chunk(s) of {chunk_duration} seconds.")

    # 3. Define the processing pipeline to match dataloader.py (Whisper dimensions)
    batch_tensors = []
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((80, 1500), antialias=True),
    ])

    for i, chunk in enumerate(chunks):
        # 4. Use Whisper's built-in mel spectrogram preprocessing (EXACT same as dataloader)
        # Convert to float32 (Whisper requirement)
        chunk = chunk.astype(np.float32)
        log_mel_spectrogram = whisper.audio.log_mel_spectrogram(chunk, n_mels=n_mels)
        
        # 5. Convert to numpy and normalize exactly like dataloader
        mel_np = log_mel_spectrogram.numpy()
        normalized = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min())
        img = (normalized * 255).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)

        # 6. Apply the full transform pipeline
        mel_spectrogram_tensor = transform(img)
        batch_tensors.append(mel_spectrogram_tensor)

        # Save the final processed tensor as an image for the first chunk if requested
        if save_spectrogram and i == 0:
            spectrogram_path = "predict_spectrogram_final.png"
            numpy_spectrogram = mel_spectrogram_tensor.squeeze(0).cpu().numpy()
            plt.imsave(spectrogram_path, numpy_spectrogram, cmap='viridis', origin='lower')
            print(f"Saved final spectrogram of chunk 0 to {spectrogram_path}")
    
    final_batch = torch.stack(batch_tensors)

    print("\n" + "="*50)
    print("     WHISPER-COMPATIBLE SPECTROGRAM INSPECTION")
    print("="*50 + "\n")
    if final_batch.nelement() > 0:
        first_tensor = final_batch[0]
        print(f"Tensor shape: {first_tensor.shape}")
        print(f"Expected shape: torch.Size([1, 80, 1500]) (Whisper compatible)")
        print(f"Tensor dtype: {first_tensor.dtype}")
        print(f"Tensor min value: {first_tensor.min():.6f}")
        print(f"Tensor max value: {first_tensor.max():.6f}")
    else:
        print("No tensors were generated to inspect.")
    print("="*50)

    return final_batch


def predict(model, audio_batch, class_names, device):
    """
    Makes a prediction on a batch of preprocessed audio tensors from a long audio file.
    Aggregates predictions by averaging probabilities and returns individual chunk predictions.
    """
    model.eval()
    model.to(device)
    audio_batch = audio_batch.to(device)
    
    with torch.no_grad():
        outputs = model(audio_batch)
        probabilities = torch.softmax(outputs, dim=1)
        
    # Individual chunk predictions
    chunk_predictions = []
    for i in range(probabilities.shape[0]):
        chunk_probs = probabilities[i]
        chunk_confidence, chunk_predicted_index = torch.max(chunk_probs, 0)
        chunk_predicted_class = class_names[chunk_predicted_index.item()]
        chunk_predictions.append({
            "chunk": i,
            "prediction": chunk_predicted_class,
            "confidence": chunk_confidence.item()
        })
    
    # Aggregated prediction
    avg_probabilities = torch.mean(probabilities, dim=0)
    
    confidence, predicted_index = torch.max(avg_probabilities, 0)
    
    predicted_class_name = class_names[predicted_index.item()]
    
    return predicted_class_name, confidence.item(), chunk_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify the genre of a music file.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file (e.g., mp3, wav).")
    parser.add_argument("--save-spectrogram", action="store_true", help="Save the final processed spectrogram of the first chunk as an image file (predict_spectrogram_final.png).")
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_PATH = 'music_genre_classifier.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Class Names ---
    # We instantiate the dataset object just to get the class mapping
    print("Loading class names...")
    # GTZAN only has 'train' split, so we use that for metadata
    temp_dataset = MusicGenreDataset(split='train')
    CLASS_NAMES = temp_dataset.class_names
    NUM_CLASSES = temp_dataset.num_classes
    del temp_dataset
    
    # --- Load Model ---
    print("Loading model...")
    model = WhisperAudioClassifier(num_classes=NUM_CLASSES, device=device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please make sure you have a trained model file.")
        exit(1)

    # --- Preprocess Audio ---
    print(f"Processing audio file: {args.audio_file}")
    try:
        audio_batch = preprocess_audio(args.audio_file, save_spectrogram=args.save_spectrogram)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        exit(1)

    # --- Predict ---
    print("Classifying genre...")
    predicted_genre, confidence, chunk_predictions = predict(model, audio_batch, CLASS_NAMES, device)
    
    print("\n" + "="*30)
    print("   Individual Chunk Predictions")
    print("="*30)
    for chunk_pred in chunk_predictions:
        print(f"Chunk {chunk_pred['chunk']}: {chunk_pred['prediction']} (Confidence: {chunk_pred['confidence']:.2%})")

    print("\n" + "="*30)
    print("    Aggregated Final Result")
    print("="*30)
    print(f"Predicted Genre: {predicted_genre}")
    print(f"Confidence: {confidence:.2%}")
    print("="*30) 