import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_dataset_builder
import librosa
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import torchvision.transforms.v2 as transforms
import whisper

class MusicGenreDataset(Dataset):
    def __init__(self, split='train', target_sample_rate=16000, duration=30, augment=False):
        # Load GTZAN dataset with raw audio
        self.dataset = load_dataset("marsyas/gtzan", "all", split=split, trust_remote_code=True)
        self.dataset = self.dataset.shuffle(seed=42)
        
        # GTZAN genre classes
        self.class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.num_classes = len(self.class_names)
        
        # Audio processing parameters
        self.target_sample_rate = target_sample_rate
        self.duration = duration
        
        self.augment = augment
        if self.augment:
            # Using torchvision transforms for image augmentation
            # Whisper expects (80, 1500) mel spectrograms
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((80, 1500), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((80, 1500), antialias=True),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get raw audio and label from GTZAN
        audio_data = item['audio']['array']
        sample_rate = item['audio']['sampling_rate']
        label = item['genre']

        # Resample to Whisper's expected 16kHz if needed
        if sample_rate != self.target_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.target_sample_rate)

        # Trim or pad audio to target duration at 16kHz
        target_length = int(self.target_sample_rate * self.duration)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')

        # Use Whisper's built-in mel spectrogram preprocessing
        # This ensures perfect compatibility with the Whisper encoder
        # Convert to float32 (Whisper requirement)
        audio_data = audio_data.astype(np.float32)
        log_mel_spectrogram = whisper.audio.log_mel_spectrogram(audio_data, n_mels=80)
        
        # Whisper returns (n_mels, n_frames), we need to normalize for PIL conversion
        # Convert to numpy and normalize to [0, 1] range
        mel_np = log_mel_spectrogram.numpy()
        normalized = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min())
        
        # Convert to PIL Image format (RGB)
        img = (normalized * 255).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)

        # Apply transforms
        mel_spectrogram_tensor = self.transform(img)

        return mel_spectrogram_tensor, torch.tensor(label, dtype=torch.long)

def main():
    print("--- Running Dataloader Inspection ---")
    
    # Load a sample from the training set without augmentation for a clean comparison
    dataset = MusicGenreDataset(split='train', augment=False)
    
    # --- Inspecting the raw audio from GTZAN dataset ---
    raw_item = dataset.dataset[0]
    audio_data = raw_item['audio']
    print("\n--- Raw Audio from GTZAN Dataset ---")
    print(f"Audio array shape: {np.array(audio_data['array']).shape}")
    print(f"Sample rate: {audio_data['sampling_rate']}")
    print(f"Genre: {dataset.class_names[raw_item['genre']]}")
    
    # --- Inspecting the tensor after transformation ---
    spectrogram_tensor, label_id = dataset[0]
    
    print("\n--- Whisper-Compatible Tensor after transformations ---")
    print(f"Tensor shape: {spectrogram_tensor.shape}")
    print(f"Expected shape: torch.Size([1, 80, 1500]) (Whisper compatible)")
    print(f"Tensor dtype: {spectrogram_tensor.dtype}")
    print(f"Tensor min value: {spectrogram_tensor.min():.6f}")
    print(f"Tensor max value: {spectrogram_tensor.max():.6f}")
    print(f"Label: {dataset.class_names[label_id]} (index: {label_id})")
    
    # --- Saving the spectrogram for visual comparison ---
    # The tensor is (C, H, W). We need (H, W) for imsave.
    numpy_spectrogram = spectrogram_tensor.squeeze(0).cpu().numpy()
    
    save_path = "dataloader_spectrogram_sample.png"
    plt.imsave(save_path, numpy_spectrogram, cmap='viridis', origin='lower')
    
    print(f"\nSaved a sample spectrogram to '{save_path}'.")

if __name__ == "__main__":
    main()

