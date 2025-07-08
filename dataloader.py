import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_dataset_builder
import librosa
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import torchvision.transforms.v2 as transforms

class MusicGenreDataset(Dataset):
    def __init__(self, split='train', target_sample_rate=22050, duration=30, augment=False):
        self.dataset = load_dataset("ccmusic-database/music_genre", name="eval", split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        
        ds_builder = load_dataset_builder("ccmusic-database/music_genre", "eval")
        self.class_names = ds_builder.info.features['sec_level_label'].names
        self.num_classes = len(self.class_names)
        
        self.augment = augment
        if self.augment:
            # Using torchvision transforms for image augmentation
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((80, 3000), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((80, 3000), antialias=True),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        mel_spectrogram_image = item['mel'] # This is a PIL Image
        label = item['sec_level_label']

        mel_spectrogram_tensor = self.transform(mel_spectrogram_image)

        return mel_spectrogram_tensor, torch.tensor(label, dtype=torch.long)

def main():
    print("--- Running Dataloader Inspection ---")
    
    # Load a sample from the training set without augmentation for a clean comparison
    dataset = MusicGenreDataset(split='train', augment=False)
    
    # --- Inspecting the raw PIL image from the dataset ---
    raw_item = dataset.dataset[0]
    pil_image = raw_item['mel']
    print("\n--- Raw PIL Image from Hugging Face Dataset ---")
    print(f"PIL Image Mode: {pil_image.mode}")
    print(f"PIL Image Size: {pil_image.size}")
    
    # --- Inspecting the tensor after transformation ---
    spectrogram_tensor, label_id = dataset[0]
    
    print("\n--- Tensor after transformations ---")
    print(f"Tensor shape: {spectrogram_tensor.shape}")
    print(f"Tensor dtype: {spectrogram_tensor.dtype}")
    print(f"Tensor min value: {spectrogram_tensor.min():.6f}")
    print(f"Tensor max value: {spectrogram_tensor.max():.6f}")
    
    # --- Saving the spectrogram for visual comparison ---
    # The tensor is (C, H, W). We need (H, W) for imsave.
    numpy_spectrogram = spectrogram_tensor.squeeze(0).cpu().numpy()
    
    save_path = "dataloader_spectrogram_sample.png"
    plt.imsave(save_path, numpy_spectrogram, cmap='viridis', origin='lower')
    
    print(f"\nSaved a sample spectrogram to '{save_path}'.")

if __name__ == "__main__":
    main()

