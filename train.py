import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data
from dataloader import MusicGenreDataset
from model import WhisperAudioClassifier
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

def plot_confusion_matrix(cm, class_names, file_path='confusion_matrix.png'):
    """
    Plots a confusion matrix using seaborn and saves it to a file.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    if wandb.run:
        wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    print(f"Confusion matrix saved to {file_path}")

def evaluate(model, test_loader, device, class_names):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="EVALUATING"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    print("\n" + "="*50)
    print("                   EVALUATION RESULTS")
    print("="*50 + "\n")
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(report)

    if wandb.run:
        report_dict = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        wandb.log({"classification_report": report_dict})
    
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, class_names)
    

def train(model, train_loader, val_loader, epochs, learning_rate, device, model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 5

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if wandb.run:
                wandb.log({"batch_loss": loss.item(), "global_step": global_step})
            global_step += 1

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if wandb.run:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model improved and saved to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {early_stop_patience} epochs with no improvement.")
                break
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == '__main__':
    # --- Configuration ---
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 8  # Further reduced for Whisper + raw audio processing
    MODEL_SAVE_PATH = 'music_genre_classifier.pth'
    
    # --- W&B Initialization ---
    config = {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    wandb.init(
        project="music-genre-classifier-transformers",
        config=config,
        name=f"run-{wandb.util.generate_id()}"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Datasets and Dataloaders ---
    print("\nLoading datasets...")
    # GTZAN only has 'train' split, so we'll create our own splits
    full_dataset = MusicGenreDataset(split='train', augment=False)
    
    # Split GTZAN into train/val/test (70/15/15)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create augmented training dataset separately
    train_augmented = MusicGenreDataset(split='train', augment=True)
    # Use the same indices as train_dataset for consistency
    train_dataset = torch.utils.data.Subset(train_augmented, train_dataset.indices)

    # Set pin_memory and num_workers based on device
    use_pin_memory = device.type == 'cuda'
    num_workers = 4 if device.type == 'cuda' else 0  # Avoid multiprocessing issues on CPU
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=use_pin_memory)

    # --- Model ---
    model = WhisperAudioClassifier(num_classes=full_dataset.num_classes, device=device)
    wandb.watch(model, log="all", log_freq=100)
    
    # --- Training ---
    print("\nStarting training...")
    trained_model = train(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device, MODEL_SAVE_PATH)
    
    # --- Evaluation ---
    print("\nStarting evaluation on the test set...")
    evaluate(trained_model, test_loader, device, full_dataset.class_names)
    print("\nProcess finished.") 
    wandb.finish() 