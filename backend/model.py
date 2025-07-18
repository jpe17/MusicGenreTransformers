import torch
import torch.nn as nn
import whisper

class WhisperAudioClassifier(nn.Module):
    def __init__(self, num_classes, whisper_model_name="tiny", device=None):
        """
        Initializes the WhisperAudioClassifier.

        Args:
            num_classes (int): The number of classes for classification.
            whisper_model_name (str): The name of the Whisper model to use (e.g., "tiny", "base").
            device (torch.device or str, optional): The device to load the model on.
        """
        super().__init__()
        # Load Whisper model on CPU first to avoid MPS sparse tensor issues
        self.whisper_model = whisper.load_model(whisper_model_name, device="cpu")
        
        # We only need the encoder part of the Whisper model.
        self.encoder = self.whisper_model.encoder
        
        # Freeze all parameters of the encoder initially.
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze the convolutional layers to allow fine-tuning.
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True
        for param in self.encoder.conv2.parameters():
            param.requires_grad = True
            
        # The output dimension of the encoder is n_state.
        encoder_output_dim = self.encoder.ln_post.normalized_shape[0]
        
        # A linear layer to act as the classification head.
        self.classifier = nn.Linear(encoder_output_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): The input mel spectrogram of shape (batch_size, 1, n_mels, n_ctx).
        
        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        # The dataloader returns a tensor with a channel dimension,
        # but the encoder expects (batch, n_mels, n_ctx).
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Process through conv layers first (like Whisper does)
        x = torch.nn.functional.gelu(self.encoder.conv1(x))
        x = torch.nn.functional.gelu(self.encoder.conv2(x))
        
        # Permute to (batch, seq_len, d_model) for transformer
        x = x.permute(0, 2, 1)
        
        # Add positional embedding
        x = x + self.encoder.positional_embedding[:x.shape[1], :]
        
        # Pass through transformer blocks
        for block in self.encoder.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.encoder.ln_post(x)
        
        # Average pooling over the sequence dimension to get a fixed-size vector.
        x = torch.mean(x, dim=1)
        
        # Pass the pooled output through the classification head.
        logits = self.classifier(x)
        
        return logits
