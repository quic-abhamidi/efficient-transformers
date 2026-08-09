#!/usr/bin/env python3
"""
Learnable Compensation for Layer Skipping

Instead of using a fixed mean vector, this module learns a transformation
that maps h_input → h_expected when layers are skipped.

The key insight: The diagnostic shows mean vector explains 0% of variance,
meaning the transformation is input-dependent. We need a learnable function.

Architecture:
    h_compensated = h_input + MLP(h_input)

Where MLP can be:
1. Linear: W @ h_input + b
2. Two-layer: W2 @ ReLU(W1 @ h_input + b1) + b2
3. Residual: h_input + alpha * tanh(W @ h_input + b)

Training:
- Collect (h_input, h_expected) pairs from calibration data
- Minimize MSE: ||h_compensated - h_expected||^2
- Use small dataset (1000-5000 samples)
- Train for 10-50 epochs with Adam optimizer

Author: LLM Interpretability Engineer
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LearnableCompensation(nn.Module):
    """
    Learnable compensation module that predicts the residual needed
    to compensate for skipped layers.
    
    h_out = h_in + compensation_network(h_in)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        compensation_type: str = "linear",
        hidden_size: Optional[int] = None,
        alpha: float = 1.0,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            compensation_type: "linear", "mlp", or "residual"
            hidden_size: Hidden size for MLP (default: hidden_dim // 4)
            alpha: Scale factor for residual connection
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.compensation_type = compensation_type
        self.alpha = alpha
        
        if hidden_size is None:
            hidden_size = hidden_dim // 4
        
        if compensation_type == "linear":
            # Simple linear transformation: W @ h + b
            self.compensation_net = nn.Linear(hidden_dim, hidden_dim, bias=True)
            
        elif compensation_type == "mlp":
            # Two-layer MLP with ReLU
            self.compensation_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_dim),
            )
            
        elif compensation_type == "residual":
            # Residual with tanh activation for bounded output
            self.compensation_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_dim),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown compensation_type: {compensation_type}")
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply learned compensation.
        
        Args:
            h: Hidden states [batch, seq_len, hidden_dim]
        
        Returns:
            Compensated hidden states [batch, seq_len, hidden_dim]
        """
        compensation = self.compensation_net(h)
        
        if self.compensation_type == "residual":
            return h + self.alpha * compensation
        else:
            return h + compensation
    
    def compensate(
        self,
        h: torch.Tensor,
        is_decode: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        BaseCompensation-compatible interface.
        Applies learned compensation to hidden states.

        Args:
            h: Hidden states [batch, seq_len, hidden_dim]
            is_decode: True if decode step (unused, compensation is position-agnostic)
            token_positions: Unused

        Returns:
            Compensated hidden states [batch, seq_len, hidden_dim]
        """
        return self.forward(h)

    def to(self, device) -> "LearnableCompensation":
        """Move compensation module to device (BaseCompensation-compatible)."""
        return super().to(device)

    def save(self, path: Union[str, Path]):
        """Save model weights."""
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'compensation_type': self.compensation_type,
            'alpha': self.alpha,
        }, path)
        print(f"Saved learnable compensation to: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cuda") -> "LearnableCompensation":
        """Load model weights."""
        checkpoint = torch.load(path, map_location=device)
        # Infer hidden_size from saved state_dict if not stored explicitly
        hidden_size = checkpoint.get('hidden_size', None)
        if hidden_size is None and checkpoint['compensation_type'] in ('mlp', 'residual'):
            # Infer from first layer weight shape
            w = checkpoint['state_dict'].get('compensation_net.0.weight')
            if w is not None:
                hidden_size = w.shape[0]
        model = cls(
            hidden_dim=checkpoint['hidden_dim'],
            compensation_type=checkpoint['compensation_type'],
            hidden_size=hidden_size,
            alpha=checkpoint.get('alpha', 1.0),
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print(f"Loaded learnable compensation from: {path}")
        return model


class CompensationDataset(Dataset):
    """Dataset of (h_input, h_expected) pairs for training compensation."""
    
    def __init__(self, h_input_list: List[torch.Tensor], h_expected_list: List[torch.Tensor]):
        """
        Args:
            h_input_list: List of input hidden states (before skip)
            h_expected_list: List of expected hidden states (after skip without compensation)
        """
        assert len(h_input_list) == len(h_expected_list)
        self.h_input = torch.stack(h_input_list)
        self.h_expected = torch.stack(h_expected_list)
    
    def __len__(self) -> int:
        return len(self.h_input)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.h_input[idx], self.h_expected[idx]


def train_compensation(
    model: LearnableCompensation,
    train_dataset: CompensationDataset,
    val_dataset: Optional[CompensationDataset] = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """
    Train the learnable compensation module.
    
    Args:
        model: LearnableCompensation module
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        verbose: Print training progress
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    model.train()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_norm_error': [],
        'val_norm_error': [],
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_norm_errors = []
        
        for h_input, h_expected in train_loader:
            h_input = h_input.to(device)
            h_expected = h_expected.float().to(device)
            
            optimizer.zero_grad()
            h_compensated = model(h_input.float())
            loss = criterion(h_compensated, h_expected)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Compute norm error
            norm_error = (h_compensated.norm(dim=-1) - h_expected.norm(dim=-1)).abs().mean()
            train_norm_errors.append(norm_error.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_train_norm_error = np.mean(train_norm_errors)
        history['train_loss'].append(avg_train_loss)
        history['train_norm_error'].append(avg_train_norm_error)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            val_norm_errors = []
            
            with torch.no_grad():
                for h_input, h_expected in val_loader:
                    h_input = h_input.to(device)
                    h_expected = h_expected.float().to(device)
                    
                    h_compensated = model(h_input.float())
                    loss = criterion(h_compensated, h_expected)
                    val_losses.append(loss.item())
                    
                    norm_error = (h_compensated.norm(dim=-1) - h_expected.norm(dim=-1)).abs().mean()
                    val_norm_errors.append(norm_error.item())
            
            avg_val_loss = np.mean(val_losses)
            avg_val_norm_error = np.mean(val_norm_errors)
            history['val_loss'].append(avg_val_loss)
            history['val_norm_error'].append(avg_val_norm_error)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}, Norm Error: {avg_train_norm_error:.4f}")
                print(f"  Val Loss:   {avg_val_loss:.6f}, Norm Error: {avg_val_norm_error:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}, Norm Error: {avg_train_norm_error:.4f}")
    
    return history


def collect_compensation_data(
    model,
    tokenizer,
    skip_layers: List[int],
    dataset_samples: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect (h_input, h_expected) pairs for training compensation.
    
    Args:
        model: Base transformer model
        tokenizer: Tokenizer
        skip_layers: Layers to skip
        dataset_samples: List of text samples
        device: Device
        max_length: Max sequence length
    
    Returns:
        Tuple of (h_input_list, h_expected_list)
    """
    model.eval()
    
    # Get layer before skip
    compensation_layer = min(skip_layers) - 1
    
    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find decoder layers")
    
    h_input_list = []
    h_expected_list = []
    
    print(f"Collecting compensation data from {len(dataset_samples)} samples...")
    
    with torch.no_grad():
        for idx, text in enumerate(dataset_samples):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(dataset_samples)} samples")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                             truncation=True, padding=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with hooks to capture hidden states
            h_input = None
            h_expected = None
            
            def capture_input(module, input, output):
                nonlocal h_input
                if isinstance(output, tuple):
                    h_input = output[0].detach().cpu()
                else:
                    h_input = output.detach().cpu()
            
            def capture_expected(module, input, output):
                nonlocal h_expected
                if isinstance(output, tuple):
                    h_expected = output[0].detach().cpu()
                else:
                    h_expected = output.detach().cpu()
            
            # Register hooks
            hook1 = layers[compensation_layer].register_forward_hook(capture_input)
            hook2 = layers[max(skip_layers)].register_forward_hook(capture_expected)
            
            try:
                _ = model(**inputs)

                if h_input is not None and h_expected is not None:
                    # Store ALL token positions (not just last) for better generalization
                    # h_input shape: [1, seq_len, hidden_dim] → [seq_len, hidden_dim]
                    seq_len = h_input.shape[1]
                    for pos in range(seq_len):
                        h_input_list.append(h_input[0, pos, :])   # [hidden_dim]
                        h_expected_list.append(h_expected[0, pos, :])  # [hidden_dim]
            finally:
                hook1.remove()
                hook2.remove()
    
    print(f"Collected {len(h_input_list)} training samples")
    return h_input_list, h_expected_list
