"""sklearn-compatible wrapper for PyTorch nn.Module models.

Saved via cloudpickle so custom nn.Module subclasses defined in
dynamic code (exec) are handled correctly. This module lives in
tools/models-tools/training/ (which is on sys.path) so it can be
imported at load time.

The label_encoder (sklearn LabelEncoder) is stored alongside the
model so predictions on string-labelled targets are decoded back
to the original label space automatically.

For unsupervised models (autoencoders, deep clustering), use
transform() to get embeddings or reconstruct().
"""

import numpy as np
import pandas as pd


class PyTorchPredictor:
    """Wraps a PyTorch nn.Module with sklearn-style predict/predict_proba/transform.

    Works with any nn.Module subclass (Sequential, custom classes, etc.)
    because cloudpickle handles dynamically-defined classes.

    For unsupervised models:
    - predict() returns cluster labels (if the model outputs discrete labels)
      or the raw output tensor for autoencoders.
    - transform() returns the latent embedding (encoder output).
    - reconstruct() returns the full forward pass output (for autoencoders).
    """

    def __init__(self, model, preprocessor, task_type, n_classes=None, label_encoder=None,
                 encoder=None):
        """
        Args:
            encoder: Optional separate encoder module for autoencoders. If provided,
                     transform() uses this; otherwise falls back to the full model.
        """
        import torch
        self._model = model.cpu().eval()
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.n_classes = n_classes
        self.label_encoder = label_encoder
        self._encoder = encoder.cpu().eval() if encoder is not None else None

    def _get_model(self):
        self._model.eval()
        return self._model

    def _preprocess(self, X):
        if isinstance(X, pd.DataFrame) and self.preprocessor is not None:
            return self.preprocessor.transform(X).astype(np.float32)
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def predict(self, X):
        import torch
        model = self._get_model()
        X_np = self._preprocess(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_t)

        if self.task_type == "unsupervised":
            return output.numpy()

        if self.task_type == "regression":
            return output.squeeze().numpy()
        int_preds = output.argmax(dim=1).numpy()
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(int_preds)
        return int_preds

    def predict_proba(self, X):
        import torch
        model = self._get_model()
        X_np = self._preprocess(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            output = model(X_t)
        return torch.softmax(output, dim=1).numpy()

    def transform(self, X):
        """Return latent embeddings from the encoder.

        For autoencoders with a separate encoder, uses the encoder.
        Otherwise passes through the full model (useful for deep
        clustering where the forward pass IS the embedding).
        """
        import torch
        net = self._encoder if self._encoder is not None else self._get_model()
        net.eval()
        X_np = self._preprocess(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            return net(X_t).numpy()

    def reconstruct(self, X):
        """Full autoencoder forward pass — returns reconstructed output."""
        import torch
        model = self._get_model()
        X_np = self._preprocess(X)
        X_t = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            return model(X_t).numpy()

    @property
    def classes_(self):
        if self.label_encoder is not None:
            return self.label_encoder.classes_
        if self.n_classes is not None:
            return np.arange(self.n_classes)
        return None
