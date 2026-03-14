"""sklearn-compatible wrapper for PyTorch nn.Module models.

Saved via cloudpickle so custom nn.Module subclasses defined in
dynamic code (exec) are handled correctly. This module lives in
tools/models-tools/training/ (which is on sys.path) so it can be
imported at load time.

The label_encoder (sklearn LabelEncoder) is stored alongside the
model so predictions on string-labelled targets are decoded back
to the original label space automatically.
"""

import numpy as np
import pandas as pd


class PyTorchPredictor:
    """Wraps a PyTorch nn.Module with sklearn-style predict/predict_proba.

    Works with any nn.Module subclass (Sequential, custom classes, etc.)
    because cloudpickle handles dynamically-defined classes.
    """

    def __init__(self, model, preprocessor, task_type, n_classes=None, label_encoder=None):
        import torch
        self._model = model.cpu().eval()
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.n_classes = n_classes
        self.label_encoder = label_encoder

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

    @property
    def classes_(self):
        if self.label_encoder is not None:
            return self.label_encoder.classes_
        if self.n_classes is not None:
            return np.arange(self.n_classes)
        return None
