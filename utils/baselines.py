"""Fit regularized linear classifiers using training subjects only."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from .augmentation import load_augmented_views


def fit_baseline(model, train_dataset, augmentation_dir=None, regularization_c=1.0):
    if regularization_c <= 0:
        raise ValueError("baseline C must be positive")
    device = next(model.parameters()).device
    originals, augmented, targets = [], [], []
    with torch.no_grad():
        for index in range(len(train_dataset)):
            features, label, sid = train_dataset[index]
            originals.append(model.classifier_features(features.to(device)).cpu().numpy()[0])
            targets.append(label)
            subject_views = []
            if augmentation_dir is not None:
                for view in load_augmented_views(Path(augmentation_dir) / f"{sid}_aug_features.pt", features, sid):
                    subject_views.append(model.classifier_features(view.to(device)).cpu().numpy()[0])
            augmented.append(subject_views)
    if set(targets) != {0, 1}:
        raise ValueError("Baseline fitting requires both TRAINING classes.")
    # Fit feature scaling on original training slides.
    scaler = StandardScaler().fit(np.asarray(originals))
    representations, labels, sample_weights = [], [], []
    for original, views, label in zip(originals, augmented, targets):
        subject_views = [original] + views
        for view in subject_views:
            representations.append(view)
            labels.append(label)
            # Give each animal a total sample weight of one across its image views.
            sample_weights.append(1.0 / len(subject_views))
    classifier = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=5000,
                                    tol=1e-8, random_state=0)
    classifier.fit(scaler.transform(np.asarray(representations)), labels,
                   sample_weight=np.asarray(sample_weights))
    with torch.no_grad():
        model.feature_center.copy_(torch.as_tensor(scaler.mean_, device=device, dtype=torch.float32))
        model.feature_scale.copy_(torch.as_tensor(scaler.scale_, device=device, dtype=torch.float32))
        model.classifier.weight.copy_(torch.as_tensor(classifier.coef_, device=device, dtype=torch.float32))
        model.classifier.bias.copy_(torch.as_tensor(classifier.intercept_, device=device, dtype=torch.float32))
    model.eval()
