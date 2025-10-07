# src/classical.py
from typing import Callable, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import local_binary_pattern

def make_eigenfaces(pca_components: int = 150) -> Tuple[Pipeline, Callable[[np.ndarray], np.ndarray]]:
    """
    Eigenfaces pipeline: flatten grayscale -> PCA(whiten) -> LinearSVC.
    Returns (clf, extractor), where extractor(img_gray) -> 1D float vector in [0,1].
    """
    def extractor(img_gray: np.ndarray) -> np.ndarray:
        x = img_gray.reshape(-1).astype(np.float32) / 255.0
        return x

    clf = Pipeline([
        ("pca", PCA(n_components=pca_components, whiten=True, svd_solver="randomized")),
        ("svm", LinearSVC())
    ])
    return clf, extractor

def make_lbp(radius: int = 2, points: int = 16) -> Tuple[Pipeline, Callable[[np.ndarray], np.ndarray]]:
    """
    LBP-histograms pipeline: uniform LBP -> normalized histogram -> LinearSVC.
    'points' is the number of circularly symmetric neighbor set points (e.g., 8, 16, 24).
    """
    def extractor(img_gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(img_gray, P=points, R=radius, method="uniform")
        bins = points + 2  # for uniform mapping
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        return hist.astype(np.float32)

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm", LinearSVC())
    ])
    return clf, extractor
