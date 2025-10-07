# src/deep.py
import numpy as np, torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class FaceEmbedder:
    def __init__(self, size: int = 160, use_pretrained: bool = True, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.mtcnn = MTCNN(image_size=size, margin=20, post_process=True, device=self.device)
        self.net = InceptionResnetV1(pretrained='vggface2' if use_pretrained else None).eval().to(self.device)

    @torch.no_grad()
    def align(self, pil_img: Image.Image) -> Image.Image:
        t = self.mtcnn(pil_img, return_prob=False)
        if t is None:
            return pil_img.resize((self.size, self.size))
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def embed(self, pil_img: Image.Image) -> np.ndarray:
        x = np.array(pil_img).astype(np.float32) / 255.0
        if x.ndim == 2:  # gray -> RGB
            x = np.repeat(x[..., None], 3, axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        emb = self.net(x).detach().cpu().numpy()[0]
        return emb.astype(np.float32)

def compute_centroids(embeddings: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    cents = {}
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        cents[c] = np.mean(embeddings[idx], axis=0)
    return cents

def cosine_predict(vec: np.ndarray, cents: dict[str, np.ndarray], threshold: float = 0.55) -> tuple[str, float]:
    best_s, best_c = -1.0, None
    for c, mu in cents.items():
        s = float(np.dot(vec, mu) / (np.linalg.norm(vec) * np.linalg.norm(mu) + 1e-8))
        if s > best_s:
            best_s, best_c = s, c
    return (best_c if best_s >= threshold else "unknown"), best_s

def make_svm_head() -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("svm", LinearSVC())])
