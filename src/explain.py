import numpy as np
from typing import List
from PIL import Image
import matplotlib.pyplot as plt
import torch

def save_eigenfaces_components(pca, outpath: str, image_shape: tuple[int,int], n_components: int = 16):
    comps = pca.components_[:n_components]
    h, w = image_shape
    cols = int(np.ceil(np.sqrt(n_components)))
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        ax.axis('off')
        if i < len(comps):
            img = comps[i].reshape(h, w)
            ax.imshow(img, cmap='gray')
    fig.suptitle('Top PCA Components (Eigenfaces)')
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def save_lbp_heatmap(lbp_image: np.ndarray, outpath: str):
    plt.figure(figsize=(3,3))
    plt.imshow(lbp_image, cmap='hot')
    plt.axis('off')
    plt.title('LBP Response')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def gradient_saliency(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    x = input_tensor.clone().detach().requires_grad_(True)
    out = model(x)
    scalar = out.norm()
    model.zero_grad()
    scalar.backward()
    grad = x.grad.detach().cpu().numpy()[0]
    sal = np.abs(grad).mean(axis=0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal

def overlay_heatmap(pil_img: Image.Image, heatmap: np.ndarray, outpath: str, alpha: float = 0.4):
    hm = np.uint8(255 * heatmap)
    hm = Image.fromarray(hm).resize(pil_img.size, Image.NEAREST)
    hm = np.array(plt.cm.jet(np.array(hm)/255.0))[:, :, :3]
    base = np.array(pil_img).astype(np.float32)/255.0
    over = (1-alpha)*base + alpha*hm
    over = np.clip(over*255.0, 0, 255).astype(np.uint8)
    Image.fromarray(over).save(outpath)

