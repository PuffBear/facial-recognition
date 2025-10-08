import numpy as np
from typing import List
from PIL import Image
import matplotlib.pyplot as plt

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

