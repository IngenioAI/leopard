import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(logits, labels, save_dir):
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(logits)

    plt.figure(figsize=(8, 6))
    for class_label in np.unique(labels):
        class_mask = (labels == class_label)
        plt.scatter(X_embedded[class_mask, 0], X_embedded[class_mask, 1], label=f'Class {class_label}', alpha=0.6)

    plt.title("t-SNE")
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_dir}/t-sne.png')
    plt.close()