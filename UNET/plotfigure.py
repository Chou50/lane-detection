import matplotlib.pyplot as plt
import numpy as np

"""
define the function of plotting figure.
Before plotting, images data need to be transformed back to the range of (0, 255). 
It is reverse transformation of normalization.
"""

def reverse_transform(inp):
    # after ToTensor, figure shape is (H, W, 3), so need to be transposed to (3, H, W) to be plotted
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # clip to 0-1
    inp = (inp * 255).astype(np.uint8)
    return inp


def plot_figure(image, mask):
    img = reverse_transform(image)
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    # get the index of (value == 1) on mask
    mask = mask.numpy()
    mask = (mask != 0).astype(np.uint8)
    mask_idx = np.where(mask[:, :, 0])
    img[mask_idx[0], mask_idx[1], :] = [217, 22, 207]
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()



