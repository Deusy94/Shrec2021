import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_depth(path, depth):
    """Plot a single depth map

    Attributes:
        path (str): Path to save the depth map
        depth (np.ndarray): Depth map data

    """
    if len(depth.shape) > 2:
        if depth.shape[-1] != 1:
            raise ValueError("Wrong number of channel, 1 is required, got {}".format(depth.shape))
        else:
            depth = depth.squeeze()
    tmp = np.zeros((depth.shape[0], depth.shape[1], 3))
    tmp[..., 0] = depth.copy()
    tmp[..., 1] = depth.copy()
    tmp[..., 2] = depth.copy()
    tmp = ((tmp * 255) / tmp.max()).astype(np.uint8) if tmp.max() > 0 else np.zeros((depth.shape[0], depth.shape[1], 3))
    cv2.imwrite(path, tmp)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.get_cmap('Blues')):
    # plt.figure(figsize=(18, 18))
    plt.figure(figsize=(36, 36))
    # plt.title(title)
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes, fontsize=36)
    plt.yticks(tick_marks, classes, fontsize=36)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    if normalize:
        cm = (cm.T / cm.T.sum(axis=0)).T
        cm = np.nan_to_num(cm)
        fmt = '%.2f'
    else:
        fmt = '%d'
    # plt.pcolormesh(cm, edgecolors='k', linewidth=1)
    ax = plt.gca()

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    offset = 0.5
    height, width = cm.shape
    ax.hlines(y=np.arange(height + 1) - offset, xmin=-offset, xmax=width - offset, colors="#A0A0A0")
    ax.vlines(x=np.arange(width + 1) - offset, ymin=-offset, ymax=height - offset, colors="#A0A0A0")
    # plt.colorbar()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=1)

    cbar = plt.colorbar(im, cax=cax)
    # cbar = plt.colorbar(fraction=0.04575, pad=0.04)
    cbar.ax.tick_params(labelsize=28)
    thresh = cm.max() / 2.
    font = {'family': 'normal',
            'size': 28}
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if cm[i, j] in [0, 1]:
    #             plt.text(j, i, "{}".format(int(cm[i, j])),
    #                      ha="center", va="center",
    #                      color="white" if cm[i, j] > thresh else "black", fontdict=font)
    #         else:
    #             plt.text(j, i, "{}".format(cm[i, j], fmt),
    #                     ha="center", va="center",
    #                     color="white" if cm[i, j] > thresh else "black", fontdict=font)ù
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            else:
                ax.text(j, i, cm[i, j],
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontdict=font)
    # plt.savefig(title+".svg", format="svg")
    plt.savefig(title+".pdf")