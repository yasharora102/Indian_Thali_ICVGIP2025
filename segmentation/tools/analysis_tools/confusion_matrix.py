# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist, progressbar
from PIL import Image

from mmseg.registry import DATASETS

init_default_scope('mmseg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test folder result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='winter',
        help='theme of the matrix color map')
    parser.add_argument(
        '--title',
        default='Normalized Confusion Matrix',
        help='title of the matrix color map')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in config file')
    parser.add_argument(
        '--resize',
        nargs=2,
        type=int,
        metavar=('width', 'height'),
        help='resize prediction images to the given width and height before processing')
    # New argument: Comma separated list of classes
    parser.add_argument(
        '--avail',
        type=str,
        default=None,
        help='Comma separated list of available classes to use for the confusion matrix')
    args = parser.parse_args()
    return args


def confusion_matrix_for_image(dataset, pred_arr, idx):
    """Calculate confusion matrix for a single prediction image."""
    classes_all = dataset.METAINFO['classes']
    n = len(classes_all)
    gt_segm = dataset[idx]['data_samples'].gt_sem_seg.data.squeeze().numpy().astype(np.uint8)
    
    # Ensure pred_arr has the same shape as the ground truth segmentation.
    if pred_arr.shape != gt_segm.shape:
        # Use nearest-neighbor interpolation to avoid altering label values.
        pred_arr = np.array(Image.fromarray(pred_arr).resize(gt_segm.shape[::-1], Image.NEAREST))
    
    gt_segm, pred_arr = gt_segm.flatten(), pred_arr.flatten()
    if dataset.reduce_zero_label:
        gt_segm = gt_segm - 1
    to_ignore = gt_segm == dataset.ignore_index
    gt_segm, pred_arr = gt_segm[~to_ignore], pred_arr[~to_ignore]
    inds = n * gt_segm + pred_arr
    mat = np.bincount(inds, minlength=n**2).reshape(n, n)
    return mat


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='OrRd'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = (confusion_matrix.astype(np.float32) /
                        per_label_sums * 100) if per_label_sums.max() > 0 else confusion_matrix

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=300)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    colorbar = plt.colorbar(mappable=im, ax=ax)
    colorbar.ax.tick_params(labelsize=20)  # 设置 colorbar 标签的字体大小

    title_font = {'weight': 'bold', 'size': 20}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 40}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(round(confusion_matrix[i, j], 2)
                             if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='k',
                size=20)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        mkdir_or_exist(save_dir)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # If available classes are provided, update the dataset METAINFO.
    if args.avail is not None:
        classes_all = [cls.strip() for cls in args.avail.split(',')]
        dataset.METAINFO['classes'] = classes_all
    else:
        classes_all = dataset.METAINFO['classes']
        
    print(f'Using classes: {classes_all}')
    
    n = len(classes_all)
    overall_confusion = np.zeros((n, n))
    files = sorted(os.listdir(args.prediction_path))
    prog_bar = progressbar.ProgressBar(len(files))

    for idx, fname in enumerate(files):
        img_path = os.path.join(args.prediction_path, fname)
        image = Image.open(img_path)
        if args.resize is not None:
            image = image.resize((args.resize[0], args.resize[1]), Image.BILINEAR)
        pred_arr = np.array(image)
        overall_confusion += confusion_matrix_for_image(dataset, pred_arr, idx)
        prog_bar.update()
    print(overall_confusion)
    print("calculate confusion matrix done")
    plot_confusion_matrix(
        overall_confusion,
        classes_all,
        save_dir=args.save_dir,
        show=args.show,
        title=args.title,
        color_theme=args.color_theme)


if __name__ == '__main__':
    main()
