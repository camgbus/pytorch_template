# ------------------------------------------------------------------------------
# Visualize images and tensors.
# ------------------------------------------------------------------------------

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math

def plot_3d_img(img, save_path=None):
    """
    :param img: SimpleITK image or numpy array
    """
    if 'SimpleITK.SimpleITK.Image' in str(type(img)):
        img = sitk.GetArrayFromImage(img)
    assert len(img.shape) == 3
    assert img.shape[1] == img.shape[2]
    nr_slices = len(img)
    nr_cols=8
    nr_rows=int(math.ceil(nr_slices/nr_cols))
    plt.figure(figsize=(nr_cols*3,nr_rows*3))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(img.shape[0]):
        plt.subplot(nr_rows,nr_cols,i+1), plt.imshow(img[i]), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
  
def plot_3d_segmentation(img, segmentation, save_path=None):
    """
    :param img: SimpleITK image or numpy array
    """
    if 'SimpleITK.SimpleITK.Image' in str(type(img)):
        img = sitk.GetArrayFromImage(img)
        segmentation = sitk.GetArrayFromImage(segmentation)
    assert len(img.shape) == 3
    assert img.shape[1] == img.shape[2] # Channels first
    assert img.shape == segmentation.shape
    nr_slices = len(segmentation)
    nr_cols=8
    nr_rows=int(math.ceil(nr_slices/nr_cols))
    plt.figure(figsize=(nr_cols*3,nr_rows*3))
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(img.shape[0]):
        # TODO: Background image ends up blue
        plt.subplot(nr_rows,nr_cols,i+1)
        plt.imshow(img[i], cmap='gray', interpolation='none'), plt.axis('off')
        plt.subplot(nr_rows,nr_cols,i+1)
        plt.imshow(segmentation[i], cmap='jet', interpolation='none', alpha=0.5), 
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_overlay_mask(img, mask, save_path=None, figsize=(20, 20)):
    """
    Compare two 2d imgs, one on top of the other.
    """
    if 'torch' in str(type(img)):
        img, mask = img.cpu().detach().numpy(), mask.cpu().detach().numpy()
        while len(img.shape) > 2:
            img, mask = img[0], mask[0]
    assert img.shape == mask.shape
    plt.figure(figsize=figsize, frameon=False)
    plt.imshow(img, 'gray'), plt.axis('off')
    plt.imshow(mask, 'jet', alpha=0.7), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_2d_img(img, save_path=None, figsize=(20, 20)):
    assert len(img.shape) == 3
    # If channels first, rotate so channels last
    if np.argpartition(img.shape, 1)[0] == 0:
        img = np.moveaxis(img, 0, 2)
    # Plot
    plt.figure(figsize=figsize, frameon=False)
    plt.imshow(img, 'gray'), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_dataloader(dataloader, grid_size=(5, 5), save_path=None, img_size=(512, 512)):
    imgs = get_imgs_from_dataloader(dataloader, grid_size[0]*grid_size[1])
    img_grid = get_img_grid(imgs, grid_size[0], grid_size[1])
    create_img_grid(img_grid=img_grid, save_path=save_path, img_size=img_size)

def get_imgs_from_dataloader(dataloader, nr_imgs):
    imgs = []
    for x, y in dataloader:
        x = x.cpu().detach().numpy()
        for img in x:
            if len(imgs) < nr_imgs:
                imgs.append(img)
        if len(imgs) == nr_imgs:
            break  
    return imgs  

import random
def get_img_grid(img_list, nr_rows, nr_cols, randomize=False):
    if randomize:
        random.shuffle(img_list)
    img_grid = [[img_list[i+j*nr_cols] for i in range(nr_cols)] for j in range(nr_rows)]
    return img_grid

import sys

from PIL import Image
def create_img_grid(img_grid = [[]], img_size = (512, 512), 
    margin = (5, 5), background_color = (255, 255, 255, 255), save_path=None):
    bg_width = len(img_grid[0])*img_size[0] + (len(img_grid[0])+1)*margin[0]
    bg_height = len(img_grid)*img_size[1] + (len(img_grid)+1)*margin[1]
    new_img = Image.new('RGBA', (bg_width, bg_height), background_color)
    left = margin[0]
    top = margin[1]
    for row in img_grid:
        for img in row:
            if np.argpartition(img.shape, 1)[0] == 0:
                img = np.moveaxis(img, 0, 2) 
            img = Image.fromarray((img * 255).astype(np.uint8)).resize(img_size).convert('RGB')
            new_img.paste(img, (left, top))
            left += img_size[0] + margin[0]
        top += img_size[1] + margin[1]
        left = margin[0]
    if save_path is None:
        new_img.show()
    else:
        new_img.save(save_path)

