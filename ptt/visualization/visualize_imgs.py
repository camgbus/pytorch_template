#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math

from src.data.dataset_obj import Dataset, Instance
from src.data.torcherize import TorchSegmentationDataset
from src.eval.patch_based_eval.eval import patch_based_eval
from src.eval.metrics import dice

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
        plt.subplot(nr_rows,nr_cols,i+1), plt.imshow(img[i], cmap='gray', interpolation='none'), plt.axis('off')
        plt.subplot(nr_rows,nr_cols,i+1), plt.imshow(segmentation[i], cmap='jet', interpolation='none', alpha=0.5), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_overlay_mask(img, mask, save_path=None):
    if 'torch' in str(type(img)):
        img, mask = img.cpu().detach().numpy(), mask.cpu().detach().numpy()
        while len(img.shape) > 2:
            img, mask = img[0], mask[0]
    assert img.shape == mask.shape
    plt.figure(figsize=(20, 20), frameon=False)
    plt.imshow(img, 'gray'), plt.axis('off')
    plt.imshow(mask, 'jet', alpha=0.7), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compare_masks(gt_mask, pred_mask, save_path):
    assert gt_mask.shape == pred_mask.shape
    plt.figure(figsize=(20, 20), frameon=False)
    plt.imshow(gt_mask, 'gray'), plt.axis('off')
    plt.imshow(pred_mask, 'jet', alpha=0.7), plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_prediction(agent, x_slices, y_slices, norm_name):
    # Create dataset
    instances = [Instance(x=x_slices[slice_ix], mask=y_slices[slice_ix]) for
        slice_ix in range(len(x_slices))]
    ds = Dataset(name=norm_name, instances=instances)
    ds = TorchSegmentationDataset(dataset_obj=ds, transform='crop')
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    img, ground_truth, prediction  = [], [], []
    # Get predictions
    agent.model.train(False)
    patient_dice = []
    for data_batch in dataloader:
        x, y, pred = agent.get_input_target_output(data_batch)
        pred = torch.sigmoid(pred)
        dice_loss = dice(pred, y)
        x, y, pred = x.cpu().detach().numpy(), y.cpu().detach().numpy(), pred.cpu().detach().numpy()
        patient_dice.append(dice_loss.cpu().detach().numpy())
        img.append(np.squeeze(x, axis=1)) # Remove channel dimension
        ground_truth.append(np.squeeze(y, axis=1))
        prediction.append(np.squeeze(pred, axis=1))
    img = np.concatenate(img, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    prediction = np.concatenate(prediction, axis=0)
    plot_3d_segmentation(img, prediction)
    plot_3d_segmentation(ground_truth, prediction)
    return np.mean(patient_dice)

def visualize_patch_based_prediction(agent, x_slices, y_slices, norm_name, save_path):
    pred_slices = []
    patient_dice = []
    for slice_ix in range(len(x_slices)):
        pred_mask, dice = patch_based_eval(img=x_slices[slice_ix], 
            gt_mask=y_slices[slice_ix], agent=agent, ds_name=norm_name, 
            patch_size=320, stride=100, nr_classes=1)
        pred_slices.append(pred_mask)
        patient_dice.append(dice)
    pred_slices = np.array(pred_slices)
    plot_3d_segmentation(x_slices, pred_slices, os.path.join(save_path, 'overlay.png'))
    plot_3d_segmentation(y_slices, pred_slices, os.path.join(save_path, 'mask.png'))
    return np.mean(patient_dice)
