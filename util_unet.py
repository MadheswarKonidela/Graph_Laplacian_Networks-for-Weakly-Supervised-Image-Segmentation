import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F

# --- Dataset Handling ---
def load_dataset(data_dir):
    """
    Load dataset file paths.
    
    Args:
        data_dir (str): Path to the dataset directory.
        
    Returns:
        tuple: A tuple of lists containing image, scribble, and ground truth paths.
    """
    image_dir = os.path.join(data_dir, 'images')
    scribble_dir = os.path.join(data_dir, 'scribbles')
    gt_dir = os.path.join(data_dir, 'ground_truth') if os.path.isdir(os.path.join(data_dir, 'ground_truth')) else None

    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    scribble_paths = sorted([os.path.join(scribble_dir, f) for f in os.listdir(scribble_dir)])
    gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]) if gt_dir else None

    return image_paths, scribble_paths, gt_paths

def store_predictions(predictions, file_names, output_dir):
    """
    Save predicted segmentation masks as PNG files.
    
    Args:
        predictions (list): A list of numpy arrays containing the predicted masks.
        file_names (list): A list of file names for the predictions.
        output_dir (str): The directory to save the predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    for pred, file_name in zip(predictions, file_names):
        pred_img = Image.fromarray((pred * 255).astype(np.uint8))
        pred_img.putpalette([0, 0, 0, 255, 255, 255])
        pred_img.save(os.path.join(output_dir, file_name))
    print(f"Stored {len(predictions)} predictions in {output_dir}")

# --- Deep Learning Components ---

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super(UNet, self).__init__()
        self.encoder1 = self.contracting_block(in_channels, 64)
        self.encoder2 = self.contracting_block(64, 128)
        self.encoder3 = self.contracting_block(128, 256)
        self.encoder4 = self.contracting_block(256, 512)
        self.bottleneck = self.contracting_block(512, 1024)

        self.upconv4 = self.expansive_block(1024, 512)
        self.decoder4 = self.contracting_block(1024, 512)
        self.upconv3 = self.expansive_block(512, 256)
        self.decoder3 = self.contracting_block(512, 256)
        self.upconv2 = self.expansive_block(256, 128)
        self.decoder2 = self.contracting_block(256, 128)
        self.upconv1 = self.expansive_block(128, 64)
        self.decoder1 = self.contracting_block(128, 64)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def expansive_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, F.interpolate(enc4, size=dec4.shape[2:], mode="bilinear", align_corners=False)), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, F.interpolate(enc3, size=dec3.shape[2:], mode="bilinear", align_corners=False)), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, F.interpolate(enc2, size=dec2.shape[2:], mode="bilinear", align_corners=False)), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, F.interpolate(enc1, size=dec1.shape[2:], mode="bilinear", align_corners=False)), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.output_conv(dec1)

        # Ensure output size matches input size
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        return output


# Custom Dataset Class
class SegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading the images, scribbles, and ground truth.
    """
    def __init__(self, image_paths, scribble_paths, gt_paths=None, transform=None):
        self.image_paths = image_paths
        self.scribble_paths = scribble_paths
        self.gt_paths = gt_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        scribble = Image.open(self.scribble_paths[idx]).convert('L')
        
        image = np.array(image)
        scribble = np.array(scribble)

        scribble_mask = scribble.astype(np.int64)

        input_data = np.concatenate([image, scribble_mask[:, :, np.newaxis]], axis=2)

        if self.gt_paths is not None:
            ground_truth = Image.open(self.gt_paths[idx]).convert('L')
            ground_truth = np.array(ground_truth, dtype=np.int64)
            ground_truth[ground_truth == 38] = 1
            ground_truth[ground_truth == 0] = 0
            sample = {'input': input_data, 'gt': ground_truth, 'scribble_mask': scribble_mask}
        else:
            sample = {'input': input_data, 'scribble_mask': scribble_mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Data Augmentation Transforms
class PadToMultiple(object):
    """Pads an image and its masks to a multiple of a given number."""
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, sample):
        image = sample['input']
        h, w = image.shape[0], image.shape[1]
        
        pad_h = (self.multiple - h % self.multiple) % self.multiple
        pad_w = (self.multiple - w % self.multiple) % self.multiple
        
        # Pad the image, scribble mask, and ground truth mask
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        sample['scribble_mask'] = np.pad(sample['scribble_mask'], ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)
        
        if 'gt' in sample:
            sample['gt'] = np.pad(sample['gt'], ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255) # Use -1 for padding
        
        sample['input'] = image
        
        return sample

class ToTensor(object):
    """Convert numpy arrays in sample to PyTorch Tensors."""
    def __call__(self, sample):
        image = sample['input']
        image = image.transpose((2, 0, 1)) # HWC to CHW
        
        output = {'input': torch.from_numpy(image).float()}
        
        if 'gt' in sample:
            ground_truth = sample['gt']
            output['gt'] = torch.from_numpy(ground_truth).long()
        
        output['scribble_mask'] = torch.from_numpy(sample['scribble_mask']).long()
        return output

# Loss Function
class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation (with ignore index support).
    """
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets, smooth=1e-6):
        num_classes = outputs.shape[1]

        # Mask out ignore_index
        valid_mask = (targets != self.ignore_index)

        # Clamp targets into valid range [0, num_classes-1]
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0  # placeholder for ignored pixels
        safe_targets = torch.clamp(safe_targets, 0, num_classes-1)

        # One-hot encode valid targets
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Zero out ignored pixels in one-hot mask
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)

        # Softmax on outputs
        probas = F.softmax(outputs, dim=1)

        # Intersection & Union
        intersection = torch.sum(probas * targets_one_hot, dim=(2, 3))
        union = torch.sum(probas, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1. - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss using both scribbles and GT masks.
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, scribble_weight=0.5, gt_weight=1.0, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.scribble_weight = scribble_weight
        self.gt_weight = gt_weight

    def forward(self, outputs, targets=None):
        total_loss = 0.0
        
        # --- Loss on full GT if available ---
        if targets is not None:
            ce_loss_gt = self.ce(outputs, targets)
            dice_loss_gt = self.dice(outputs, targets)
            total_loss += self.gt_weight * (self.ce_weight * ce_loss_gt + self.dice_weight * dice_loss_gt)


        return total_loss

# Evaluation
def evaluate_binary_miou(predictions, ground_truths):
    """
    Computes the mean Intersection over Union (mIoU) for binary segmentation.
    
    Args:
        predictions (list): A list of numpy arrays with predicted masks (0 or 1).
        ground_truths (list): A list of numpy arrays with ground truth masks (0 or 1).
        
    Returns:
        float: The mean IoU score.
    """
    ious = []
    for pred, gt in zip(predictions, ground_truths):
        for cls in range(2): # 0 for background, 1 for foreground
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union == 0:
                iou = 1.0 if np.all(~pred_mask) and np.all(~gt_mask) else 0.0
            else:
                iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious)

# Visualization
def visualize(image, scribble, ground_truth, prediction):
    """
    Displays the image, scribbles, ground truth, and predicted mask.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(scribble, cmap='gray')
    axes[1].set_title('Scribbles (0:B, 1:F, 255:U)')
    axes[1].axis('off')

    axes[2].imshow(ground_truth, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    axes[3].imshow(prediction, cmap='gray')
    axes[3].set_title('Prediction')
    axes[3].axis('off')
    
    plt.show()