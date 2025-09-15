import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR

# Import all necessary components from util.py
from util_1 import (
    load_dataset, 
    store_predictions, 
    evaluate_binary_miou, 
    UNet, 
    SegmentationDataset, 
    ToTensor, 
    CrossEntropyDiceLoss, 
    visualize
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Trains the model with a validation loop.
    
    Args:
        model (nn.Module): The segmentation model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) to use.
        num_epochs (int): Number of epochs to train.
        
    Returns:
        nn.Module: The best performing model on the validation set.
    """
    best_val_miou = 0.0
    best_model_weights = model.state_dict()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, sample in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")):
            inputs = sample['input'].to(device)
            targets = sample['gt'].to(device)
            scribble_masks = sample['scribble_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if i == 0:  # only print for first batch (to avoid spam)
                import numpy as np
                print("GT unique labels:", np.unique(targets.cpu().numpy()))
                print("Scribble unique labels:", np.unique(scribble_masks.cpu().numpy()))
            
            # The WeaklySupervisedLoss only computes loss on scribbled pixels.
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1} Training Loss: {train_loss / len(train_loader):.4f}")
        
        # --- Validation Loop ---
        val_predictions, val_ground_truths = get_predictions(model, val_loader, device)
        val_miou = evaluate_binary_miou(val_predictions, val_ground_truths)
        print(f"Epoch {epoch+1} Validation mIoU: {val_miou:.3f}")
        
        scheduler.step()
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_model_weights = model.state_dict()
            print(f"New best mIoU: {best_val_miou:.4f}. Saving model weights.")

    model.load_state_dict(best_model_weights)
    return model

def get_predictions(model, data_loader, device):
    """
    Generates predictions for a dataset.
    
    Args:
        model (nn.Module): The trained segmentation model.
        data_loader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to use.
        
    Returns:
        tuple: A tuple containing a list of numpy predictions and a list of
               numpy ground truths (if available).
    """
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader, desc="Predicting")):
            inputs = sample['input'].to(device)
            outputs = model(inputs)
            
            # Get the predicted class (0 or 1) for each pixel
            pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()
            pred_mask[pred_mask == 1] = 38
            for pm in pred_mask:  # loop over batch
                predictions.append(pm)
            
            if 'gt' in sample:
                ground_truths.append(sample['gt'].squeeze().numpy())
    
    return predictions, ground_truths

if __name__ == "__main__":
    # Define paths
    training_data_dir = "dataset/train"
    test_data_dir = "dataset/test1"
    output_dir = "dataset/predictions"

    # Load data paths
    train_image_paths, train_scribble_paths, train_gt_paths = load_dataset(training_data_dir)
    test_image_paths, test_scribble_paths, _ = load_dataset(test_data_dir)

    # Split training data into training and validation sets
    split_index = int(0.8 * len(train_image_paths))
    train_paths = (train_image_paths[:split_index], train_scribble_paths[:split_index], train_gt_paths[:split_index])
    val_paths = (train_image_paths[split_index:], train_scribble_paths[split_index:], train_gt_paths[split_index:])

    # Data transformation
    transform = ToTensor()

    # Create PyTorch datasets and data loaders
    train_dataset = SegmentationDataset(*train_paths, transform=transform)
    val_dataset = SegmentationDataset(*val_paths, transform=transform)
    test_dataset = SegmentationDataset(test_image_paths, test_scribble_paths, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # in_channels=4 because we have 3 RGB channels + 1 scribble channel
    model = UNet(in_channels=4, num_classes=2).to(device)
    criterion = CrossEntropyDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    # Generate and store predictions for the test set
    test_predictions, _ = get_predictions(trained_model, test_loader, device)
    test_file_names = [os.path.basename(p) for p in test_image_paths]
    store_predictions(test_predictions, test_file_names, os.path.join(output_dir, "test1"))
    
    print("Final predictions stored. Submission to CMS is ready.")