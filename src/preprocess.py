import os
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MultiFileRoofDataset(Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=512, stride=256, augment=False, subset_size=None):
        """
        Improved dataset class that properly handles:
        - Correct path joining
        - Proper file filtering
        - Window computation
        - File pairing
        
        Args:
            img_dir: Relative path to image directory
            mask_dir: Relative path to mask directory
            patch_size: Size of square patches
            stride: Step between patches
            augment: Whether to apply data augmentation
        """
        # Proper path handling using os.path.join()
        self.img_dir = os.path.join(os.getcwd(), img_dir)
        self.mask_dir = os.path.join(os.getcwd(), mask_dir)
        
        # Get and sort files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.tif') and not f.endswith('vis.tif')])
        
        # Verify we have matching pairs
        assert len(self.img_files) == len(self.mask_files), \
            f"Mismatch: {len(self.img_files)} images vs {len(self.mask_files)} masks"
        
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.subset_size = subset_size
        self.windows = []
        
        # Precompute windows for all files
        for file_idx, (img_file, mask_file) in enumerate(zip(self.img_files, self.mask_files)):
            img_path = os.path.join(self.img_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            try:
                with rasterio.open(img_path) as src:
                    height, width = src.height, src.width
                    
                    # Generate windows for this file
                    file_windows = [
                        (file_idx, Window(x, y, patch_size, patch_size))
                        for x in range(0, width - patch_size + 1, stride)
                        for y in range(0, height - patch_size + 1, stride)
                    ]
                    self.windows.extend(file_windows)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
                
            # Randomly sample a subset of windows if subset_size is set
        if self.subset_size:
            self.windows = random.sample(self.windows, self.subset_size)
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        file_idx, window = self.windows[idx]
        img_path = os.path.join(self.img_dir, self.img_files[file_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[file_idx])
        
        try:
            with rasterio.open(img_path) as img_src, \
                 rasterio.open(mask_path) as mask_src:
                
                # Read and normalize image [C, H, W]
                img_patch = img_src.read(window=window).astype('float32') / 255.0
                assert 0 <= img_patch.min() and img_patch.max() <= 1, "Image values out of range"
                
                # Read and binarize mask
                mask_patch = mask_src.read(1, window=window) > 0
                mask_patch = mask_patch.astype('float32')
                
                if self.augment:
                    img_patch, mask_patch = self._augment(img_patch, mask_patch)
                
                return (
                    torch.from_numpy(img_patch),
                    torch.from_numpy(mask_patch).unsqueeze(0))  # Add channel dim
                
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return empty patch of correct size
            empty_img = torch.zeros((3, self.patch_size, self.patch_size))
            empty_mask = torch.zeros((1, self.patch_size, self.patch_size))
            return empty_img, empty_mask
    
    def _augment(self, img, mask):
        """Apply identical augmentation to image and mask"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation (0, 90, 180, or 270 degrees)
        rot = np.random.randint(0, 4)
        img = np.rot90(img, rot, axes=(1, 2)).copy()
        mask = np.rot90(mask, rot, axes=(0, 1)).copy()
        
        return img, mask