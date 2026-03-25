import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RadarDataset(Dataset):
    def __init__(self, samples, labels=None, train=False, T=64):
        self.samples = samples
        self.labels = labels
        self.train = train
        self.T = T

    def load_rtm(self, path):
        files = os.listdir(path)
        r1 = np.load(os.path.join(path, [f for f in files if "RTM1" in f][0]))
        r2 = np.load(os.path.join(path, [f for f in files if "RTM2" in f][0]))
        r3 = np.load(os.path.join(path, [f for f in files if "RTM3" in f][0]))
        # Shape: (3, T_raw, R)
        x = np.stack([r1, r2, r3], axis=0).astype(np.float32)
        return x

    def temporal_stretch(self, x):
        if self.train and np.random.rand() < 0.35:
            factor = np.random.uniform(0.85, 1.15)
            new_t = max(8, int(round(x.shape[1] * factor)))
            old_idx = np.arange(x.shape[1])
            new_idx = np.linspace(0, x.shape[1] - 1, new_t)
            
            x_interp = np.zeros((x.shape[0], new_t, x.shape[2]), dtype=x.dtype)
            for a in range(x.shape[0]):
                for rr in range(x.shape[2]):
                    x_interp[a, :, rr] = np.interp(new_idx, old_idx, x[a, :, rr])
            return x_interp
        return x

    def pre_feature_augment(self, x):
        if self.train and np.random.rand() < 0.6:
            shift = np.random.randint(-5, 6)
            if shift > 0:
                x[:, :, shift:] = x[:, :, :-shift]
                x[:, :, :shift] = -120.0  
            elif shift < 0:
                x[:, :, :shift] = x[:, :, -shift:]
                x[:, :, shift:] = -120.0  

        # Amplitude scaling (simulates reflectivity differences)
        if self.train and np.random.rand() < 0.5:
            scale_db = np.random.uniform(-3.0, 3.0)
            x = x + scale_db
            
        return x

    def compute_features(self, x_db):
        # Convert dB to linear scale for physically accurate derivatives
        # x_db is 20*log10(A) -> A = 10^(x_db/20)
        x_lin = 10.0 ** (x_db / 20.0)
        
        # Velocity (change in amplitude over time)
        vel = np.diff(x_lin, axis=1)
        vel = np.pad(vel, ((0,0),(0,1),(0,0)), mode='constant', constant_values=0)  

        # Acceleration
        acc = np.diff(vel, axis=1)
        acc = np.pad(acc, ((0,0),(0,1),(0,0)), mode='constant', constant_values=0) 

        # Range Gradient
        rgrad = np.diff(x_lin, axis=2)
        rgrad = np.pad(rgrad, ((0,0),(0,0),(0,1)), mode='constant', constant_values=0) 

        return vel.astype(np.float32), acc.astype(np.float32), rgrad.astype(np.float32)

    def fix_temporal_multichannel(self, out, rtm_energy):
        c, t, r = out.shape

        if t <= self.T:
            pad = self.T - t
            rtm_padded = np.pad(out[:3], ((0,0),(0,pad),(0,0)), constant_values=-120.0)
            feat_padded = np.pad(out[3:], ((0,0),(0,pad),(0,0)), constant_values=0.0)
            return np.concatenate([rtm_padded, feat_padded], axis=0)

        energy = np.mean(np.abs(np.diff(rtm_energy, axis=1)), axis=(0,2))
        energy = np.pad(energy, (1,0))
        center = int(np.argmax(energy))

        if self.train:
            jitter = int(np.round(np.random.normal(0, 4)))
            center = np.clip(center + jitter, 0, t-1)

        start = max(0, center - self.T // 2)
        if start + self.T > t:
            start = t - self.T
            
        return out[:, start:start + self.T, :]

    def normalize(self, x):
        # Channel-wise normalization
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True) + 1e-6
        return (x - mean) / std

    def augment_post_norm(self, x):
        """Structural augmentations applied AFTER normalization (masking with 0.0)"""
        # Antenna dropout
        if self.train and np.random.rand() < 0.12:
            drop = np.random.choice([0,1,2])
            # Drop the specific antenna across all 4 of its feature channels
            channels_to_drop = [drop, drop+3, drop+6, drop+9]
            x[channels_to_drop] = 0.0

        # Time masking (SpecAugment)
        if self.train and np.random.rand() < 0.45:
            w = np.random.randint(3, 16)
            start = np.random.randint(0, max(1, x.shape[1] - w))
            x[:, start:start + w, :] = 0.0  # 0.0 is the mean value after standardization

        # Range masking
        if self.train and np.random.rand() < 0.45:
            w = np.random.randint(3, 20)
            start = np.random.randint(0, max(1, x.shape[2] - w))
            x[:, :, start:start + w] = 0.0

        return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        x_rtm = self.load_rtm(path)           # (3, T_raw, R)
        x_rtm = self.temporal_stretch(x_rtm)  
        x_rtm = self.pre_feature_augment(x_rtm) 
        
        vel, acc, rgrad = self.compute_features(x_rtm)
        
        out = np.concatenate([x_rtm, vel, acc, rgrad], axis=0)

        # Fix length based on energy of RTM
        out = self.fix_temporal_multichannel(out, x_rtm)
        
        # Clip only RTM channels to [-120, 0] dB
        out[:3] = np.clip(out[:3], a_min=-120.0, a_max=0.0)

        # Normalize per channel
        out = self.normalize(out)
        
        # Masking after normalization
        if self.train:
            out = self.augment_post_norm(out)

        out = torch.tensor(out, dtype=torch.float32)
        if self.labels is None:
            return out
        
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return out, y