import numpy as np

class Normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=5):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        
        # Variables to track running statistics
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        
        # The actual mean and std used for normalization
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        """Update the running statistics with new data."""
        v = v.reshape(-1, self.size)
        
        self.local_sum += v.sum(axis=0)
        self.local_sumsq += (np.square(v)).sum(axis=0)
        self.local_count[0] += v.shape[0]
        
        # Re-calculate mean and std
        self.mean = self.local_sum / self.local_count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.local_sumsq / self.local_count) - np.square(self.mean)))

    def normalize(self, v):
        """Normalize the input vector v using current mean and std."""
        return np.clip((v - self.mean) / self.std, -self.default_clip_range, self.default_clip_range).astype(np.float32)
    