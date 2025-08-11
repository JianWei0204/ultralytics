"""
Domain Adaptation Data Handling
This module provides data loaders and utilities for domain adaptation training
that can handle both source and target domain data simultaneously.
"""

import random
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

__all__ = "DomainAwareTrainerMixin"


class DomainAdaptationDataset(Dataset):
    """
    Combined dataset for domain adaptation that provides both source and target domain samples.
    """

    def __init__(self, source_dataset, target_dataset, mode='alternating'):
        """
        Initialize domain adaptation dataset.

        Args:
            source_dataset: Source domain dataset
            target_dataset: Target domain dataset  
            mode: How to combine datasets ('alternating', 'random', 'balanced')
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.mode = mode

        # Calculate combined length
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)

        if mode == 'balanced':
            # Use the larger dataset length and repeat smaller one
            self.length = max(self.source_len, self.target_len)
        else:
            # Use sum of both datasets
            self.length = self.source_len + self.target_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'alternating':
            # Alternate between source and target
            if idx % 2 == 0:
                # Source domain
                source_idx = (idx // 2) % self.source_len
                sample = self.source_dataset[source_idx]
                sample['domain'] = 'source'
                sample['domain_label'] = 0
            else:
                # Target domain
                target_idx = ((idx - 1) // 2) % self.target_len
                sample = self.target_dataset[target_idx]
                sample['domain'] = 'target'
                sample['domain_label'] = 1

        elif self.mode == 'random':
            # Randomly choose between source and target
            if random.random() < 0.5:
                source_idx = random.randint(0, self.source_len - 1)
                sample = self.source_dataset[source_idx]
                sample['domain'] = 'source'
                sample['domain_label'] = 0
            else:
                target_idx = random.randint(0, self.target_len - 1)
                sample = self.target_dataset[target_idx]
                sample['domain'] = 'target'
                sample['domain_label'] = 1

        elif self.mode == 'balanced':
            # Balanced sampling from both domains
            if idx < self.source_len:
                sample = self.source_dataset[idx]
                sample['domain'] = 'source'
                sample['domain_label'] = 0
            else:
                target_idx = idx % self.target_len
                sample = self.target_dataset[target_idx]
                sample['domain'] = 'target'
                sample['domain_label'] = 1

        return sample


class DomainBatchSampler:
    """
    Custom batch sampler that ensures each batch contains both source and target samples.
    """

    def __init__(self, source_dataset, target_dataset, batch_size, shuffle=True):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)

        # Calculate batches per epoch
        self.batches_per_epoch = max(
            (self.source_len + batch_size - 1) // batch_size,
            (self.target_len + batch_size - 1) // batch_size
        )

    def __iter__(self):
        # Create indices for source and target
        source_indices = list(range(self.source_len))
        target_indices = list(range(self.target_len))

        if self.shuffle:
            random.shuffle(source_indices)
            random.shuffle(target_indices)

        # Create batches
        for batch_idx in range(self.batches_per_epoch):
            batch = []

            # Add source samples to batch
            start_idx = batch_idx * (self.batch_size // 2)
            end_idx = start_idx + (self.batch_size // 2)

            for i in range(start_idx, end_idx):
                if i < len(source_indices):
                    batch.append(('source', source_indices[i]))
                else:
                    # Wrap around if we run out of source samples
                    batch.append(('source', source_indices[i % len(source_indices)]))

            # Add target samples to batch
            for i in range(start_idx, end_idx):
                if i < len(target_indices):
                    batch.append(('target', target_indices[i]))
                else:
                    # Wrap around if we run out of target samples
                    batch.append(('target', target_indices[i % len(target_indices)]))

            yield batch

    def __len__(self):
        return self.batches_per_epoch


def create_domain_adaptation_dataloader(source_dataset, target_dataset, batch_size,
                                        num_workers=0, shuffle=True, mode='alternating'):
    """
    Create a data loader for domain adaptation.

    Args:
        source_dataset: Source domain dataset
        target_dataset: Target domain dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        mode: How to combine datasets

    Returns:
        DataLoader for domain adaptation
    """
    combined_dataset = DomainAdaptationDataset(source_dataset, target_dataset, mode=mode)

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=domain_collate_fn,
        pin_memory=True
    )


def domain_collate_fn(batch):
    """
    Custom collate function for domain adaptation batches.

    Separates source and target domain samples and creates appropriate batch structure.
    """
    source_samples = []
    target_samples = []

    for sample in batch:
        if sample['domain'] == 'source':
            source_samples.append(sample)
        else:
            target_samples.append(sample)

    # Create separate batches for source and target
    source_batch = None
    target_batch = None

    if source_samples:
        source_batch = default_collate_fn(source_samples)

    if target_samples:
        target_batch = default_collate_fn(target_samples)

    return {
        'source': source_batch,
        'target': target_batch
    }


def default_collate_fn(samples):
    """Default collate function for a list of samples."""
    if not samples:
        return None

    # This is a simplified collate function
    # In practice, you would use the YOLO-specific collate function
    batch = {}

    # Collect all keys from samples
    keys = set()
    for sample in samples:
        keys.update(sample.keys())

    # Collate each key
    for key in keys:
        values = [sample.get(key) for sample in samples if key in sample]

        if key == 'img':
            # Stack images
            batch[key] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values])
        elif key in ['cls', 'bboxes', 'batch_idx']:
            # Concatenate labels
            if values and values[0] is not None:
                batch[key] = torch.cat([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values])
        elif key == 'domain_label':
            # Stack domain labels
            batch[key] = torch.tensor(values)
        else:
            # Keep as list for other keys
            batch[key] = values

    return batch


class DomainAwareTrainerMixin:
    """
    Mixin class to add domain adaptation data handling to existing trainers.
    """

    def get_domain_dataloader(self, source_dataset_path, target_dataset_path,
                              batch_size=16, rank=0, mode="train"):
        """
        Create domain adaptation data loader.

        Args:
            source_dataset_path: Path to source domain dataset
            target_dataset_path: Path to target domain dataset
            batch_size: Batch size
            rank: Process rank for distributed training
            mode: Training mode

        Returns:
            Domain adaptation data loader
        """
        # Build source and target datasets separately
        source_dataset = self.build_dataset(source_dataset_path, mode, batch_size)
        target_dataset = self.build_dataset(target_dataset_path, mode, batch_size)

        # Create combined domain adaptation dataloader
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        return create_domain_adaptation_dataloader(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=(mode == "train"),
            mode='alternating'
        )

    def split_domain_batch(self, batch):
        """
        Split a domain adaptation batch into source and target components.

        Args:
            batch: Combined domain batch

        Returns:
            Tuple of (source_batch, target_batch)
        """
        if isinstance(batch, dict) and 'source' in batch and 'target' in batch:
            return batch['source'], batch['target']

        # Fallback: split based on domain labels if available
        if 'domain_label' in batch:
            domain_labels = batch['domain_label']
            source_mask = domain_labels == 0
            target_mask = domain_labels == 1

            source_batch = {
                k: v[source_mask] if isinstance(v, torch.Tensor) else [v[i] for i in range(len(v)) if source_mask[i]]
                for k, v in batch.items()}
            target_batch = {
                k: v[target_mask] if isinstance(v, torch.Tensor) else [v[i] for i in range(len(v)) if target_mask[i]]
                for k, v in batch.items()}

            return source_batch, target_batch

        # If no domain information, treat entire batch as source
        return batch, None