import torch
from torch.utils.data import Dataset


class SnP500_dataset(Dataset):

    def __init__(self, examples, labels, transform=None):
        """
        Args:

        """
        self.transform = transform
        self.examples = examples
        self.targets = labels

    def __len__(self):
        return self.examples.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        examples = self.examples[idx, :]
        labels = self.targets[idx, :]
        if self.transform:
            examples = self.transform(examples)
            labels = self.transform(labels)

        return examples, labels
