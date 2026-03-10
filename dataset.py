"""
Dataset module for loading and preprocessing spam email data.
Uses the Hugging Face 'sms_spam' dataset as a starting point.
You can replace this with any email spam dataset (CSV with 'text' and 'label' columns).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split


class SpamDataset(Dataset):
    """PyTorch Dataset for spam classification with BERT tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_spam_data(dataset_name="sms_spam", test_size=0.2, random_state=42):
    """
    Load spam dataset from Hugging Face Hub.

    Args:
        dataset_name: Name of the dataset on HF Hub. Default is 'sms_spam'.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name)

    texts = dataset["train"]["sms"]
    labels = dataset["train"]["label"]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples:  {len(test_texts)}")
    print(f"Spam ratio (train): {sum(train_labels) / len(train_labels):.2%}")

    return train_texts, test_texts, train_labels, test_labels


def load_custom_csv(csv_path, text_col="text", label_col="label", test_size=0.2, random_state=42):
    """
    Load a custom CSV dataset with text and label columns.

    Args:
        csv_path: Path to CSV file.
        text_col: Name of the text column.
        label_col: Name of the label column (0=ham, 1=spam).
        test_size: Fraction for test split.
        random_state: Random seed.

    Returns:
        train_texts, test_texts, train_labels, test_labels
    """
    import pandas as pd

    print(f"Loading custom dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples:  {len(test_texts)}")

    return train_texts, test_texts, train_labels, test_labels


def create_data_loaders(
    train_texts, test_texts, train_labels, test_labels,
    tokenizer, max_length=256, batch_size=16
):
    """Create PyTorch DataLoaders for training and testing."""
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = SpamDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
