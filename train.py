"""
Training and evaluation module for the BERT Spam Classifier.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model import BertSpamClassifier
from dataset import load_spam_data, load_custom_csv, create_data_loaders


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    csv_path=None,
    bert_model_name="bert-base-uncased",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    max_length=256,
    freeze_bert=False,
    save_dir="saved_model",
):
    """
    Full training pipeline.

    Args:
        csv_path: Path to custom CSV dataset (optional). If None, uses HF 'sms_spam'.
        bert_model_name: Pretrained BERT model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for AdamW optimizer.
        max_length: Max token sequence length.
        freeze_bert: Whether to freeze BERT layers.
        save_dir: Directory to save the trained model.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {bert_model_name}")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Load data
    if csv_path:
        train_texts, test_texts, train_labels, test_labels = load_custom_csv(csv_path)
    else:
        train_texts, test_texts, train_labels, test_labels = load_spam_data()

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_texts, test_texts, train_labels, test_labels,
        tokenizer, max_length=max_length, batch_size=batch_size,
    )

    # Initialize model
    print("Initializing BERT Spam Classifier...")
    model = BertSpamClassifier(
        bert_model_name=bert_model_name,
        dropout_rate=0.3,
        freeze_bert=freeze_bert,
    )
    model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    best_accuracy = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, test_loader, criterion, device
        )
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "bert_model_name": bert_model_name,
                "max_length": max_length,
                "accuracy": val_acc,
                "epoch": epoch + 1,
            }, os.path.join(save_dir, "best_model.pt"))
            tokenizer.save_pretrained(save_dir)
            print(f"* Saved best model (accuracy: {val_acc:.4f})")

    # Final evaluation with detailed metrics
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    _, final_acc, final_preds, final_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nAccuracy: {final_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        final_labels, final_preds, target_names=["Ham", "Spam"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))

    print(f"\nModel saved to: {save_dir}/")
    return model, tokenizer


if __name__ == "__main__":
    train_model()
