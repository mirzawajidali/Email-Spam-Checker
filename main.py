"""
Main entry point for the BERT Spam Email Classifier.

Usage:
    Train:    python main.py train
    Predict:  python main.py predict "Your message here"
    Demo:     python main.py demo
"""

import argparse
import sys

from train import train_model
from predict import SpamPredictor


def main():
    parser = argparse.ArgumentParser(description="BERT Spam Email Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the spam classifier")
    train_parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV dataset")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    train_parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    train_parser.add_argument("--freeze-bert", action="store_true", help="Freeze BERT layers")
    train_parser.add_argument("--save-dir", type=str, default="saved_model", help="Model save directory")
    train_parser.add_argument(
        "--bert-model", type=str, default="bert-base-uncased",
        help="Pretrained BERT model name",
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict spam/ham for a message")
    predict_parser.add_argument("text", type=str, help="Text to classify")
    predict_parser.add_argument("--model-dir", type=str, default="saved_model", help="Model directory")

    # Demo command
    subparsers.add_parser("demo", help="Run demo predictions on sample messages")

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            csv_path=args.csv,
            bert_model_name=args.bert_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            freeze_bert=args.freeze_bert,
            save_dir=args.save_dir,
        )

    elif args.command == "predict":
        predictor = SpamPredictor(model_dir=args.model_dir)
        result = predictor.predict(args.text)
        print(f"\nResult: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Ham probability:  {result['probabilities']['ham']:.4f}")
        print(f"Spam probability: {result['probabilities']['spam']:.4f}")

    elif args.command == "demo":
        predictor = SpamPredictor()
        messages = [
            "Hey, are we still meeting for lunch tomorrow?",
            "CONGRATULATIONS! You've won a $1000 gift card! Click here NOW!",
            "Please review the attached document before our meeting.",
            "FREE entry to win iPhone! Text WIN to 80085!",
            "Your Amazon order #12345 has been shipped.",
            "URGENT: Verify your bank account immediately or it will be closed!",
        ]
        print("\n" + "=" * 70)
        for msg in messages:
            result = predictor.predict(msg)
            tag = "SPAM" if result["label"] == "spam" else "HAM "
            print(f"[{tag}] ({result['confidence']:.0%}) {msg}")
        print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
