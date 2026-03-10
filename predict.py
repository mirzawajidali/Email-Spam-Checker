"""
Inference module for the trained BERT Spam Classifier.
Load a saved model and predict whether an email/message is spam or ham.
"""

import torch
from transformers import BertTokenizer
from model import BertSpamClassifier


class SpamPredictor:
    """Load a trained model and make predictions on new text."""

    def __init__(self, model_dir="saved_model", device=None):
        """
        Args:
            model_dir: Path to the directory containing the saved model.
            device: Device to run inference on. Auto-detects if None.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint_path = f"{model_dir}/best_model.pt"
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        bert_model_name = checkpoint["bert_model_name"]
        self.max_length = checkpoint["max_length"]

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertSpamClassifier(bert_model_name=bert_model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded (trained accuracy: {checkpoint['accuracy']:.4f})")

    def predict(self, text):
        """
        Predict if a single text is spam or ham.

        Args:
            text: The email/message text to classify.

        Returns:
            dict with 'label' ('spam'/'ham'), 'confidence', and 'probabilities'.
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "spam" if pred == 1 else "ham"

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "ham": probs[0][0].item(),
                "spam": probs[0][1].item(),
            },
        }

    def predict_batch(self, texts):
        """Predict labels for a list of texts."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    predictor = SpamPredictor()

    test_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "CONGRATULATIONS! You've won a $1000 gift card! Click here to claim NOW!",
        "Please find the attached report for Q3 earnings.",
        "FREE entry to win a brand new iPhone! Text WIN to 80085 now!",
        "Can you review the pull request I sent yesterday?",
        "URGENT: Your account has been compromised. Click this link to verify your identity.",
    ]

    print("\n" + "=" * 70)
    print("SPAM DETECTION RESULTS")
    print("=" * 70)

    for msg in test_messages:
        result = predictor.predict(msg)
        status = "SPAM" if result["label"] == "spam" else "HAM "
        conf = result["confidence"]
        print(f"\n[{status}] (confidence: {conf:.2%})")
        print(f"  > {msg[:80]}...")
