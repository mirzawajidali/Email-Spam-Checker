"""
BERT-based Spam Classifier Model.

Fine-tunes a pretrained BERT model for binary classification (spam vs ham).
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BertSpamClassifier(nn.Module):
    """
    BERT-based binary classifier for spam detection.

    Architecture:
        BERT encoder -> Dropout -> Fully Connected -> Sigmoid
    """

    def __init__(self, bert_model_name="bert-base-uncased", dropout_rate=0.3, freeze_bert=False):
        """
        Args:
            bert_model_name: Pretrained BERT model name from Hugging Face.
            dropout_rate: Dropout probability for regularization.
            freeze_bert: If True, freeze BERT weights and only train the classifier head.
        """
        super(BertSpamClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),  # 2 classes: ham (0) and spam (1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            logits: Raw predictions [batch_size, 2]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        return logits
