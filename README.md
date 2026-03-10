# Spam Email Detector

A BERT-based spam email classifier built with PyTorch. Fine-tunes `bert-base-uncased` for binary classification (spam vs not spam) and includes a Streamlit web UI for easy interaction.

## Project Structure

```
spamchecker/
├── app.py             # Streamlit web UI
├── dataset.py         # Data loading & preprocessing
├── model.py           # BERT spam classifier model
├── train.py           # Training & evaluation loop
├── predict.py         # Inference module
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
└── saved_model/       # Trained model checkpoint (generated after training)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit
```

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Train the model

```bash
python main.py train
```

This downloads the SMS Spam dataset from Hugging Face and fine-tunes BERT on it.

**Training options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Number of training epochs | 3 |
| `--batch-size` | Batch size | 16 |
| `--lr` | Learning rate | 2e-5 |
| `--max-length` | Max token sequence length | 256 |
| `--freeze-bert` | Freeze BERT layers (faster training) | False |
| `--csv` | Path to custom CSV dataset | None |
| `--bert-model` | Pretrained BERT model name | bert-base-uncased |

**Train with custom dataset:**

```bash
python main.py train --csv your_data.csv --epochs 5
```

CSV must have `text` and `label` columns (0 = not spam, 1 = spam).

### 3. Predict

**CLI:**

```bash
python main.py predict "Congratulations! You won a free iPhone!"
```

**Demo:**

```bash
python main.py demo
```

**Streamlit UI:**

```bash
streamlit run app.py
```

## Model Architecture

```
Input Text → BERT Tokenizer → BERT Encoder (12 layers) → [CLS] token → Classifier Head → Spam / Not Spam
```

- **Base model:** `bert-base-uncased` (pretrained)
- **Classifier head:** Linear(768→256) → ReLU → Linear(256→2)
- **Optimizer:** AdamW with linear warmup scheduler

## License

MIT
