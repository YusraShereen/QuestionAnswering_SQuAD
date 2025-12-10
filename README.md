# Extractive Question Answering on SQuAD v1.1

This project implements and compares multiple transformer-based models for extractive question answering using the Stanford Question Answering Dataset (SQuAD v1.1).

## Project Overview

The system trains and evaluates five different transformer architectures:
- BERT-base-uncased (baseline)
- ALBERT-base-v2
- MobileBERT
- MobileBERT (tuned hyperparameters)
- ALBERT-base-v2 (tuned hyperparameters)

## Requirements

### Environment Setup

```bash
# Create virtual environment
python -m venv qa_env
source qa_env/bin/activate  # On Windows: qa_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
simpletransformers==0.64.3
transformers==4.30.0
torch==2.0.1
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

## Dataset Preparation

### Download SQuAD v1.1

```bash
mkdir -p data
cd data

# Training set
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O train_simple.json

# Development set
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dev_simple.json

cd ..
```

### Dataset Structure

The SQuAD format requires:
```json
[
  {
    "context": "paragraph text...",
    "qas": [
      {
        "id": "unique_id",
        "question": "question text?",
        "answers": [
          {"text": "answer", "answer_start": 123}
        ]
      }
    ]
  }
]
```

## Running the Code

### Option 1: Run Complete Pipeline

Execute the main training script:

```bash
python train_qa_models.py
```

This will:
1. Train model
2. Evaluate on the development set
3. Extract misclassified examples
4. Generate comparison metrics
5. Save results to `outputs/`

### Option 2: Train Individual Models

```python
from qa_pipeline import full_pipeline

# Configure model arguments
model_args = {
    "learning_rate": 3e-5,
    "num_train_epochs": 1,
    "max_seq_length": 384,
    "doc_stride": 128,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "n_best_size": 20,
    "max_answer_length": 30,
    "overwrite_output_dir": True,
    "reprocess_input_data": True,
    "do_lower_case": True,
    "output_dir": "outputs/bert-squad",
}

# Train and evaluate
results = full_pipeline(
    model_type="bert",
    model_name="bert-base-uncased",
    train_file="data/train_simple.json",
    eval_file="data/dev_simple.json",
    model_args=model_args,
    experiment_name="BERT_Baseline"
)
```

## Output Files

### Model Checkpoints

Each model saves:
- `outputs/{model_name}/best_model/` - Best performing checkpoint
- `outputs/{model_name}/checkpoint-*/` - Training checkpoints

### Evaluation Results

- `outputs/*_misclassified.json` - Incorrect predictions with context
- `outputs/model_comparison.json` - Side-by-side model comparison
- `outputs/model_comparison_results.csv` - Performance metrics table

### Misclassified Examples Format

```json
{
  "metadata": {
    "model_name": "bert-base-uncased",
    "misclassified_count": 2494
  },
  "misclassified_examples": [
    {
      "question_id": "...",
      "question": "...",
      "context": "...",
      "predicted_answer": "...",
      "true_answers": ["..."],
      "exact_match": 0,
      "f1_score": 0.0
    }
  ]
}
```

## Hyperparameter Tuning

### Baseline Configuration

```python
baseline_args = {
    "learning_rate": 3e-5,
    "num_train_epochs": 1,
    "max_seq_length": 384,
    "doc_stride": 128,
    "train_batch_size": 8
}
```

### Tuned Configuration (MobileBERT/ALBERT)

```python
tuned_args = {
    "learning_rate": 5e-5,      # Increased
    "num_train_epochs": 5,       # More epochs
    "max_seq_length": 512,       # Longer sequences
    "doc_stride": 64,            # More overlap
    "train_batch_size": 16       # Larger batches
}
```

## Evaluation Metrics

The system reports three key metrics:

1. **Exact Match (EM)**: Percentage of predictions matching ground truth exactly
2. **F1 Score**: Token-level overlap between prediction and ground truth
3. **Training Time**: Wall-clock time for complete training

### Interpreting Results

- **correct**: Exact match (EM = 1)
- **similar**: Partial match (EM = 0, F1 > 0)
- **incorrect**: Wrong answer (EM = 0, F1 ≈ 0)

For error analysis, only the `incorrect` category is examined.

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes:
```python
model_args["train_batch_size"] = 4
model_args["eval_batch_size"] = 4
```

### Slow Training

Use CPU instead:
```python
model = QuestionAnsweringModel(
    model_type,
    model_name,
    args=model_args,
    use_cuda=False  # Force CPU
)
```

### Missing Dependencies

```bash
# Install specific versions
pip install simpletransformers==0.64.3
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 16 GB
- Disk: 10 GB free space

### Recommended
- GPU: NVIDIA with 8+ GB VRAM (e.g., RTX 3070)
- RAM: 32 GB
- Disk: 20 GB free space (for model checkpoints)
