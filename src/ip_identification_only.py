from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Example function to generate token-level labels based on incorrect and correct sentence pairs
def generate_token_labels(incorrect_sentence, correct_sentence):
    # Tokenize sentences
    incorrect_tokens = incorrect_sentence.split()
    correct_tokens = correct_sentence.split()
    
    # Use SequenceMatcher to find matching and differing blocks
    matcher = SequenceMatcher(None, incorrect_tokens, correct_tokens)
    labels = [0] * len(incorrect_tokens)  # Initialize all labels as correct (0)

    # Mark differing tokens as incorrect
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                labels[i] = 1  # Mark as incorrect

    return labels

# Read sentences and generate token-level labels
def read_sentences_and_labels(src_path, tgt_path, limit=1000000):
    with open(src_path, "r", encoding="utf-8") as src_file, open(tgt_path, "r", encoding="utf-8") as tgt_file:
        incorrect_sentences = [line.strip() for line in tqdm(src_file.readlines(), desc="Reading Incorrect Sentences")]
        correct_sentences = [line.strip() for line in tqdm(tgt_file.readlines(), desc="Reading Correct Sentences")]

    # Limit dataset size for faster processing
    incorrect_sentences = incorrect_sentences[:limit]
    correct_sentences = correct_sentences[:limit]

    sentences = []
    labels = []

    # Generate token-level labels for each sentence pair
    for incorrect, correct in tqdm(zip(incorrect_sentences, correct_sentences), desc="Generating Labels", total=len(incorrect_sentences)):
        sentences.append(incorrect)
        token_labels = generate_token_labels(incorrect, correct)
        labels.append(token_labels)

    return sentences, labels

# Load sentences and labels (limited subset)
src_path = "wikiExtractsData/data/train_merge.src"
tgt_path = "wikiExtractsData/data/train_merge.tgt"
sentences, labels = read_sentences_and_labels(src_path, tgt_path)

# Tokenize and prepare dataset
class TokenClassificationDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=64):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Tokenize the sentence and align labels with tokens
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Get input ids and attention mask
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Align the labels with tokens
        word_ids = encoding.word_ids(batch_index=0)  # Map word_ids to tokens

        labels_aligned = []
        for word_id in word_ids:
            if word_id is None:
                labels_aligned.append(-100)  # Ignore these tokens (e.g., padding tokens)
            else:
                labels_aligned.append(label[word_id] if word_id < len(label) else 0)
        
        labels_aligned = torch.tensor(labels_aligned)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_aligned
        }

# Create the dataset
dataset = TokenClassificationDataset(sentences, labels, tokenizer, 128)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Load pre-trained BERT model for token classification
model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=2  # Binary classification: correct or incorrect
)

training_args = TrainingArguments(
    output_dir="./token_classification_results",
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=True
)


# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = torch.argmax(torch.tensor(pred.predictions), axis=-1).flatten()

    # Filter out -100 labels (ignored labels for padding, etc.)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the model
model.save_pretrained("./token_error_detection_model")
tokenizer.save_pretrained("./token_error_detection_model")

# Make predictions for new sentence
sentence = "यह एक गलत वाक्य है।"
inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to("cuda" if torch.cuda.is_available() else "cpu")

# Move model to CUDA if available
model.to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, axis=-1)

# Map predictions back to tokens
tokens = tokenizer.tokenize(sentence)
for token, prediction in zip(tokens, predictions[0][:len(tokens)]):
    status = "Incorrect" if prediction == 1 else "Correct"
    print(f"Token: {token}, Status: {status}")
