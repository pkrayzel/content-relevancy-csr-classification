import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
df = pd.read_csv("labeled_dataset.csv")

# Ensure correct data types
df['label'] = df['label'].astype(int)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Dataset object
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        [f"Title: {title} URL: {url} Sentence: {sentence}" for title, url, sentence in zip(examples['title'], examples['url'], examples['sentence'])],
        padding="max_length",
        truncation=True
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set dataset format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("fine_tuned_distilbert")
tokenizer.save_pretrained("fine_tuned_distilbert")

print("Fine-tuned model and tokenizer saved to 'fine_tuned_distilbert'")
