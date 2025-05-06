#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spam Detection - Topic Extrapolation

This script implements the spam detection component of the NLP Topic Extrapolation Analysis project.
It covers:
1. Loading and preprocessing spam detection datasets
2. Training models on each dataset
3. Evaluating each model's performance on all datasets to assess extrapolation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import kagglehub


# Create directories for datasets if they don't exist
def create_directories():
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)


# Download spam detection datasets
def download_datasets():
    print("Downloading datasets...")
    sms_spam_path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    email_spam_path = kagglehub.dataset_download("venky73/spam-mails-dataset")
    yt_spam_path = kagglehub.dataset_download("lakshmi25npathi/spam-dataset")

    print(f"SMS Spam dataset: {sms_spam_path}")
    print(f"Email Spam dataset: {email_spam_path}")
    print(f"YouTube Comments Spam dataset: {yt_spam_path}")
    
    return sms_spam_path, email_spam_path, yt_spam_path


# Load SMS Spam dataset
def load_sms_spam_dataset(sms_spam_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(sms_spam_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in SMS Spam dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(sms_spam_path, csv_files[0]))
    
    # Map labels (typically v1 or label column contains spam/ham)
    label_col = None
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() <= 2:
            # Check if this column contains spam/ham labels
            if 'spam' in df[col].str.lower().unique() or 'ham' in df[col].str.lower().unique():
                label_col = col
                break
    
    if label_col:
        df['label'] = df[label_col].apply(lambda x: 1 if x.lower() == 'spam' else 0)
    
    # Identify text column (typically v2 or message)
    text_col = None
    for col in df.columns:
        if col not in [label_col, 'label'] and df[col].dtype == 'object':
            text_col = col
            break
    
    if text_col:
        df['text'] = df[text_col]
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'sms'
    
    return df


# Load Email Spam dataset
def load_email_spam_dataset(email_spam_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(email_spam_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Email Spam dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(email_spam_path, csv_files[0]))
    
    # Map labels (adapt based on actual column names)
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'category' in col.lower()]
    if label_cols:
        label_col = label_cols[0]
        # Determine how labels are encoded (0/1, spam/ham, etc.)
        if df[label_col].dtype == 'object':
            df['label'] = df[label_col].apply(lambda x: 1 if 'spam' in str(x).lower() else 0)
        else:
            df['label'] = df[label_col]
    
    # Identify text column
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'message' in col.lower() or 'body' in col.lower() or 'email' in col.lower() or 'content' in col.lower()]
    if text_cols:
        df['text'] = df[text_cols[0]]
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'email'
    
    return df


# Load YouTube Comments Spam dataset
def load_yt_spam_dataset(yt_spam_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(yt_spam_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in YouTube Comments Spam dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(yt_spam_path, csv_files[0]))
    
    # Map labels (adapt based on actual column names)
    label_cols = [col for col in df.columns if 'class' in col.lower() or 'label' in col.lower() or 'spam' in col.lower()]
    if label_cols:
        label_col = label_cols[0]
        if df[label_col].dtype == 'object':
            df['label'] = df[label_col].apply(lambda x: 1 if 'spam' in str(x).lower() else 0)
        else:
            df['label'] = df[label_col]
    
    # Identify text column
    text_cols = [col for col in df.columns if 'comment' in col.lower() or 'content' in col.lower() or 'text' in col.lower() or 'message' in col.lower()]
    if text_cols:
        df['text'] = df[text_cols[0]]
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'youtube'
    
    return df


# Data preprocessing
def preprocess_dataset(df):
    if df.empty:
        return df
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text'])
    
    # Basic text cleaning
    df['text'] = df['text'].astype(str)  # Ensure text is string
    df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)   # Remove punctuation
    df['text'] = df['text'].str.lower()                              # Convert to lowercase
    
    # Ensure label is integer
    df['label'] = df['label'].astype(int)
    
    return df


# Balance datasets to ensure equal representation of classes
def balance_dataset(df, max_samples_per_class=5000):
    if df.empty:
        return df
    
    balanced_df = pd.DataFrame()
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if len(label_df) > max_samples_per_class:
            label_df = label_df.sample(max_samples_per_class, random_state=42)
        balanced_df = pd.concat([balanced_df, label_df])
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


# PyTorch dataset class
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, 
                                 truncation=True, 
                                 padding='max_length', 
                                 max_length=self.max_length, 
                                 return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Function to split dataset into train and test
def prepare_datasets(df, tokenizer, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    
    train_dataset = SpamDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = SpamDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    return train_dataset, test_dataset


# Function to train and evaluate a model
def train_and_evaluate(dataset_name, train_dataset, test_dataset):
    print(f"\nTraining model on {dataset_name} dataset...")
    
    # Initialize model
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"../models/spam_{dataset_name}_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"../results/logs/spam_{dataset_name}",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained(f"../models/spam_{dataset_name}_model")
    
    # Evaluate on test dataset
    results = trainer.evaluate()
    print(f"Evaluation results on {dataset_name} test set: {results}")
    
    return model, trainer


# Function to evaluate model on a different dataset
def evaluate_cross_dataset(model, source_dataset, target_dataset_name, target_test_dataset):
    print(f"\nEvaluating {source_dataset} model on {target_dataset_name} dataset...")
    
    # Create trainer for evaluation
    eval_args = TrainingArguments(
        output_dir=f"../results/spam_cross_eval/{source_dataset}_on_{target_dataset_name}",
        per_device_eval_batch_size=16,
    )
    
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=target_test_dataset
    )
    
    # Evaluate
    results = eval_trainer.evaluate()
    print(f"Results of {source_dataset} model on {target_dataset_name} dataset: {results}")
    
    return results


# Extract evaluation loss for comparison
def extract_results_for_visualization(models, trainers, cross_eval_results):
    # Collect results
    results_data = []
    
    for source_name in models.keys():
        # Same dataset evaluation
        same_dataset_loss = trainers[source_name].evaluate().get('eval_loss', 0)
        results_data.append({
            'Source': source_name,
            'Target': source_name,
            'Evaluation Loss': same_dataset_loss
        })
        
        # Cross-dataset evaluation
        for target_name, results in cross_eval_results.get(source_name, {}).items():
            eval_loss = results.get('eval_loss', 0)
            results_data.append({
                'Source': source_name,
                'Target': target_name,
                'Evaluation Loss': eval_loss
            })
    
    return pd.DataFrame(results_data)


# Visualize results
def visualize_results(results_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Source', y='Evaluation Loss', hue='Target', data=results_df)
    plt.title('Spam Detection Cross-Dataset Evaluation Results')
    plt.xlabel('Source Dataset (Training)')
    plt.ylabel('Evaluation Loss (Lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/spam_cross_dataset_evaluation.png")
    
    # Create a heatmap for better visualization
    pivot_df = results_df.pivot(index='Source', columns='Target', values='Evaluation Loss')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Spam Detection Cross-Dataset Evaluation Heatmap')
    plt.tight_layout()
    plt.savefig("../results/spam_cross_dataset_heatmap.png")


def main():
    # Setup
    create_directories()
    sms_spam_path, email_spam_path, yt_spam_path = download_datasets()
    
    # Load datasets
    try:
        sms_df = load_sms_spam_dataset(sms_spam_path)
        print(f"SMS Spam dataset loaded: {sms_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading SMS Spam dataset: {e}")
        sms_df = pd.DataFrame()
        
    try:
        email_df = load_email_spam_dataset(email_spam_path)
        print(f"Email Spam dataset loaded: {email_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading Email Spam dataset: {e}")
        email_df = pd.DataFrame()
        
    try:
        youtube_df = load_yt_spam_dataset(yt_spam_path)
        print(f"YouTube Comments Spam dataset loaded: {youtube_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading YouTube Comments Spam dataset: {e}")
        youtube_df = pd.DataFrame()
    
    # Adjust loaders based on actual file structure if needed
    print("\nExamining dataset structures...")
    for path, name in zip([sms_spam_path, email_spam_path, yt_spam_path], ["SMS Spam", "Email Spam", "YouTube Comments Spam"]):
        print(f"\n{name} dataset files:")
        for f in os.listdir(path):
            print(f"  {f}")
        
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            sample_df = pd.read_csv(os.path.join(path, csv_files[0]), nrows=2)
            print(f"\n{name} sample columns: {sample_df.columns.tolist()}")
    
    # Preprocess datasets
    sms_df = preprocess_dataset(sms_df)
    email_df = preprocess_dataset(email_df)
    youtube_df = preprocess_dataset(youtube_df)
    
    # Display stats after preprocessing
    for name, df in zip(["SMS Spam", "Email Spam", "YouTube Comments Spam"], [sms_df, email_df, youtube_df]):
        if not df.empty:
            print(f"{name} dataset after preprocessing: {df.shape[0]} rows")
            print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Balance datasets
    sms_df = balance_dataset(sms_df)
    email_df = balance_dataset(email_df)
    youtube_df = balance_dataset(youtube_df)
    
    # Save processed datasets
    sms_df.to_csv("../data/processed/sms_spam.csv", index=False)
    email_df.to_csv("../data/processed/email_spam.csv", index=False)
    youtube_df.to_csv("../data/processed/youtube_spam.csv", index=False)
    
    print("Datasets have been balanced and saved to disk.")
    
    # Initialize tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    datasets = {}
    
    if not sms_df.empty:
        datasets['sms'] = prepare_datasets(sms_df, tokenizer)
        print("SMS Spam datasets prepared")
        
    if not email_df.empty:
        datasets['email'] = prepare_datasets(email_df, tokenizer)
        print("Email Spam datasets prepared")
        
    if not youtube_df.empty:
        datasets['youtube'] = prepare_datasets(youtube_df, tokenizer)
        print("YouTube Comments Spam datasets prepared")
    
    # Train models on each dataset
    models = {}
    trainers = {}
    
    for dataset_name, (train_dataset, test_dataset) in datasets.items():
        model, trainer = train_and_evaluate(dataset_name, train_dataset, test_dataset)
        models[dataset_name] = model
        trainers[dataset_name] = trainer
    
    # Perform cross-dataset evaluation
    cross_eval_results = {}
    
    for source_name, model in models.items():
        cross_eval_results[source_name] = {}
        
        for target_name, (_, test_dataset) in datasets.items():
            # Skip same dataset evaluation (already done during training)
            if source_name == target_name:
                continue
                
            results = evaluate_cross_dataset(model, source_name, target_name, test_dataset)
            cross_eval_results[source_name][target_name] = results
    
    # Save results
    with open("../results/spam_cross_evaluation_results.json", "w") as f:
        json.dump(cross_eval_results, f, indent=4)
    
    # Extract and visualize results
    results_df = extract_results_for_visualization(models, trainers, cross_eval_results)
    visualize_results(results_df)
    
    print("\nSpam detection analysis complete. Results saved to ../results/ directory.")


if __name__ == "__main__":
    main() 