#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fake News Detection - Topic Extrapolation

This script implements the fake news detection component of the NLP Topic Extrapolation Analysis project.
It covers:
1. Loading and preprocessing fake news datasets
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


# Download fake news datasets
def download_datasets():
    print("Downloading datasets...")
    liar_path = kagglehub.dataset_download("mrisdal/fake-news")
    isot_path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    kaggle_fn_path = kagglehub.dataset_download("jruvika/fake-news-detection")

    print(f"LIAR dataset: {liar_path}")
    print(f"ISOT dataset: {isot_path}")
    print(f"Kaggle Fake News dataset: {kaggle_fn_path}")
    
    return liar_path, isot_path, kaggle_fn_path


# Load LIAR dataset
def load_liar_dataset(liar_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(liar_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in LIAR dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(liar_path, csv_files[0]))
    
    # Map labels (adjust based on actual column names and values)
    # This assumes a binary classification (fake=1, real=0)
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)
    
    # Standardize column names
    if 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'liar'
    
    return df


# Load ISOT dataset
def load_isot_dataset(isot_path):
    # This dataset typically has separate files for real and fake news
    real_csv = None
    fake_csv = None
    
    for f in os.listdir(isot_path):
        if 'real' in f.lower() and f.endswith('.csv'):
            real_csv = os.path.join(isot_path, f)
        elif 'fake' in f.lower() and f.endswith('.csv'):
            fake_csv = os.path.join(isot_path, f)
    
    if not real_csv or not fake_csv:
        raise FileNotFoundError("Real or fake news files not found in ISOT dataset")
    
    # Load real news with label 0
    real_df = pd.read_csv(real_csv)
    real_df['label'] = 0
    
    # Load fake news with label 1
    fake_df = pd.read_csv(fake_csv)
    fake_df['label'] = 1
    
    # Combine datasets
    df = pd.concat([real_df, fake_df], ignore_index=True)
    
    # Standardize column names
    if 'text' not in df.columns and 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'] + " " + df['text']
    elif 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'isot'
    
    return df


# Load Kaggle Fake News dataset
def load_kaggle_fn_dataset(kaggle_fn_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(kaggle_fn_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Kaggle Fake News dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(kaggle_fn_path, csv_files[0]))
    
    # Map labels (adjust based on actual column names and values)
    if 'label' in df.columns:
        # Ensure binary classification (1 for fake, 0 for real)
        df['label'] = df['label'].astype(int)
    
    # Standardize column names
    if 'text' not in df.columns and 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'] + " " + df['text']
    elif 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'kaggle_fn'
    
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
    df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
    df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)   # Remove punctuation
    df['text'] = df['text'].str.lower()                              # Convert to lowercase
    
    # Ensure label is integer
    df['label'] = df['label'].astype(int)
    
    return df


# Balance datasets to ensure equal representation of classes
def balance_dataset(df, max_samples_per_class=10000):
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
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
    
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = FakeNewsDataset(
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
        output_dir=f"../models/fake_news_{dataset_name}_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"../results/logs/fake_news_{dataset_name}",
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
    model.save_pretrained(f"../models/fake_news_{dataset_name}_model")
    
    # Evaluate on test dataset
    results = trainer.evaluate()
    print(f"Evaluation results on {dataset_name} test set: {results}")
    
    return model, trainer


# Function to evaluate model on a different dataset
def evaluate_cross_dataset(model, source_dataset, target_dataset_name, target_test_dataset):
    print(f"\nEvaluating {source_dataset} model on {target_dataset_name} dataset...")
    
    # Create trainer for evaluation
    eval_args = TrainingArguments(
        output_dir=f"../results/fake_news_cross_eval/{source_dataset}_on_{target_dataset_name}",
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
    plt.title('Fake News Cross-Dataset Evaluation Results')
    plt.xlabel('Source Dataset (Training)')
    plt.ylabel('Evaluation Loss (Lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/fake_news_cross_dataset_evaluation.png")
    
    # Create a heatmap for better visualization
    pivot_df = results_df.pivot(index='Source', columns='Target', values='Evaluation Loss')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Fake News Cross-Dataset Evaluation Heatmap')
    plt.tight_layout()
    plt.savefig("../results/fake_news_cross_dataset_heatmap.png")


def main():
    # Setup
    create_directories()
    liar_path, isot_path, kaggle_fn_path = download_datasets()
    
    # Load datasets
    try:
        liar_df = load_liar_dataset(liar_path)
        print(f"LIAR dataset loaded: {liar_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading LIAR dataset: {e}")
        liar_df = pd.DataFrame()
        
    try:
        isot_df = load_isot_dataset(isot_path)
        print(f"ISOT dataset loaded: {isot_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading ISOT dataset: {e}")
        isot_df = pd.DataFrame()
        
    try:
        kaggle_fn_df = load_kaggle_fn_dataset(kaggle_fn_path)
        print(f"Kaggle Fake News dataset loaded: {kaggle_fn_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading Kaggle Fake News dataset: {e}")
        kaggle_fn_df = pd.DataFrame()
    
    # Adjust loaders based on actual file structure if needed
    print("\nExamining dataset structures...")
    for path, name in zip([liar_path, isot_path, kaggle_fn_path], ["LIAR", "ISOT", "Kaggle Fake News"]):
        print(f"\n{name} dataset files:")
        for f in os.listdir(path):
            print(f"  {f}")
        
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            sample_df = pd.read_csv(os.path.join(path, csv_files[0]), nrows=2)
            print(f"\n{name} sample columns: {sample_df.columns.tolist()}")
    
    # Preprocess datasets
    liar_df = preprocess_dataset(liar_df)
    isot_df = preprocess_dataset(isot_df)
    kaggle_fn_df = preprocess_dataset(kaggle_fn_df)
    
    # Display stats after preprocessing
    for name, df in zip(["LIAR", "ISOT", "Kaggle Fake News"], [liar_df, isot_df, kaggle_fn_df]):
        if not df.empty:
            print(f"{name} dataset after preprocessing: {df.shape[0]} rows")
            print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Balance datasets
    liar_df = balance_dataset(liar_df)
    isot_df = balance_dataset(isot_df)
    kaggle_fn_df = balance_dataset(kaggle_fn_df)
    
    # Save processed datasets
    liar_df.to_csv("../data/processed/liar_fake_news.csv", index=False)
    isot_df.to_csv("../data/processed/isot_fake_news.csv", index=False)
    kaggle_fn_df.to_csv("../data/processed/kaggle_fake_news.csv", index=False)
    
    print("Datasets have been balanced and saved to disk.")
    
    # Initialize tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    datasets = {}
    
    if not liar_df.empty:
        datasets['liar'] = prepare_datasets(liar_df, tokenizer)
        print("LIAR datasets prepared")
        
    if not isot_df.empty:
        datasets['isot'] = prepare_datasets(isot_df, tokenizer)
        print("ISOT datasets prepared")
        
    if not kaggle_fn_df.empty:
        datasets['kaggle_fn'] = prepare_datasets(kaggle_fn_df, tokenizer)
        print("Kaggle Fake News datasets prepared")
    
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
    with open("../results/fake_news_cross_evaluation_results.json", "w") as f:
        json.dump(cross_eval_results, f, indent=4)
    
    # Extract and visualize results
    results_df = extract_results_for_visualization(models, trainers, cross_eval_results)
    visualize_results(results_df)
    
    print("\nFake news detection analysis complete. Results saved to ../results/ directory.")


if __name__ == "__main__":
    main() 