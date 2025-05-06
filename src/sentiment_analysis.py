#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Analysis - Topic Extrapolation

This script implements the sentiment analysis component of the NLP Topic Extrapolation Analysis project.
It covers:
1. Loading and preprocessing three sentiment analysis datasets
2. Converting the Amazon dataset from 5-point scale to ternary classification
3. Training models on each dataset
4. Evaluating each model's performance on all datasets to assess extrapolation
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


# Download datasets
def download_datasets():
    print("Downloading datasets...")
    twitter_path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
    financial_path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
    amazon_path = kagglehub.dataset_download("tarkkaanko/amazon")

    print(f"Twitter dataset: {twitter_path}")
    print(f"Financial dataset: {financial_path}")
    print(f"Amazon dataset: {amazon_path}")
    
    return twitter_path, financial_path, amazon_path


# Load Twitter dataset
def load_twitter_dataset(twitter_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(twitter_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Twitter dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(twitter_path, csv_files[0]))
    
    # Map sentiment labels if needed
    sentiment_mapping = {
        'Positive': 2,
        'Neutral': 1,
        'Negative': 0
    }
    
    # Check column names and format
    if 'sentiment' in df.columns:
        if df['sentiment'].dtype == 'object':
            df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)
        else:
            # Adjust based on the actual numeric mapping in the dataset
            df['sentiment_label'] = df['sentiment']
    
    # Standardize column names
    df = df.rename(columns={'text': 'text', 'sentiment_label': 'label'})
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'twitter'
    
    return df


# Load Financial dataset
def load_financial_dataset(financial_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(financial_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Financial dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(financial_path, csv_files[0]))
    
    # Map sentiment labels if needed
    sentiment_mapping = {
        'positive': 2,
        'neutral': 1,
        'negative': 0
    }
    
    # Check column names and format
    if 'sentiment' in df.columns:
        if df['sentiment'].dtype == 'object':
            df['label'] = df['sentiment'].map(sentiment_mapping)
        else:
            # Adjust based on the actual numeric mapping in the dataset
            df['label'] = df['sentiment']
    
    # Standardize column names
    df = df.rename(columns={'sentence': 'text'})
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'financial'
    
    return df


# Load Amazon dataset
def load_amazon_dataset(amazon_path):
    # Identify the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(amazon_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in Amazon dataset directory")
    
    # Load the dataset
    df = pd.read_csv(os.path.join(amazon_path, csv_files[0]))
    
    # Map 1-5 ratings to sentiment categories
    def map_rating_to_sentiment(rating):
        if rating <= 2:  # 1-2 stars: Negative
            return 0
        elif rating == 3:  # 3 stars: Neutral
            return 1
        else:  # 4-5 stars: Positive
            return 2
    
    # Apply mapping
    if 'rating' in df.columns:
        df['label'] = df['rating'].apply(map_rating_to_sentiment)
    
    # Standardize column names
    df = df.rename(columns={'review_text': 'text'})
    
    # Select relevant columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']]
    
    # Add dataset identifier
    df['dataset'] = 'amazon'
    
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
    df['text'] = df['text'].str.replace(r'#\w+', '', regex=True)     # Remove hashtags
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
class SentimentDataset(Dataset):
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
    
    train_dataset = SentimentDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer
    )
    
    test_dataset = SentimentDataset(
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"../models/{dataset_name}_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"../results/logs/{dataset_name}",
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
    model.save_pretrained(f"../models/{dataset_name}_model")
    
    # Evaluate on test dataset
    results = trainer.evaluate()
    print(f"Evaluation results on {dataset_name} test set: {results}")
    
    return model, trainer


# Function to evaluate model on a different dataset
def evaluate_cross_dataset(model, source_dataset, target_dataset_name, target_test_dataset):
    print(f"\nEvaluating {source_dataset} model on {target_dataset_name} dataset...")
    
    # Create trainer for evaluation
    eval_args = TrainingArguments(
        output_dir=f"../results/cross_eval/{source_dataset}_on_{target_dataset_name}",
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
    plt.title('Cross-Dataset Evaluation Results')
    plt.xlabel('Source Dataset (Training)')
    plt.ylabel('Evaluation Loss (Lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/cross_dataset_evaluation.png")
    
    # Create a heatmap for better visualization
    pivot_df = results_df.pivot(index='Source', columns='Target', values='Evaluation Loss')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Cross-Dataset Evaluation Heatmap')
    plt.tight_layout()
    plt.savefig("../results/cross_dataset_heatmap.png")


def main():
    # Setup
    create_directories()
    twitter_path, financial_path, amazon_path = download_datasets()
    
    # Load datasets
    try:
        twitter_df = load_twitter_dataset(twitter_path)
        print(f"Twitter dataset loaded: {twitter_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading Twitter dataset: {e}")
        twitter_df = pd.DataFrame()
        
    try:
        financial_df = load_financial_dataset(financial_path)
        print(f"Financial dataset loaded: {financial_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading Financial dataset: {e}")
        financial_df = pd.DataFrame()
        
    try:
        amazon_df = load_amazon_dataset(amazon_path)
        print(f"Amazon dataset loaded: {amazon_df.shape[0]} rows")
    except Exception as e:
        print(f"Error loading Amazon dataset: {e}")
        amazon_df = pd.DataFrame()
    
    # Adjust loaders based on actual file structure if needed
    print("\nExamining dataset structures...")
    for path, name in zip([twitter_path, financial_path, amazon_path], ["Twitter", "Financial", "Amazon"]):
        print(f"\n{name} dataset files:")
        for f in os.listdir(path):
            print(f"  {f}")
        
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            sample_df = pd.read_csv(os.path.join(path, csv_files[0]), nrows=2)
            print(f"\n{name} sample columns: {sample_df.columns.tolist()}")
    
    # Preprocess datasets
    twitter_df = preprocess_dataset(twitter_df)
    financial_df = preprocess_dataset(financial_df)
    amazon_df = preprocess_dataset(amazon_df)
    
    # Display stats after preprocessing
    for name, df in zip(["Twitter", "Financial", "Amazon"], [twitter_df, financial_df, amazon_df]):
        if not df.empty:
            print(f"{name} dataset after preprocessing: {df.shape[0]} rows")
            print(f"Label distribution:\n{df['label'].value_counts()}\n")
    
    # Balance datasets
    twitter_df = balance_dataset(twitter_df)
    financial_df = balance_dataset(financial_df)
    amazon_df = balance_dataset(amazon_df)
    
    # Save processed datasets
    twitter_df.to_csv("../data/processed/twitter_sentiment.csv", index=False)
    financial_df.to_csv("../data/processed/financial_sentiment.csv", index=False)
    amazon_df.to_csv("../data/processed/amazon_sentiment.csv", index=False)
    
    print("Datasets have been balanced and saved to disk.")
    
    # Initialize tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare datasets
    datasets = {}
    
    if not twitter_df.empty:
        datasets['twitter'] = prepare_datasets(twitter_df, tokenizer)
        print("Twitter datasets prepared")
        
    if not financial_df.empty:
        datasets['financial'] = prepare_datasets(financial_df, tokenizer)
        print("Financial datasets prepared")
        
    if not amazon_df.empty:
        datasets['amazon'] = prepare_datasets(amazon_df, tokenizer)
        print("Amazon datasets prepared")
    
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
    with open("../results/cross_evaluation_results.json", "w") as f:
        json.dump(cross_eval_results, f, indent=4)
    
    # Extract and visualize results
    results_df = extract_results_for_visualization(models, trainers, cross_eval_results)
    visualize_results(results_df)
    
    print("\nSentiment analysis complete. Results saved to ../results/ directory.")


if __name__ == "__main__":
    main() 