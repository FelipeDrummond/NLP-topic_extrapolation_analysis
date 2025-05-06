#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Task Comparison - Topic Extrapolation Analysis

This script compares the topic extrapolation results across all three NLP tasks:
1. Sentiment Analysis
2. Fake News Detection
3. Spam Detection

The goal is to determine which NLP task shows the best generalization capabilities across datasets.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Create results directory if it doesn't exist
def create_directories():
    os.makedirs("../results", exist_ok=True)


# Load results from each task
def load_results():
    sentiment_results = {}
    fake_news_results = {}
    spam_results = {}
    
    try:
        with open("../results/cross_evaluation_results.json", "r") as f:
            sentiment_results = json.load(f)
        print("Sentiment analysis results loaded successfully")
    except Exception as e:
        print(f"Error loading sentiment analysis results: {e}")
        
    try:
        with open("../results/fake_news_cross_evaluation_results.json", "r") as f:
            fake_news_results = json.load(f)
        print("Fake news detection results loaded successfully")
    except Exception as e:
        print(f"Error loading fake news detection results: {e}")
        
    try:
        with open("../results/spam_cross_evaluation_results.json", "r") as f:
            spam_results = json.load(f)
        print("Spam detection results loaded successfully")
    except Exception as e:
        print(f"Error loading spam detection results: {e}")
    
    return sentiment_results, fake_news_results, spam_results


# Function to convert results to DataFrame
def results_to_dataframe(results_dict, task_name):
    data = []
    
    for source, targets in results_dict.items():
        for target, metrics in targets.items():
            data.append({
                'Task': task_name,
                'Source': source,
                'Target': target,
                'Evaluation Loss': metrics.get('eval_loss', float('nan')),
                'Accuracy': metrics.get('eval_accuracy', float('nan'))
            })
    
    return pd.DataFrame(data)


# Calculate normalized losses within each task
def normalize_losses(results_df):
    normalized_df = results_df.copy()
    for task in normalized_df['Task'].unique():
        task_min = normalized_df[normalized_df['Task'] == task]['Evaluation Loss'].min()
        task_max = normalized_df[normalized_df['Task'] == task]['Evaluation Loss'].max()
        
        # Min-max normalization
        if task_max > task_min:  # Avoid division by zero
            normalized_df.loc[normalized_df['Task'] == task, 'Normalized Loss'] = \
                (normalized_df.loc[normalized_df['Task'] == task, 'Evaluation Loss'] - task_min) / (task_max - task_min)
        else:
            normalized_df.loc[normalized_df['Task'] == task, 'Normalized Loss'] = 0
    
    return normalized_df


# Visualize results
def visualize_cross_task_comparison(all_results_df, normalized_df):
    # Box plot of evaluation loss across tasks
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Task', y='Evaluation Loss', data=all_results_df)
    plt.title('Cross-Task Evaluation Loss Comparison')
    plt.ylabel('Evaluation Loss (Lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/cross_task_loss_boxplot.png")
    plt.close()
    
    # Bar plot of average evaluation loss by task
    task_avg_loss = all_results_df.groupby('Task')['Evaluation Loss'].mean()
    plt.figure(figsize=(10, 6))
    task_avg_loss.plot(kind='bar', color='skyblue')
    plt.title('Average Evaluation Loss by NLP Task')
    plt.ylabel('Average Evaluation Loss (Lower is better)')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/cross_task_avg_loss.png")
    plt.close()
    
    # Visualize normalized loss by task
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Task', y='Normalized Loss', data=normalized_df)
    plt.title('Normalized Cross-Dataset Evaluation Loss by Task')
    plt.ylabel('Normalized Loss (Lower is better)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/cross_task_normalized_loss.png")
    plt.close()
    
    # Calculate average normalized loss by task
    task_avg_normalized = normalized_df.groupby('Task')['Normalized Loss'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    task_avg_normalized.plot(kind='bar', color='skyblue')
    plt.title('Average Normalized Loss by Task (Lower = Better Extrapolation)')
    plt.ylabel('Average Normalized Loss')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/task_extrapolation_ranking.png")
    plt.close()
    
    # Calculate variance in normalized loss for each task
    task_variance = normalized_df.groupby('Task')['Normalized Loss'].var().sort_values()
    plt.figure(figsize=(10, 6))
    task_variance.plot(kind='bar', color='salmon')
    plt.title('Variance in Normalized Loss by Task (Lower = More Consistent Extrapolation)')
    plt.ylabel('Variance in Normalized Loss')
    plt.xlabel('Task')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../results/task_extrapolation_consistency.png")
    plt.close()
    
    return task_avg_normalized, task_variance


# Generate conclusions based on the analysis
def generate_conclusions(task_avg_normalized, task_variance):
    # Sort tasks by extrapolation ability
    best_tasks = task_avg_normalized.index.tolist()
    most_consistent = task_variance.index.tolist()
    
    conclusions = f"""
Cross-Task Topic Extrapolation Analysis Conclusions
==================================================

Based on our cross-task comparison analysis, we can draw the following conclusions about 
topic extrapolation capabilities across the three NLP tasks:

1. Ranking of Tasks by Extrapolation Ability (best to worst): 
   {', '.join(best_tasks)}

2. Consistency of Extrapolation (most to least consistent):
   {', '.join(most_consistent)}

3. Key Findings:
   - The {best_tasks[0]} task showed the best overall generalization to unseen datasets.
   - The {most_consistent[0]} task had the most consistent performance across different datasets.
   - {best_tasks[-1]} showed the poorest generalization capabilities.

4. Implications for NLP Model Development:
   - Models trained on {best_tasks[0]} datasets appear to learn more transferable features.
   - When developing models that need to generalize well, the {best_tasks[0]} domain may 
     provide better training foundations.
   - {most_consistent[0]} models show more predictable performance when applied to new domains.

These findings contribute to our understanding of how different NLP tasks transfer 
learned knowledge across domains, which is valuable for developing more robust and 
generalizable NLP systems.
"""
    
    # Save conclusions to file
    with open("../results/cross_task_conclusions.txt", "w") as f:
        f.write(conclusions)
    
    return conclusions


def main():
    # Setup
    create_directories()
    
    # Load results
    sentiment_results, fake_news_results, spam_results = load_results()
    
    # Convert results to DataFrames
    sentiment_df = results_to_dataframe(sentiment_results, 'Sentiment Analysis')
    fake_news_df = results_to_dataframe(fake_news_results, 'Fake News Detection')
    spam_df = results_to_dataframe(spam_results, 'Spam Detection')
    
    # Combine all results
    all_results_df = pd.concat([sentiment_df, fake_news_df, spam_df], ignore_index=True)
    print(f"Combined results shape: {all_results_df.shape}")
    
    # Normalize losses within each task
    normalized_df = normalize_losses(all_results_df)
    
    # Visualize results
    task_avg_normalized, task_variance = visualize_cross_task_comparison(all_results_df, normalized_df)
    
    # Generate conclusions
    conclusions = generate_conclusions(task_avg_normalized, task_variance)
    print("\nAnalysis complete. Results and conclusions saved to ../results/ directory.")
    print("\nKey conclusions:")
    print(conclusions)


if __name__ == "__main__":
    main() 