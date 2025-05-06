#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP Topic Extrapolation Analysis - Master Runner

This script runs all components of the project in sequence:
1. Sentiment Analysis
2. Fake News Detection 
3. Spam Detection
4. Cross-Task Comparison

Usage:
    python run_all.py [--skip_sentiment] [--skip_fake_news] [--skip_spam] [--only_comparison]
"""

import os
import sys
import argparse
import subprocess
import time


def run_with_timer(script_name, script_path):
    """Run a script and time its execution"""
    print(f"\n\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        end_time = time.time()
        
        # Calculate elapsed time
        elapsed = end_time - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{script_name} completed successfully in {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{script_name} failed with exit code {e.returncode}")
        return False


def create_directories():
    """Create necessary directories"""
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run NLP Topic Extrapolation Analysis pipeline')
    parser.add_argument('--skip_sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--skip_fake_news', action='store_true', help='Skip fake news detection')
    parser.add_argument('--skip_spam', action='store_true', help='Skip spam detection')
    parser.add_argument('--only_comparison', action='store_true', help='Only run cross-task comparison')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Create directories
    create_directories()
    
    # Track overall execution time
    total_start_time = time.time()
    
    # If only running comparison, set all other flags to skip
    if args.only_comparison:
        args.skip_sentiment = True
        args.skip_fake_news = True
        args.skip_spam = True
    
    # Sentiment Analysis
    if not args.skip_sentiment:
        run_with_timer("Sentiment Analysis", "sentiment_analysis.py")
    
    # Fake News Detection
    if not args.skip_fake_news:
        run_with_timer("Fake News Detection", "fake_news_detection.py")
    
    # Spam Detection
    if not args.skip_spam:
        run_with_timer("Spam Detection", "spam_detection.py")
    
    # Cross-Task Comparison (always run this)
    run_with_timer("Cross-Task Comparison", "cross_task_comparison.py")
    
    # Calculate total time
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n\n{'='*80}")
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f"Results saved to ../results/ directory")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main() 