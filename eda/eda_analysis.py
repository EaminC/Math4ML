#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) for Depression Detection Dataset
Analyzes both unbalanced and balanced datasets
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for plots (no Chinese fonts)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_jsonl_chunk(filepath, chunk_size=1000):
    """Load JSONL file in chunks"""
    with open(filepath, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            chunk.append(json.loads(line.strip()))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def calculate_age(birthday_str):
    """Calculate age from birthday string with validation"""
    if not birthday_str or birthday_str == "无" or len(birthday_str) < 4:
        return None
    try:
        # Extract year (first 4 digits)
        year = int(birthday_str[:4])
        current_year = datetime.now().year
        
        # Validate year is reasonable (between 1900 and current year)
        if year < 1900 or year > current_year:
            return None
        
        age = current_year - year
        
        # Only return age if it's reasonable (0-120 years)
        if 0 <= age <= 120:
            return age
        else:
            return None
    except:
        return None

def extract_tweet_features(user_data):
    """Extract features from user's tweets"""
    tweets = user_data.get('tweets', [])
    if not tweets:
        return {
            'avg_tweet_length': 0,
            'total_likes': 0,
            'total_forwards': 0,
            'total_comments': 0,
            'avg_likes': 0,
            'avg_forwards': 0,
            'avg_comments': 0,
            'original_ratio': 0,
            'has_picture_ratio': 0,
            'tweet_count': 0
        }
    
    tweet_lengths = []
    total_likes = 0
    total_forwards = 0
    total_comments = 0
    original_count = 0
    has_picture_count = 0
    
    for tweet in tweets:
        content = tweet.get('tweet_content', '')
        if content and content != '无' and content != '':
            tweet_lengths.append(len(content))
        
        total_likes += int(tweet.get('num_of_likes', 0) or 0)
        total_forwards += int(tweet.get('num_of_forwards', 0) or 0)
        total_comments += int(tweet.get('num_of_comments', 0) or 0)
        
        if tweet.get('tweet_is_original') == 'True':
            original_count += 1
        
        pic_url = tweet.get('posted_picture_url', '无')
        if pic_url and pic_url != '无' and pic_url != '':
            has_picture_count += 1
    
    tweet_count = len(tweets)
    return {
        'avg_tweet_length': np.mean(tweet_lengths) if tweet_lengths else 0,
        'total_likes': total_likes,
        'total_forwards': total_forwards,
        'total_comments': total_comments,
        'avg_likes': total_likes / tweet_count if tweet_count > 0 else 0,
        'avg_forwards': total_forwards / tweet_count if tweet_count > 0 else 0,
        'avg_comments': total_comments / tweet_count if tweet_count > 0 else 0,
        'original_ratio': original_count / tweet_count if tweet_count > 0 else 0,
        'has_picture_ratio': has_picture_count / tweet_count if tweet_count > 0 else 0,
        'tweet_count': tweet_count
    }

def process_user_chunk(users, label):
    """Process a chunk of users"""
    chunk_data = []
    for user in users:
        user_info = {
            'label': label,
            'nickname': user.get('nickname', ''),
            'gender': user.get('gender', ''),
            'profile': user.get('profile', ''),
            'birthday': user.get('birthday', ''),
            'age': calculate_age(user.get('birthday', '')),
            'num_of_follower': int(user.get('num_of_follower', 0) or 0),
            'num_of_following': int(user.get('num_of_following', 0) or 0),
            'all_tweet_count': int(user.get('all_tweet_count', 0) or 0),
            'original_tweet_count': int(user.get('original_tweet_count', 0) or 0),
            'repost_tweet_count': int(user.get('repost_tweet_count', 0) or 0),
        }
        tweet_features = extract_tweet_features(user)
        user_info.update(tweet_features)
        chunk_data.append(user_info)
    return chunk_data

def analyze_dataset(depressed_file, normal_file, dataset_name):
    """Analyze a dataset (unbalanced or balanced) - using chunk processing"""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name.upper()} Dataset")
    print(f"{'='*60}")
    
    # Process data in chunks
    print("Loading and processing data in chunks...")
    chunk_size = 500  # Process 500 users at a time
    
    all_data = []
    depressed_count = 0
    normal_count = 0
    
    # Process depressed users
    print("Processing depressed users...")
    for chunk_idx, chunk in enumerate(load_jsonl_chunk(depressed_file, chunk_size)):
        chunk_data = process_user_chunk(chunk, 1)
        all_data.extend(chunk_data)
        depressed_count += len(chunk)
        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {depressed_count} depressed users...")
    
    print(f"Depressed users: {depressed_count}")
    
    # Process normal users
    print("Processing normal users...")
    for chunk_idx, chunk in enumerate(load_jsonl_chunk(normal_file, chunk_size)):
        chunk_data = process_user_chunk(chunk, 0)
        all_data.extend(chunk_data)
        normal_count += len(chunk)
        if (chunk_idx + 1) % 20 == 0:
            print(f"  Processed {normal_count} normal users...")
    
    print(f"Normal users: {normal_count}")
    
    # Create DataFrame from processed data
    print("Creating DataFrame...")
    df = pd.DataFrame(all_data)
    del all_data  # Free memory
    
    # Basic statistics
    print(f"\n{'='*60}")
    print("BASIC STATISTICS")
    print(f"{'='*60}")
    print(f"\nDataset size: {len(df)}")
    print(f"Depressed: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.2f}%)")
    print(f"Normal: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.2f}%)")
    
    # Save statistics
    stats = {
        'dataset_name': dataset_name,
        'total_users': len(df),
        'depressed_count': len(df[df['label']==1]),
        'normal_count': len(df[df['label']==0]),
        'depressed_ratio': len(df[df['label']==1])/len(df),
        'normal_ratio': len(df[df['label']==0])/len(df),
    }
    
    # Gender distribution
    print(f"\n{'='*60}")
    print("GENDER DISTRIBUTION")
    print(f"{'='*60}")
    gender_stats = df.groupby(['label', 'gender']).size().unstack(fill_value=0)
    print(gender_stats)
    stats['gender_distribution'] = gender_stats.to_dict()
    
    # Age statistics
    print(f"\n{'='*60}")
    print("AGE STATISTICS")
    print(f"{'='*60}")
    age_stats = df.groupby('label')['age'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(age_stats)
    stats['age_statistics'] = age_stats.to_dict()
    
    # Social media statistics
    print(f"\n{'='*60}")
    print("SOCIAL MEDIA STATISTICS")
    print(f"{'='*60}")
    social_cols = ['num_of_follower', 'num_of_following', 'all_tweet_count']
    for col in social_cols:
        print(f"\n{col}:")
        print(df.groupby('label')[col].agg(['mean', 'median', 'std']))
    
    # Tweet features statistics
    print(f"\n{'='*60}")
    print("TWEET FEATURES STATISTICS")
    print(f"{'='*60}")
    tweet_cols = ['avg_tweet_length', 'avg_likes', 'avg_forwards', 'avg_comments', 
                  'original_ratio', 'has_picture_ratio']
    for col in tweet_cols:
        print(f"\n{col}:")
        print(df.groupby('label')[col].agg(['mean', 'median', 'std']))
    
    # Create visualizations
    create_visualizations(df, dataset_name, stats)
    
    return df, stats

def create_visualizations(df, dataset_name, stats):
    """Create visualization plots"""
    output_dir = f"eda/figures_{dataset_name}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class distribution
    plt.figure(figsize=(10, 6))
    class_counts = df['label'].value_counts()
    plt.bar(['Normal', 'Depressed'], [class_counts[0], class_counts[1]], 
            color=['#3498db', '#e74c3c'])
    plt.title(f'Class Distribution - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Users')
    plt.xlabel('Class')
    for i, v in enumerate([class_counts[0], class_counts[1]]):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gender distribution
    plt.figure(figsize=(12, 6))
    gender_counts = df.groupby(['label', 'gender']).size().unstack(fill_value=0)
    gender_counts.plot(kind='bar', color=['#9b59b6', '#f39c12', '#1abc9c'])
    plt.title(f'Gender Distribution by Class - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Users')
    plt.xlabel('Class')
    plt.xticks([0, 1], ['Normal', 'Depressed'], rotation=0)
    plt.legend(title='Gender', labels=['Female', 'Male', 'Unknown'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Age distribution (filtered to reasonable range)
    plt.figure(figsize=(12, 6))
    df_with_age = df[df['age'].notna() & (df['age'] >= 0) & (df['age'] <= 100)]
    for label in [0, 1]:
        age_data = df_with_age[df_with_age['label'] == label]['age']
        if len(age_data) > 0:
            plt.hist(age_data, bins=30, alpha=0.6, label=['Normal', 'Depressed'][label],
                    color=['#3498db', '#e74c3c'][label], edgecolor='black', linewidth=0.5)
    plt.title(f'Age Distribution by Class - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Frequency')
    plt.xlim(0, 100)  # Set reasonable x-axis limits
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Social media metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = [
        ('num_of_follower', 'Number of Followers'),
        ('num_of_following', 'Number of Following'),
        ('all_tweet_count', 'Total Tweet Count'),
        ('avg_tweet_length', 'Average Tweet Length')
    ]
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data_to_plot = [df[df['label']==0][col].dropna(), 
                       df[df['label']==1][col].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Normal', 'Depressed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Social Media Metrics Comparison - {dataset_name.upper()}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_social_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Engagement metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    engagement_metrics = [
        ('avg_likes', 'Average Likes per Tweet'),
        ('avg_forwards', 'Average Forwards per Tweet'),
        ('avg_comments', 'Average Comments per Tweet'),
        ('original_ratio', 'Original Tweet Ratio')
    ]
    
    for idx, (col, title) in enumerate(engagement_metrics):
        ax = axes[idx // 2, idx % 2]
        data_to_plot = [df[df['label']==0][col].dropna(), 
                       df[df['label']==1][col].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['Normal', 'Depressed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Engagement Metrics Comparison - {dataset_name.upper()}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_engagement_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap
    plt.figure(figsize=(14, 10))
    numeric_cols = ['num_of_follower', 'num_of_following', 'all_tweet_count',
                   'avg_tweet_length', 'avg_likes', 'avg_forwards', 'avg_comments',
                   'original_ratio', 'has_picture_ratio', 'age']
    corr_matrix = df[numeric_cols + ['label']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Matrix - {dataset_name.upper()}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Statistical summary comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    summary_cols = ['num_of_follower', 'num_of_following', 'all_tweet_count',
                   'avg_tweet_length', 'avg_likes', 'avg_forwards', 'avg_comments']
    
    normal_means = [df[df['label']==0][col].mean() for col in summary_cols]
    depressed_means = [df[df['label']==1][col].mean() for col in summary_cols]
    
    x = np.arange(len(summary_cols))
    width = 0.35
    
    ax.bar(x - width/2, normal_means, width, label='Normal', color='#3498db')
    ax.bar(x + width/2, depressed_means, width, label='Depressed', color='#e74c3c')
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Mean Value')
    ax.set_title(f'Mean Feature Values Comparison - {dataset_name.upper()}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([col.replace('_', ' ').title() for col in summary_cols], 
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}/")

def main():
    """Main function"""
    print("="*60)
    print("DEPRESSION DETECTION DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Analyze unbalanced dataset
    df_unbalanced, stats_unbalanced = analyze_dataset(
        'Dataset/unbalanced/depressed.jsonl',
        'Dataset/unbalanced/normal.jsonl',
        'unbalanced'
    )
    
    # Analyze balanced dataset
    df_balanced, stats_balanced = analyze_dataset(
        'Dataset/balance/depressed.jsonl',
        'Dataset/balance/normal.jsonl',
        'balanced'
    )
    
    # Save statistics to JSON
    import json
    with open('eda/statistics_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'unbalanced': stats_unbalanced,
            'balanced': stats_balanced
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - eda/figures_unbalanced/ (visualizations for unbalanced dataset)")
    print("  - eda/figures_balanced/ (visualizations for balanced dataset)")
    print("  - eda/statistics_summary.json (statistical summary)")

if __name__ == "__main__":
    main()

