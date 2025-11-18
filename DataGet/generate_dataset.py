"""
Generate dataset from database
Export labeled data to JSONL format, consistent with existing dataset format
"""

import sqlite3
import json
import os
from typing import List, Dict
from datetime import datetime


def generate_dataset(db_path: str = 'weibo_data.db', 
                     output_dir: str = '../Dataset',
                     balanced: bool = True):
    """
    Generate dataset from database
    
    Args:
        db_path: Database file path
        output_dir: Output directory
        balanced: Whether to generate balanced dataset
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all labeled users
    cursor.execute('''
        SELECT u.*, ls.label, ls.dimension_scores
        FROM users u
        INNER JOIN labeling_status ls ON u.user_id = ls.user_id
        WHERE ls.is_labeled = 1
    ''')
    
    users = cursor.fetchall()
    
    # Separate depressed and normal users
    depressed_users = []
    normal_users = []
    
    for user in users:
        user_dict = dict(user)
        label = user_dict['label']
        
        # Get all posts for this user
        cursor.execute('''
            SELECT * FROM tweets WHERE user_id = ? ORDER BY posting_time DESC
        ''', (user_dict['user_id'],))
        
        tweets = []
        for tweet_row in cursor.fetchall():
            tweet_dict = dict(tweet_row)
            # Handle picture URL (might be JSON string)
            picture_url = tweet_dict.get('posted_picture_url', '')
            if picture_url and picture_url != 'None':
                try:
                    picture_url = json.loads(picture_url)
                except:
                    pass
            
            tweet = {
                'tweet_content': tweet_dict.get('tweet_content', ''),
                'posting_time': tweet_dict.get('posting_time', ''),
                'posted_picture_url': picture_url if isinstance(picture_url, list) else (picture_url if picture_url else 'None'),
                'num_of_likes': tweet_dict.get('num_of_likes', 0),
                'num_of_forwards': tweet_dict.get('num_of_forwards', 0),
                'num_of_comments': tweet_dict.get('num_of_comments', 0),
                'tweet_is_original': tweet_dict.get('tweet_is_original', 'False')
            }
            tweets.append(tweet)
        
        # Build user data
        user_data = {
            'label': label,
            'nickname': user_dict.get('nickname', ''),
            'gender': user_dict.get('gender', ''),
            'profile': user_dict.get('profile', 'None'),
            'birthday': user_dict.get('birthday', 'None'),
            'num_of_follower': user_dict.get('num_of_follower', 0),
            'num_of_following': user_dict.get('num_of_following', 0),
            'all_tweet_count': user_dict.get('all_tweet_count', 0),
            'original_tweet_count': user_dict.get('original_tweet_count', 0),
            'repost_tweet_count': user_dict.get('repost_tweet_count', 0),
            'tweets': tweets
        }
        
        if label == '1':
            depressed_users.append(user_data)
        else:
            normal_users.append(user_data)
    
    conn.close()
    
    print(f"Found {len(depressed_users)} depressed users, {len(normal_users)} normal users")
    
    # Determine output directory
    if balanced:
        output_path = os.path.join(output_dir, 'balance')
    else:
        output_path = os.path.join(output_dir, 'unbalanced')
    
    os.makedirs(output_path, exist_ok=True)
    
    # Generate balanced dataset
    if balanced:
        # Find smaller class count
        min_count = min(len(depressed_users), len(normal_users))
        
        # Random sampling
        import random
        random.seed(42)  # Fixed random seed for reproducibility
        depressed_sampled = random.sample(depressed_users, min_count)
        normal_sampled = random.sample(normal_users, min_count)
        
        print(f"Generating balanced dataset: {min_count} depressed users, {min_count} normal users")
        
        # Write JSONL files
        with open(os.path.join(output_path, 'depressed.jsonl'), 'w', encoding='utf-8') as f:
            for user in depressed_sampled:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        with open(os.path.join(output_path, 'normal.jsonl'), 'w', encoding='utf-8') as f:
            for user in normal_sampled:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
    
    else:
        # Generate unbalanced dataset (use all data)
        print(f"Generating unbalanced dataset: {len(depressed_users)} depressed users, {len(normal_users)} normal users")
        
        with open(os.path.join(output_path, 'depressed.jsonl'), 'w', encoding='utf-8') as f:
            for user in depressed_users:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
        
        with open(os.path.join(output_path, 'normal.jsonl'), 'w', encoding='utf-8') as f:
            for user in normal_users:
                f.write(json.dumps(user, ensure_ascii=False) + '\n')
    
    print(f"Dataset generated to: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate dataset from database')
    parser.add_argument('--db', type=str, default='weibo_data.db', help='Database file path')
    parser.add_argument('--output', type=str, default='../Dataset', help='Output directory')
    parser.add_argument('--balanced', action='store_true', help='Generate balanced dataset')
    parser.add_argument('--unbalanced', action='store_true', help='Generate unbalanced dataset')
    
    args = parser.parse_args()
    
    if args.balanced:
        generate_dataset(args.db, args.output, balanced=True)
    
    if args.unbalanced:
        generate_dataset(args.db, args.output, balanced=False)
    
    if not args.balanced and not args.unbalanced:
        # Default: generate both datasets
        generate_dataset(args.db, args.output, balanced=True)
        generate_dataset(args.db, args.output, balanced=False)


if __name__ == '__main__':
    main()
