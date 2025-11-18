"""
Depression labeling web application
Provides 10-dimension evaluation interface, 5 or more positive dimensions indicate depression
Uses Flask framework
"""

from flask import Flask, render_template, request, jsonify, session
import sqlite3
import json
from datetime import datetime
from typing import Dict, List
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['DATABASE'] = 'weibo_data.db'

# 10 evaluation dimensions
DIMENSIONS = [
    {
        'id': 'dim1',
        'name': 'Depressed Mood',
        'description': 'User posts show persistent depressed mood, sadness, hopelessness and other negative emotions'
    },
    {
        'id': 'dim2',
        'name': 'Loss of Interest',
        'description': 'Loss of interest in daily activities, hobbies or social activities, showing obvious interest reduction'
    },
    {
        'id': 'dim3',
        'name': 'Fatigue',
        'description': 'Frequently expressing fatigue, weakness, lack of energy, feeling tired doing anything'
    },
    {
        'id': 'dim4',
        'name': 'Sleep Problems',
        'description': 'Mentioning insomnia, early awakening, excessive sleep or poor sleep quality'
    },
    {
        'id': 'dim5',
        'name': 'Appetite Changes',
        'description': 'Significant decrease or increase in appetite, significant weight changes'
    },
    {
        'id': 'dim6',
        'name': 'Low Self-Worth',
        'description': 'Showing inferiority, self-blame, self-negation, considering oneself worthless or useless'
    },
    {
        'id': 'dim7',
        'name': 'Difficulty Concentrating',
        'description': 'Difficulty concentrating, memory decline, decision-making difficulties'
    },
    {
        'id': 'dim8',
        'name': 'Suicidal Ideation',
        'description': 'Mentioning death, suicidal thoughts, self-harm behavior or related hints'
    },
    {
        'id': 'dim9',
        'name': 'Social Withdrawal',
        'description': 'Reduced social activities, avoiding others, small social network or little interaction'
    },
    {
        'id': 'dim10',
        'name': 'Somatic Symptoms',
        'description': 'Mentioning headaches, stomach pain, chest tightness and other physical discomfort without obvious organic causes'
    }
]

# Labeler list
LABELERS = ['labeler1', 'labeler2']


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn


def load_user_from_jsonl(jsonl_path: str, line_number: int = 0):
    """
    Load a user from JSONL file and import to database
    
    Args:
        jsonl_path: Path to JSONL file
        line_number: Line number to load (0-indexed)
    
    Returns:
        User data dictionary or None
    """
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if line_number >= len(lines):
                return None
            
            user_data = json.loads(lines[line_number].strip())
            
            # Generate user_id if not exists
            user_id = user_data.get('user_id', f"user_{line_number}_{hash(user_data.get('nickname', ''))}")
            user_data['user_id'] = user_id
            
            # Import to database
            conn = get_db()
            cursor = conn.cursor()
            
            try:
                # Insert user
                cursor.execute('''
                    INSERT OR REPLACE INTO users
                    (user_id, nickname, gender, profile, birthday, num_of_follower,
                     num_of_following, all_tweet_count, original_tweet_count, repost_tweet_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    user_data.get('nickname', ''),
                    user_data.get('gender', ''),
                    user_data.get('profile', 'None'),
                    user_data.get('birthday', 'None'),
                    user_data.get('num_of_follower', 0),
                    user_data.get('num_of_following', 0),
                    user_data.get('all_tweet_count', 0),
                    user_data.get('original_tweet_count', 0),
                    user_data.get('repost_tweet_count', 0)
                ))
                
                # Insert tweets
                tweets = user_data.get('tweets', [])
                for idx, tweet in enumerate(tweets[:100]):  # Limit to 100 tweets
                    tweet_id = f"{user_id}_tweet_{idx}"
                    picture_url = tweet.get('posted_picture_url', 'None')
                    if isinstance(picture_url, list):
                        picture_url = json.dumps(picture_url, ensure_ascii=False)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO tweets
                        (tweet_id, user_id, tweet_content, posting_time, posted_picture_url,
                         num_of_likes, num_of_forwards, num_of_comments, tweet_is_original)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        tweet_id,
                        user_id,
                        tweet.get('tweet_content', ''),
                        tweet.get('posting_time', ''),
                        picture_url,
                        tweet.get('num_of_likes', 0),
                        tweet.get('num_of_forwards', 0),
                        tweet.get('num_of_comments', 0),
                        tweet.get('tweet_is_original', 'False')
                    ))
                
                conn.commit()
                return user_data
            except Exception as e:
                conn.rollback()
                print(f"Error importing user: {e}")
                return None
            finally:
                conn.close()
    except Exception as e:
        print(f"Error loading JSONL: {e}")
        return None


def get_unlabeled_user():
    """Get an unlabeled user"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Find unlabeled users
    cursor.execute('''
        SELECT u.*, 
               GROUP_CONCAT(
                   json_object(
                       'tweet_id', t.tweet_id,
                       'tweet_content', t.tweet_content,
                       'posting_time', t.posting_time,
                       'num_of_likes', t.num_of_likes,
                       'num_of_forwards', t.num_of_forwards,
                       'num_of_comments', t.num_of_comments,
                       'tweet_is_original', t.tweet_is_original
                   )
               ) as tweets_json
        FROM users u
        LEFT JOIN tweets t ON u.user_id = t.user_id
        LEFT JOIN labeling_status ls ON u.user_id = ls.user_id
        WHERE ls.is_labeled IS NULL OR ls.is_labeled = 0
        GROUP BY u.user_id
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        user_data = dict(row)
        # Parse tweets JSON
        if user_data.get('tweets_json'):
            try:
                tweets = []
                for tweet_str in user_data['tweets_json'].split('},'):
                    if tweet_str:
                        if not tweet_str.endswith('}'):
                            tweet_str += '}'
                        tweets.append(json.loads(tweet_str))
                user_data['tweets'] = tweets[:50]  # Show at most 50 posts
            except:
                user_data['tweets'] = []
        else:
            user_data['tweets'] = []
        return user_data
    
    # If no unlabeled user in database, try to load from JSONL
    jsonl_paths = [
        '../Dataset/balance/depressed.jsonl',
        '../Dataset/balance/normal.jsonl',
        '../Dataset/unbalanced/depressed.jsonl',
        '../Dataset/unbalanced/normal.jsonl'
    ]
    
    for jsonl_path in jsonl_paths:
        if os.path.exists(jsonl_path):
            user_data = load_user_from_jsonl(jsonl_path, 0)
            if user_data:
                # Recursively call to get the newly imported user
                return get_unlabeled_user()
    
    return None


@app.route('/')
def index():
    """Home page"""
    if 'labeler_name' not in session:
        return render_template('login.html', labelers=LABELERS)
    
    user_data = get_unlabeled_user()
    if not user_data:
        return render_template('no_more_users.html')
    
    return render_template('labeling.html', 
                         user_data=user_data,
                         dimensions=DIMENSIONS,
                         labeler_name=session['labeler_name'])


@app.route('/login', methods=['POST'])
def login():
    """Login"""
    labeler_name = request.form.get('labeler_name')
    if labeler_name in LABELERS:
        session['labeler_name'] = labeler_name
        return jsonify({'success': True, 'redirect': '/'})
    return jsonify({'success': False, 'message': 'Invalid labeler name'})


@app.route('/logout')
def logout():
    """Logout"""
    session.pop('labeler_name', None)
    return jsonify({'success': True, 'redirect': '/'})


@app.route('/submit_label', methods=['POST'])
def submit_label():
    """Submit labeling"""
    if 'labeler_name' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.json
    user_id = data.get('user_id')
    dimension_scores = data.get('dimension_scores', {})
    
    # Calculate number of positive dimensions
    positive_count = sum(1 for score in dimension_scores.values() if score == 1)
    
    # 5 or more positive dimensions indicate depression
    label = '1' if positive_count >= 5 else '0'
    
    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO labeling_status
            (user_id, is_labeled, label, labeler_name, labeling_time, dimension_scores)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            1,
            label,
            session['labeler_name'],
            datetime.now().isoformat(),
            json.dumps(dimension_scores, ensure_ascii=False)
        ))
        
        conn.commit()
        
        return jsonify({
            'success': True,
            'label': label,
            'positive_count': positive_count,
            'message': f'Labeling successful! Positive dimensions: {positive_count}/10, Final label: {"Depressed" if label == "1" else "Normal"}'
        })
        
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'message': f'Save failed: {str(e)}'})
    finally:
        conn.close()


@app.route('/skip_user', methods=['POST'])
def skip_user():
    """Skip current user"""
    data = request.json
    user_id = data.get('user_id')
    
    # Can record skip operation, here simply return success
    return jsonify({'success': True, 'message': 'User skipped'})


@app.route('/load_from_jsonl', methods=['POST'])
def load_from_jsonl():
    """Load a user from JSONL file"""
    if 'labeler_name' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    data = request.json
    jsonl_path = data.get('jsonl_path', '../Dataset/balance/depressed.jsonl')
    line_number = data.get('line_number', 0)
    
    user_data = load_user_from_jsonl(jsonl_path, line_number)
    
    if user_data:
        return jsonify({
            'success': True,
            'message': f'Loaded user from {jsonl_path} line {line_number}',
            'user_id': user_data.get('user_id')
        })
    else:
        return jsonify({'success': False, 'message': 'Failed to load user from JSONL'})


@app.route('/stats')
def stats():
    """Statistics"""
    if 'labeler_name' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Count labeled users
    cursor.execute('SELECT COUNT(*) as total FROM labeling_status WHERE is_labeled = 1')
    total_labeled = cursor.fetchone()['total']
    
    # Count depressed and normal
    cursor.execute('SELECT label, COUNT(*) as count FROM labeling_status WHERE is_labeled = 1 GROUP BY label')
    label_stats = {row['label']: row['count'] for row in cursor.fetchall()}
    
    # Count current labeler's labeled count
    cursor.execute('''
        SELECT COUNT(*) as count FROM labeling_status 
        WHERE is_labeled = 1 AND labeler_name = ?
    ''', (session['labeler_name'],))
    my_labeled = cursor.fetchone()['count']
    
    conn.close()
    
    return jsonify({
        'success': True,
        'stats': {
            'total_labeled': total_labeled,
            'depressed_count': label_stats.get('1', 0),
            'normal_count': label_stats.get('0', 0),
            'my_labeled': my_labeled
        }
    })


if __name__ == '__main__':
    # Ensure database exists
    if not os.path.exists(app.config['DATABASE']):
        from crawler import WeiboCrawler
        crawler = WeiboCrawler(app.config['DATABASE'])
    
    app.run(debug=True, host='0.0.0.0', port=5000)
