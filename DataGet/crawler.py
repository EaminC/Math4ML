"""
Weibo user information and post crawler
Crawls user basic information, social metrics, and posts, stores in SQLite database
"""

import sqlite3
import json
import time
import random
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeiboCrawler:
    """Weibo crawler class"""
    
    def __init__(self, db_path: str = 'weibo_data.db'):
        """
        Initialize crawler
        
        Args:
            db_path: Database file path
        """
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.init_database()
    
    def init_database(self):
        """Initialize database table structure"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User information table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                nickname TEXT,
                gender TEXT,
                profile TEXT,
                birthday TEXT,
                num_of_follower INTEGER,
                num_of_following INTEGER,
                all_tweet_count INTEGER,
                original_tweet_count INTEGER,
                repost_tweet_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweets (
                tweet_id TEXT PRIMARY KEY,
                user_id TEXT,
                tweet_content TEXT,
                posting_time TEXT,
                posted_picture_url TEXT,
                num_of_likes INTEGER,
                num_of_forwards INTEGER,
                num_of_comments INTEGER,
                tweet_is_original TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        # Labeling status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labeling_status (
                user_id TEXT PRIMARY KEY,
                is_labeled INTEGER DEFAULT 0,
                label TEXT,
                labeler_name TEXT,
                labeling_time TIMESTAMP,
                dimension_scores TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def crawl_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Crawl user basic information
        
        Args:
            user_id: User ID
            
        Returns:
            User information dictionary, None if failed
        """
        try:
            # This needs to be implemented based on actual Weibo API or web structure
            # Example code, needs modification based on actual requirements
            url = f"https://weibo.com/u/{user_id}"
            
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML using BeautifulSoup (needs adjustment based on actual web structure)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract user information (needs adjustment based on actual HTML selectors)
            user_info = {
                'user_id': user_id,
                'nickname': self._extract_nickname(soup),
                'gender': self._extract_gender(soup),
                'profile': self._extract_profile(soup),
                'birthday': self._extract_birthday(soup),
                'num_of_follower': self._extract_follower_count(soup),
                'num_of_following': self._extract_following_count(soup),
                'all_tweet_count': self._extract_tweet_count(soup),
            }
            
            logger.info(f"Successfully crawled user info: {user_id}")
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to crawl user info {user_id}: {str(e)}")
            return None
    
    def crawl_user_tweets(self, user_id: str, max_tweets: int = 100) -> List[Dict]:
        """
        Crawl user posts
        
        Args:
            user_id: User ID
            max_tweets: Maximum number of posts to crawl
            
        Returns:
            List of posts
        """
        tweets = []
        try:
            # This needs to be implemented based on actual Weibo API
            # Example code, needs modification based on actual requirements
            url = f"https://weibo.com/u/{user_id}"
            
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract posts (needs adjustment based on actual web structure)
            tweet_elements = soup.select('.tweet-item')[:max_tweets]
            
            for element in tweet_elements:
                tweet = {
                    'tweet_id': self._extract_tweet_id(element),
                    'user_id': user_id,
                    'tweet_content': self._extract_tweet_content(element),
                    'posting_time': self._extract_posting_time(element),
                    'posted_picture_url': self._extract_picture_url(element),
                    'num_of_likes': self._extract_likes(element),
                    'num_of_forwards': self._extract_forwards(element),
                    'num_of_comments': self._extract_comments(element),
                    'tweet_is_original': self._extract_is_original(element),
                }
                tweets.append(tweet)
            
            logger.info(f"Successfully crawled {len(tweets)} posts: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to crawl posts {user_id}: {str(e)}")
        
        return tweets
    
    def save_user_data(self, user_info: Dict, tweets: List[Dict]):
        """
        Save user data and posts to database
        
        Args:
            user_info: User information dictionary
            tweets: List of posts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Calculate original and repost counts
            original_count = sum(1 for t in tweets if t.get('tweet_is_original', 'False') == 'True')
            repost_count = len(tweets) - original_count
            
            user_info['original_tweet_count'] = original_count
            user_info['repost_tweet_count'] = repost_count
            user_info['all_tweet_count'] = len(tweets)
            user_info['updated_at'] = datetime.now().isoformat()
            
            # Insert or update user information
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, nickname, gender, profile, birthday, num_of_follower, 
                 num_of_following, all_tweet_count, original_tweet_count, 
                 repost_tweet_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_info['user_id'],
                user_info.get('nickname', ''),
                user_info.get('gender', ''),
                user_info.get('profile', ''),
                user_info.get('birthday', ''),
                user_info.get('num_of_follower', 0),
                user_info.get('num_of_following', 0),
                user_info.get('all_tweet_count', 0),
                user_info.get('original_tweet_count', 0),
                user_info.get('repost_tweet_count', 0),
                user_info.get('updated_at', datetime.now().isoformat())
            ))
            
            # Insert posts
            for tweet in tweets:
                cursor.execute('''
                    INSERT OR REPLACE INTO tweets
                    (tweet_id, user_id, tweet_content, posting_time, posted_picture_url,
                     num_of_likes, num_of_forwards, num_of_comments, tweet_is_original)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tweet.get('tweet_id', ''),
                    tweet.get('user_id', ''),
                    tweet.get('tweet_content', ''),
                    tweet.get('posting_time', ''),
                    json.dumps(tweet.get('posted_picture_url', [])) if isinstance(tweet.get('posted_picture_url'), list) else tweet.get('posted_picture_url', ''),
                    tweet.get('num_of_likes', 0),
                    tweet.get('num_of_forwards', 0),
                    tweet.get('num_of_comments', 0),
                    tweet.get('tweet_is_original', 'False')
                ))
            
            conn.commit()
            logger.info(f"Successfully saved user data: {user_info['user_id']}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save data: {str(e)}")
        finally:
            conn.close()
    
    def crawl_user(self, user_id: str, max_tweets: int = 100):
        """
        Crawl complete data for a single user
        
        Args:
            user_id: User ID
            max_tweets: Maximum number of posts to crawl
        """
        logger.info(f"Starting to crawl user: {user_id}")
        
        # Crawl user information
        user_info = self.crawl_user_info(user_id)
        if not user_info:
            return
        
        # Crawl posts
        tweets = self.crawl_user_tweets(user_id, max_tweets)
        
        # Save data
        self.save_user_data(user_info, tweets)
    
    # The following methods need to be adjusted based on actual web structure
    def _extract_nickname(self, soup) -> str:
        """Extract nickname"""
        # Example: needs adjustment based on actual HTML structure
        return "Unknown"
    
    def _extract_gender(self, soup) -> str:
        """Extract gender"""
        return "Unknown"
    
    def _extract_profile(self, soup) -> str:
        """Extract profile"""
        return "None"
    
    def _extract_birthday(self, soup) -> str:
        """Extract birthday"""
        return "None"
    
    def _extract_follower_count(self, soup) -> int:
        """Extract follower count"""
        return 0
    
    def _extract_following_count(self, soup) -> int:
        """Extract following count"""
        return 0
    
    def _extract_tweet_count(self, soup) -> int:
        """Extract total post count"""
        return 0
    
    def _extract_tweet_id(self, element) -> str:
        """Extract post ID"""
        return ""
    
    def _extract_tweet_content(self, element) -> str:
        """Extract post content"""
        return ""
    
    def _extract_posting_time(self, element) -> str:
        """Extract posting time"""
        return ""
    
    def _extract_picture_url(self, element) -> str:
        """Extract picture URL"""
        return "None"
    
    def _extract_likes(self, element) -> int:
        """Extract likes count"""
        return 0
    
    def _extract_forwards(self, element) -> int:
        """Extract forwards count"""
        return 0
    
    def _extract_comments(self, element) -> int:
        """Extract comments count"""
        return 0
    
    def _extract_is_original(self, element) -> str:
        """Extract whether post is original"""
        return "True"


def main():
    """Main function"""
    crawler = WeiboCrawler()
    
    # Example: crawl user list
    # user_ids = ['user1', 'user2', 'user3']
    # for user_id in user_ids:
    #     crawler.crawl_user(user_id, max_tweets=100)
    #     time.sleep(random.uniform(2, 5))  # Avoid requests too fast
    
    logger.info("Crawler started, please modify crawling logic based on actual requirements")


if __name__ == '__main__':
    main()

