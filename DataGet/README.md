# Data Collection and Labeling System

This directory contains tools for collecting Weibo user data and labeling users for depression detection.

## Components

### 1. Web Crawler (`crawler.py`)

A Python-based web crawler that extracts user information and posts from Weibo and stores them in a SQLite database.

**Features:**
- Collects user profile information (nickname, gender, profile, birthday)
- Collects social metrics (followers, following, post counts)
- Collects post content with engagement metrics
- Stores data in SQLite database

**Note:** The crawler code is a framework template. You need to implement the actual extraction methods based on Weibo's current web structure or API.

### 2. Labeling Application (`labeling_app.py`)

A Flask-based web application for annotating users with depression labels.

**Features:**
- 10-dimension evaluation framework
- Two annotators can work simultaneously
- Automatic label determination (5+ positive dimensions = depressed)
- User-friendly interface

**Usage:**
```bash
python labeling_app.py
```

Then open `http://localhost:5000` in your browser.

### 3. Dataset Generator (`generate_dataset.py`)

Generates JSONL format datasets from the labeled database.

**Usage:**
```bash
# Generate balanced dataset
python generate_dataset.py --balanced

# Generate unbalanced dataset
python generate_dataset.py --unbalanced

# Generate both
python generate_dataset.py --balanced --unbalanced
```

## Installation

```bash
pip install -r requirements.txt
```

## Database Schema

The system uses SQLite with three main tables:

1. **users**: User profile information
2. **tweets**: Individual posts
3. **labeling_status**: Annotation results with dimension scores

## Labeling Dimensions

The 10 evaluation dimensions are:

1. Depressed Mood
2. Loss of Interest
3. Fatigue
4. Sleep Problems
5. Appetite Changes
6. Low Self-Worth
7. Difficulty Concentrating
8. Suicidal Ideation
9. Social Withdrawal
10. Somatic Symptoms

**Labeling Rule:** If 5 or more dimensions are marked as positive, the user is labeled as "depressed" (1); otherwise "normal" (0).

