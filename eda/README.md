# Exploratory Data Analysis (EDA) for Depression Detection Dataset

This directory contains the exploratory data analysis results for both unbalanced and balanced datasets.

## Files Structure

```
eda/
├── eda_analysis.py          # Main EDA script (chunk-based processing)
├── statistics_summary.json  # Statistical summary in JSON format
├── figures_unbalanced/     # Visualizations for unbalanced dataset
│   ├── 01_class_distribution.png
│   ├── 02_gender_distribution.png
│   ├── 03_age_distribution.png
│   ├── 04_social_metrics_comparison.png
│   ├── 05_engagement_metrics.png
│   ├── 06_correlation_heatmap.png
│   └── 07_feature_comparison.png
└── figures_balanced/        # Visualizations for balanced dataset
    ├── 01_class_distribution.png
    ├── 02_gender_distribution.png
    ├── 03_age_distribution.png
    ├── 04_social_metrics_comparison.png
    ├── 05_engagement_metrics.png
    ├── 06_correlation_heatmap.png
    └── 07_feature_comparison.png
```

## Key Findings

### Unbalanced Dataset
- **Total Users**: 32,570
- **Depressed**: 10,325 (31.70%)
- **Normal**: 22,245 (68.30%)

### Balanced Dataset
- **Total Users**: 20,650
- **Depressed**: 10,325 (50.00%)
- **Normal**: 10,325 (50.00%)

### Key Observations

1. **Gender Distribution**: 
   - Female users are more prevalent in both classes
   - Similar gender ratios between depressed and normal users

2. **Age Distribution**:
   - Median age is around 28-30 years for both classes
   - Some outliers in age data (likely data quality issues)

3. **Social Media Metrics**:
   - Normal users tend to have more followers and following
   - Normal users post more tweets on average
   - Depressed users have longer average tweet length (121 vs 57 characters)

4. **Engagement Metrics**:
   - Depressed users have slightly higher average likes and comments
   - Depressed users have higher original tweet ratio (86% vs 73%)
   - Normal users post more pictures (61% vs 37%)

5. **Tweet Characteristics**:
   - Depressed users write significantly longer tweets
   - Depressed users prefer original content over reposts
   - Normal users are more likely to include pictures in tweets

## Visualizations

All visualizations are saved as high-resolution PNG files (300 DPI) with English labels only (no Chinese text in figures).

### 1. Class Distribution
Shows the balance/imbalance between depressed and normal users.

### 2. Gender Distribution
Gender breakdown by class.

### 3. Age Distribution
Age histogram comparison between classes.

### 4. Social Metrics Comparison
Box plots comparing followers, following, tweet count, and average tweet length.

### 5. Engagement Metrics
Box plots comparing likes, forwards, comments, and original tweet ratio.

### 6. Correlation Heatmap
Feature correlation matrix showing relationships between variables.

### 7. Feature Comparison
Bar chart comparing mean feature values between classes.

## Running the Analysis

To run the EDA analysis:

```bash
python3 eda/eda_analysis.py
```

The script uses chunk-based processing to handle large JSON files efficiently and avoid memory issues.

## Notes

- All visualizations use English labels only (as requested)
- The script processes data in chunks of 500 users to manage memory
- Statistical summaries are saved in JSON format for easy programmatic access

