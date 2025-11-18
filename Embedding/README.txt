Embedding Files Download Information
=====================================

The embedding files (vectorized datasets) are too large to be stored in this repository.

Please download the embedding files from the following Google Drive link:

https://drive.google.com/drive/folders/1w3JKnouiu-FNb2Qmk_QmMlcsFGuoVg-o?usp=drive_link

Embedding Details:
- The Embedding directory contains 16 datasets:
  - 2 datasets (balanced/unbalanced) × 4 methods (TF-IDF, N-grams, Word2Vec, GloVe) × 2 versions (original/PCA)
- Each dataset contains:
  - depressed.npy: Vectorized embeddings for depressed users
  - normal.npy: Vectorized embeddings for normal users

Directory Structure:
- Embedding/balanced/{method}/{original|pca}/
- Embedding/unbalanced/{method}/{original|pca}/

After downloading, extract the Embedding folder to the project root directory.

