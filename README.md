# Shopee — Product Matching

## Overview
Multi-modal product matching system built for the Shopee Price Match 
Guarantee Kaggle competition. Identifies duplicate product listings 
using image similarity, text similarity and a combined approach.

## The Challenge
E-commerce platforms like Shopee have thousands of duplicate product 
listings with different titles and images. The goal is to automatically 
identify which listings represent the same product.

## Three Approaches

### Approach 1 — Image Embeddings (EfficientNetB4)
- Loaded pretrained EfficientNetB4 model (trained on ImageNet)
- Used model as feature extractor — removed classification head
- Generated 1792-dimensional embedding vector per product image
- Applied Nearest Neighbors (KD-Tree) to find visually similar products
- Images resized to 192×192 with augmentation

### Approach 2 — Text Embeddings (BERT + TF-IDF)
- Cleaned product titles — removed stopwords, special characters, numbers
- Applied lemmatization to normalize words
- Generated TF-IDF vectors from cleaned titles
- Also generated BERT embeddings for semantic understanding
- Applied Nearest Neighbors on text vectors to find similar products

### Approach 3 — Combined (Image + Text)
- Combined image embeddings and text embeddings
- Nearest Neighbors on combined representation
- Leverages both visual and textual signals for better matching

## Key Learnings
- Multi-modal approaches outperform single modality for product matching
- EfficientNetB4 generates powerful image representations without fine-tuning
- TF-IDF is surprisingly effective for product title matching
- BERT captures semantic meaning beyond keyword matching
- KD-Tree Nearest Neighbors scales well for large product catalogs

## Technologies
- Python, TensorFlow, Keras
- EfficientNetB4 (pretrained on ImageNet)
- BERT (via TensorFlow Hub)
- TF-IDF (scikit-learn)
- Nearest Neighbors (scikit-learn)
- NLTK for text preprocessing

## Dataset
Kaggle — Shopee Price Match Guarantee
https://www.kaggle.com/c/shopee-product-matching
