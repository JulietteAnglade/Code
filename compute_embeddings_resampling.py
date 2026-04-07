"""
compute_embeddings_resampled.py

Script to compute embeddings for resampled news articles and save them to JSON.

Workflow:
1. Load Kaggle dataset
2. Filter by categories (POLITICS, ENTERTAINMENT, TRAVEL, PARENTING)
3. Filter by date (<= 2018-06-01)
4. Create combined_text column (headline + short_description)
5. Resample weekly (10 articles per week, random_state=42)
6. Compute ModernBERT embeddings (768D) with batching
7. Save results to JSON with atomic writes (crash-safe)

Usage:
    python compute_embeddings_resampled.py

Output:
    ./data_narratives/resampled_articles_with_embeddings.json (~5-10MB)
"""

import pandas as pd
import numpy as np
import torch
import os
import json
import tempfile
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "answerdotai/ModernBERT-base"
KAGGLE_PATH = '/users/eleves-b/2023/juliette.anglade/.cache/kagglehub/datasets/rmisra/news-category-dataset/versions/3'
KAGGLE_FILE = os.path.join(KAGGLE_PATH, 'News_Category_Dataset_v3.json')

OUTPUT_DIR = './data_narratives'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'resampled_articles_with_embeddings.json')

CATEGORIES = ['POLITICS', 'ENTERTAINMENT', 'TRAVEL', 'PARENTING']
DATE_CUTOFF = '2018-06-01'

RESAMPLE = False
N_SAMPLES_PER_WEEK = 10
RANDOM_STATE = 42
BATCH_SIZE = 32
CHECKPOINT_EVERY = 5000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(device=None):
    """Load ModernBERT model and tokenizer."""
    print(f"Loading ModernBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.eval()
    print(f"  Device: {device}")
    return tokenizer, model, device


def get_embedding(text, tokenizer, model, device):
    """
    Generate embedding for a single text using ModernBERT.
    
    Args:
        text (str): Input text
        tokenizer: Tokenizer instance
        model: Model instance
        device: torch device
    
    Returns:
        np.ndarray: 768-dimensional embedding
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling of last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy().flatten()


def resample_weekly_random(df, n_samples=10, random_state=42, date_col='date'):
    """
    Resample dataframe to weekly frequency and randomly select n_samples per week.
    
    Accepts:
    - DataFrame with DatetimeIndex, or
    - DataFrame with date column (default: 'date')
    
    Args:
        df (pd.DataFrame): Input dataframe
        n_samples (int): Number of samples per week
        random_state (int): Random seed for reproducibility
        date_col (str): Name of date column (if not indexed)
    
    Returns:
        pd.DataFrame: Resampled dataframe with DatetimeIndex
    """
    df_copy = df.copy()
    
    # Ensure DatetimeIndex
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        if date_col in df_copy.columns:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.set_index(date_col)
        else:
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
            except Exception as e:
                raise ValueError(
                    f"DataFrame must have DatetimeIndex or date column '{date_col}' "
                    "parseable to datetime"
                ) from e
    
    # Create week identifier
    df_copy['week'] = df_copy.index.to_period('W').to_timestamp()
    
    # Sample randomly per week
    weekly_sample = (
        df_copy.groupby('week', group_keys=False)
               .apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=random_state))
    )
    
    # Clean up
    if 'week' in weekly_sample.columns:
        weekly_sample = weekly_sample.drop(columns=['week'])
    
    return weekly_sample


def save_json_atomic(data, filepath):
    """
    Save JSON data with atomic writes (crash-safe).
    
    Writes to temporary file first, then renames to target.
    
    Args:
        data (list): Data to save (list of dicts)
        filepath (str): Target file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write to temporary file in same directory (atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filepath))
    
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        os.replace(temp_path, filepath)
        print(f"✓ Saved to: {filepath}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline: load, filter, resample, embed, save."""
    
    print("=" * 80)
    print("COMPUTE EMBEDDINGS FOR RESAMPLED NEWS ARTICLES")
    print("=" * 80)
    
    # Step 1: Load Kaggle dataset
    print("\n[Step 1] Loading Kaggle dataset...")
    print(f"  Path: {KAGGLE_FILE}")
    
    if not os.path.exists(KAGGLE_FILE):
        raise FileNotFoundError(f"Dataset not found: {KAGGLE_FILE}")
    
    df = pd.read_json(KAGGLE_FILE, lines=True)
    print(f"  ✓ Loaded {len(df):,} articles")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Step 2: Filter by categories and date
    print(f"\n[Step 2] Filtering by categories {CATEGORIES} and date <= {DATE_CUTOFF}...")
    df['date'] = pd.to_datetime(df['date'])
    
    news_subset_df = df[
        (df['category'].isin(CATEGORIES)) &
        (df['date'] <= DATE_CUTOFF)
    ].copy().reset_index(drop=True)
    
    print(f"  ✓ Filtered to {len(news_subset_df):,} articles")
    print(f"  Date range: {news_subset_df['date'].min()} to {news_subset_df['date'].max()}")
    
    # Step 3: Create combined_text column
    print("\n[Step 3] Creating combined_text column...")
    news_subset_df['combined_text'] = (
        news_subset_df['headline'].fillna('') + ' ' +
        news_subset_df['short_description'].fillna('')
    )
    print(f"  ✓ Combined_text created (sample: {news_subset_df['combined_text'].iloc[0][:80]}...)")
    
    # Step 4: Resample weekly
    print(f"\n[Step 4] Resampling weekly ({N_SAMPLES_PER_WEEK} articles/week, random_state={RANDOM_STATE})...")
    news_subset_df = news_subset_df.set_index('date')
    
    if RESAMPLE:
        resampled_df = resample_weekly_random(
            news_subset_df,
            n_samples=N_SAMPLES_PER_WEEK,
            random_state=RANDOM_STATE
        ).sort_index()
    else:
        resampled_df = news_subset_df.sort_index()
    
    print(f"  Original shape: {len(news_subset_df):,} articles")
    print(f"  Resampled shape: {len(resampled_df):,} articles")
    print(f"  Date range: {resampled_df.index.min()} to {resampled_df.index.max()}")
    
    # Step 5: Load model
    print("\n[Step 5] Loading ModernBERT model...")
    tokenizer, model, device = load_model()
    
    # Step 6: Compute embeddings with batching
    print(f"\n[Step 6] Computing embeddings ({BATCH_SIZE} batch size, checkpoint every {CHECKPOINT_EVERY})...")
    
    results = []
    checkpoint_path = os.path.join(OUTPUT_DIR, 'embeddings_checkpoint.json')
    
    # Try to load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"  Found checkpoint, loading {checkpoint_path}...")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  ✓ Loaded {start_idx:,} articles from checkpoint")
    else:
        start_idx = 0
    
    resampled_list = resampled_df.reset_index().to_dict('records')
    
    for idx, row in tqdm(
        enumerate(resampled_list[start_idx:], start=start_idx),
        total=len(resampled_list),
        initial=start_idx,
        desc="Computing embeddings"
    ):
        # Compute embedding
        embedding = get_embedding(row['combined_text'], tokenizer, model, device)
        
        # Store result
        result = {
            'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
            'category': row['category'],
            'headline': row['headline'],
            'short_description': row.get('short_description', ''),
            'link': row.get('link', ''),
            'embedding': embedding.tolist()  # Convert to list for JSON serialization
        }
        results.append(result)
        
        # Checkpoint save
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            save_json_atomic(results, checkpoint_path)
            print(f"    Checkpoint: saved {idx + 1:,} articles")
    
    # Step 7: Save final results
    print(f"\n[Step 7] Saving results...")
    save_json_atomic(results, OUTPUT_JSON)
    print(f"  Total articles: {len(results):,}")
    print(f"  Embedding dimension: {len(results[0]['embedding']) if results else 'N/A'}")
    
    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  ✓ Cleaned up checkpoint")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput file: {OUTPUT_JSON}")
    print(f"File size: {os.path.getsize(OUTPUT_JSON) / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()