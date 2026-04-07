import pandas as pd
import numpy as np
import torch
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
MODEL_NAME = "answerdotai/ModernBERT-base"
INPUT_CSV = './data_narratives/narrative_exposures_df.csv'
OUTPUT_JSON = './data_narratives/articles_with_embeddings.json'
BATCH_SIZE = 32

def load_model():
    """Load ModernBERT model and tokenizer."""
    print("Loading ModernBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return tokenizer, model, device

def get_embedding(text, tokenizer, model, device):
    """Generate embedding for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()

def compute_embeddings_from_kaggle():
    """
    Load articles from Kaggle dataset and compute embeddings.
    Saves results to JSON file.
    """
    # Load model once
    tokenizer, model, device = load_model()
    
    # Load Kaggle dataset
    fname = '/users/eleves-b/2023/juliette.anglade/.cache/kagglehub/datasets/rmisra/news-category-dataset/versions/3'
    full_path = os.path.join(fname, 'News_Category_Dataset_v3.json')
    
    print(f"Loading dataset from: {full_path}")
    df = pd.read_json(full_path, lines=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create output directory
    output_dir = Path(OUTPUT_JSON).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process articles and compute embeddings
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
        # Combine headline and description
        combined_text = (row.get('headline', '') or '') + ' ' + (row.get('short_description', '') or '')
        combined_text = combined_text.strip()
        
        if not combined_text:
            continue
        
        # Compute embedding
        embedding = get_embedding(combined_text, tokenizer, model, device)
        
        # Store result
        results.append({
            'idx': idx,
            'headline': row.get('headline'),
            'short_description': row.get('short_description'),
            'category': row.get('category'),
            'date': str(row.get('date')),
            'link': row.get('link'),
            'combined_text': combined_text,
            'embedding': embedding.astype(float).tolist()
        })
        
        # Save progressively every 5000 articles
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1} articles, saving checkpoint...")
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Final save
    print(f"Saving {len(results)} articles with embeddings...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Completed! Results saved to: {OUTPUT_JSON}")
    return results

if __name__ == "__main__":
    compute_embeddings_from_kaggle()