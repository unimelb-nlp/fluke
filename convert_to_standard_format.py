#!/usr/bin/env python3
"""
Convert FLUKE dataset to standard Hugging Face format (Parquet files)
This eliminates the need for custom loading script and enables the dataset viewer.
"""

import json
import os
import pandas as pd
from pathlib import Path

def load_task_data(task_dir):
    """Load all modification files for a task into a single DataFrame."""
    data = []
    
    modification_types = [
        "active_to_passive", "capitalization", "casual", "compound_word", 
        "concept_replacement", "coordinating_conjunction", "derivation", 
        "dialectal", "discourse", "geographical_bias", "grammatical_role",
        "length_bias", "negation", "punctuation", "sentiment", 
        "temporal_bias", "typo_bias"
    ]
    
    for mod_type in modification_types:
        file_path = task_dir / f"{mod_type}_100.json"
        
        if file_path.exists():
            print(f"Loading {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            # Handle different data structures
            for example in examples:
                if isinstance(example, list) and len(example) == 2:
                    # Dialogue format: [id, data_dict]
                    example_data = example[1]
                elif isinstance(example, dict):
                    # Other tasks format: direct dict
                    example_data = example
                else:
                    print(f"Unexpected data structure in {file_path}: {type(example)}")
                    continue
                
                data.append({
                    "original": example_data.get("original_text", example_data.get("original", "")),
                    "modified": example_data.get("modified_text", example_data.get("modified", "")),
                    "label": str(example_data.get("modified_label", example_data.get("label", ""))),
                    "modification_type": example_data.get("test", example_data.get("type", mod_type)),
                })
    
    return pd.DataFrame(data)

def convert_dataset():
    """Convert FLUKE dataset to standard Hugging Face format."""
    
    # Create new directory structure
    output_dir = Path("fluke_dataset_standard")
    output_dir.mkdir(exist_ok=True)
    
    # Copy documentation files
    import shutil
    docs_to_copy = [
        "fluke_dataset/README.md",
        "fluke_dataset/requirements.txt", 
        "fluke_dataset/docs/USAGE.md"
    ]
    
    for doc_file in docs_to_copy:
        if Path(doc_file).exists():
            if "docs/" in doc_file:
                (output_dir / "docs").mkdir(exist_ok=True)
                shutil.copy2(doc_file, output_dir / "docs" / Path(doc_file).name)
            else:
                shutil.copy2(doc_file, output_dir / Path(doc_file).name)
    
    # Process each task
    tasks = ["coref", "ner", "sa", "dialogue"]
    
    for task in tasks:
        task_dir = Path("fluke_dataset/data") / task
        
        if task_dir.exists():
            print(f"\nProcessing {task} task...")
            
            # Load all data for this task
            df = load_task_data(task_dir)
            df["task"] = task
            
            print(f"Loaded {len(df)} examples for {task}")
            
            # Save as Parquet (Hugging Face preferred format)
            output_file = output_dir / f"{task}.parquet"
            df.to_parquet(output_file, index=False)
            print(f"Saved to {output_file}")
    
    # Create a combined dataset
    print("\nCreating combined dataset...")
    all_data = []
    
    for task in tasks:
        parquet_file = output_dir / f"{task}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_parquet(output_dir / "train.parquet", index=False)
        print(f"Combined dataset saved with {len(combined_df)} total examples")
    
    print(f"\nConversion complete! New dataset structure:")
    print(f"📁 {output_dir}/")
    for file in sorted(output_dir.rglob("*")):
        if file.is_file():
            size = file.stat().st_size / 1024 / 1024  # MB
            print(f"   📄 {file.relative_to(output_dir)} ({size:.1f} MB)")

if __name__ == "__main__":
    convert_dataset() 