"""
Script to convert CSV files and HuggingFace datasets to SQL files.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

from datasets import load_dataset


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))


def sanitize_column_name(name):
    """Sanitize column names for SQL compatibility."""
    # Replace spaces and special characters
    name = name.replace(' ', '_').replace('-', '_')
    # Remove any remaining special characters
    name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
    return name


def df_to_sql_file(df, table_name, output_path):
    """Convert a DataFrame to a SQL file with CREATE TABLE and INSERT statements."""
    
    # Sanitize column names
    df.columns = [sanitize_column_name(col) for col in df.columns]
    
    # Convert complex types (dict, list, ndarray) to JSON strings
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list, np.ndarray))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, cls=NumpyEncoder) if isinstance(x, (dict, list, np.ndarray)) else x)
    
    # Create in-memory SQLite database to generate proper SQL
    conn = sqlite3.connect(':memory:')
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    
    # Get the CREATE TABLE statement
    cursor = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    create_statement = cursor.fetchone()[0]
    
    # Generate INSERT statements
    with open(output_path, 'w') as f:
        f.write(f"-- SQL dump for {table_name}\n")
        f.write(f"-- Generated from dataset conversion\n\n")
        f.write(f"DROP TABLE IF EXISTS {table_name};\n\n")
        f.write(f"{create_statement};\n\n")
        
        # Write INSERT statements in batches
        # Quote column names to handle reserved keywords like "Default"
        columns = ', '.join(f'"{col}"' for col in df.columns)
        f.write(f"-- Data rows: {len(df)}\n")
        
        for idx, row in df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('NULL')
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped = val.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(val, (int, float)):
                    values.append(str(val))
                else:
                    escaped = str(val).replace("'", "''")
                    values.append(f"'{escaped}'")
            
            values_str = ', '.join(values)
            f.write(f"INSERT INTO {table_name} ({columns}) VALUES ({values_str});\n")
    
    conn.close()
    print(f"Created: {output_path}")


def convert_csv_to_sql(csv_path, table_name, output_path):
    """Convert a CSV file to SQL."""
    print(f"Converting {csv_path}...")
    df = pd.read_csv(csv_path)
    df_to_sql_file(df, table_name, output_path)


def convert_huggingface_dataset_to_sql(dataset_name, table_name, output_path):
    """Load a HuggingFace dataset and convert to SQL."""
    print(f"Loading HuggingFace dataset: {dataset_name}...")
    ds = load_dataset(dataset_name)
    
    # Get the first split (usually 'train')
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    
    df_to_sql_file(df, table_name, output_path)


def main():
    # Ensure sql directory exists
    sql_dir = os.path.join(DATASETS_DIR, 'sql')
    csv_dir = os.path.join(DATASETS_DIR, 'csv')
    os.makedirs(sql_dir, exist_ok=True)
    
    # Convert CSV files
    csv_files = [
        ('insurance.csv', 'insurance'),
        ('Financial Distress.csv', 'financial_distress'),
        ('Loan_default.csv', 'loan_default'),
    ]
    
    for csv_file, table_name in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        if os.path.exists(csv_path):
            output_path = os.path.join(sql_dir, f'{table_name}.sql')
            convert_csv_to_sql(csv_path, table_name, output_path)
        else:
            print(f"Warning: {csv_path} not found")
    
    # Convert HuggingFace datasets
    hf_datasets = [
        ('saifhmb/CreditCardRisk', 'credit_card_risk'),
        ('JETech/underwriting-dataset-blocks', 'underwriting_dataset_blocks'),
    ]
    
    for dataset_name, table_name in hf_datasets:
        output_path = os.path.join(sql_dir, f'{table_name}.sql')
        try:
            convert_huggingface_dataset_to_sql(dataset_name, table_name, output_path)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")


if __name__ == '__main__':
    main()

