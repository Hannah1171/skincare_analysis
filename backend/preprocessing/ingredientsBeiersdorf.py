import pandas as pd
from collections import defaultdict

def load_ingredient_map(filepath: str, sep: str = ';') -> dict:
    """
    Load and preprocess ingredient mapping from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.
        sep (str): Delimiter used in the CSV file (default is ';').

    Returns:
        dict: Dictionary with ingredients as keys and list of search terms as values.
    """
    ingredient_df = pd.read_csv(filepath, sep=sep)

    # Normalize text (strip + lowercase)
    ingredient_df["Ingredient"] = ingredient_df["Ingredient"].astype(str).str.strip().str.lower()
    ingredient_df["searchTerm"] = ingredient_df["searchTerm"].astype(str).str.strip().str.lower()

    # Build mapping
    ingredient_map = defaultdict(list)
    for _, row in ingredient_df.iterrows():
        ingredient_map[row["Ingredient"]].append(row["searchTerm"])

    # Remove duplicates
    return {k: list(set(v)) for k, v in ingredient_map.items()}
