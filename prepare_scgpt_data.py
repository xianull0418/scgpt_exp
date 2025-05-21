import scanpy as sc
import json
from pathlib import Path

# --- User-defined placeholder paths ---
# Please replace these with your actual file paths
adata_file_path = "YOUR_PATH_TO_ANNDATA.h5ad"
# This directory should contain your vocab.json
model_dir_path = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/"
# --- End of User-defined placeholder paths ---

def load_and_prepare_data(adata_path_str: str, model_dir_str: str):
    """
    Loads a single-cell AnnData object and an scGPT vocabulary, then filters 
    the AnnData object to include only genes common to both.

    Args:
        adata_path_str (str): Path to the .h5ad AnnData file.
        model_dir_str (str): Path to the directory containing the vocab.json file.

    Returns:
        tuple: A tuple containing:
            - filtered_adata (sc.AnnData | None): The AnnData object filtered to common genes.
                                                 Returns None if an error occurs.
            - model_gene_vocab (list[str] | None): The gene vocabulary loaded from vocab.json.
                                                 Returns None if an error occurs.
    """
    print(f"Loading AnnData from: {adata_path_str}")
    adata_path = Path(adata_path_str)
    if not adata_path.exists():
        print(f"Error: AnnData file not found at {adata_path_str}")
        print("Please ensure the placeholder path 'adata_file_path' is correctly set.")
        return None, None
    
    try:
        adata = sc.read_h5ad(adata_path)
        print(f"Successfully loaded AnnData object with {adata.n_obs} cells and {adata.n_vars} genes.")
    except Exception as e:
        print(f"Error reading AnnData file at {adata_path_str}: {e}")
        return None, None

    vocab_file = Path(model_dir_str) / "vocab.json"
    print(f"Loading vocabulary from: {vocab_file}")

    if not vocab_file.exists():
        print(f"Error: Vocabulary file (vocab.json) not found at {vocab_file}")
        print("Please ensure 'model_dir_path' is correct and 'vocab.json' exists in that directory.")
        return None, None

    try:
        with open(vocab_file, "r") as f:
            vocab_content = json.load(f)
        
        if isinstance(vocab_content, list):
            model_gene_vocab = vocab_content
        elif isinstance(vocab_content, dict):
            model_gene_vocab = list(vocab_content.keys())
        else:
            print(f"Error: Unexpected vocabulary format in {vocab_file}. Expected a list of genes or a dictionary.")
            return None, None
        
        if not all(isinstance(gene, str) for gene in model_gene_vocab):
            print(f"Error: Vocabulary in {vocab_file} contains non-string elements. Genes must be strings.")
            return None, None
            
        print(f"Successfully loaded vocabulary with {len(model_gene_vocab)} genes.")

    except json.JSONDecodeError:
        print(f"Error: Error decoding JSON from vocabulary file: {vocab_file}")
        return None, None
    except Exception as e:
        print(f"Error: Error loading vocabulary file {vocab_file}: {e}")
        return None, None

    original_adata_genes = list(adata.var_names)
    print(f"Number of genes in original AnnData: {len(original_adata_genes)}")

    # Identify common genes
    common_genes = sorted(list(set(original_adata_genes) & set(model_gene_vocab)))
    
    if not common_genes:
        print("Error: No common genes found between AnnData and vocabulary.")
        print("Please check your AnnData gene names and vocabulary content (e.g. gene symbols vs Ensemble IDs).")
        return None, model_gene_vocab # Return vocab even if no common genes, as per example logic implicitly
        
    print(f"Number of common genes found: {len(common_genes)}")

    # Filter AnnData to retain only common genes
    adata_filtered = adata[:, common_genes].copy()
    print(f"Filtered AnnData to {adata_filtered.n_vars} common genes.")

    return adata_filtered, model_gene_vocab # Return the full model_gene_vocab as per example

if __name__ == '__main__':
    print("Starting data preparation process...")
    
    if adata_file_path == "YOUR_PATH_TO_ANNDATA.h5ad" or \
       model_dir_path == "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/":
        print("Warning: Placeholder paths are still set to default values.")
        print("Please edit the script to set 'adata_file_path' and 'model_dir_path' to your actual paths.")
    else:
        print(f"Using AnnData path: {adata_file_path}")
        print(f"Using Model directory: {model_dir_path}")
        
        filtered_adata, vocabulary = load_and_prepare_data(adata_file_path, model_dir_path)

        if filtered_adata is not None: # Check if anndata is returned (vocab might be returned even on common_gene failure)
            print("----------------------------------------------------")
            print("Data preparation successful!")
            print(f"Filtered AnnData shape: {filtered_adata.shape}")
            if vocabulary: # vocabulary might be None if file loading failed catastrophically
                 print(f"Vocabulary size: {len(vocabulary)}")
            # You can now proceed with this filtered_adata for fine-tuning scGPT
        else:
            print("----------------------------------------------------")
            print("Data preparation failed. Please check the error messages above.")
            print("Ensure your paths are correct and files are in the expected format.")
    
    print("Data preparation script finished.")
```
