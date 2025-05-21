import numpy as np
import pandas as pd
import scanpy as sc
import random
from typing import Optional, List, Union

def generate_pseudo_bulk_samples(
    adata_filtered: sc.AnnData,
    cell_type_col: str,
    n_pseudo_bulk_samples: int = 1000,
    n_cells_per_sample_min: int = 50,
    n_cells_per_sample_max: int = 200,
    aggregation_method: str = "sum"
) -> Optional[sc.AnnData]:
    """
    Generates pseudo-bulk samples from a single-cell AnnData object.

    Args:
        adata_filtered (sc.AnnData): Filtered AnnData object containing single-cell gene expression.
        cell_type_col (str): Column name in adata_filtered.obs for cell type annotations.
        n_pseudo_bulk_samples (int): Number of pseudo-bulk samples to generate.
        n_cells_per_sample_min (int): Minimum number of cells per pseudo-bulk sample.
        n_cells_per_sample_max (int): Maximum number of cells per pseudo-bulk sample.
        aggregation_method (str): Method to aggregate gene expression ("sum" or "mean").

    Returns:
        Optional[sc.AnnData]: A new AnnData object containing pseudo-bulk data,
                              or None if an error occurs.
    """
    print(f"Starting pseudo-bulk sample generation with {n_pseudo_bulk_samples} samples...")
    print(f"Parameters: min_cells={n_cells_per_sample_min}, max_cells={n_cells_per_sample_max}, aggregation='{aggregation_method}'")

    # Validate cell_type_col
    if cell_type_col not in adata_filtered.obs.columns:
        print(f"Error: Cell type column '{cell_type_col}' not found in adata_filtered.obs. Available columns are: {list(adata_filtered.obs.columns)}")
        return None

    # Validate aggregation_method
    if aggregation_method not in ["sum", "mean"]:
        print(f"Error: Unknown aggregation_method '{aggregation_method}'. Choose 'sum' or 'mean'.")
        return None

    # Ensure n_cells_per_sample_max is not less than n_cells_per_sample_min
    if n_cells_per_sample_max < n_cells_per_sample_min:
        print(f"Error: n_cells_per_sample_max ({n_cells_per_sample_max}) cannot be less than n_cells_per_sample_min ({n_cells_per_sample_min}).")
        return None
        
    # Ensure there are enough cells to sample from
    if adata_filtered.n_obs < n_cells_per_sample_min:
        print(f"Error: Not enough cells in adata_filtered ({adata_filtered.n_obs}) to sample {n_cells_per_sample_min} cells per sample.")
        return None

    all_gene_names: List[str] = list(adata_filtered.var_names)
    unique_cell_types: List[str] = sorted(list(adata_filtered.obs[cell_type_col].astype('category').cat.categories))
    
    if not unique_cell_types:
        print(f"Error: No unique cell types found in column '{cell_type_col}'. Please check the column content.")
        return None
    print(f"Found {len(unique_cell_types)} unique cell types: {unique_cell_types}")

    pseudo_bulk_expressions: List[np.ndarray] = []
    pseudo_bulk_fractions: List[np.ndarray] = []

    # Determine if input data is sparse
    is_sparse = isinstance(adata_filtered.X, (sc.sparse.csr_matrix, sc.sparse.csc_matrix))
    if is_sparse:
        print("Input AnnData.X is sparse. Using sparse matrix operations.")
    else:
        print("Input AnnData.X is dense.")


    for i in range(n_pseudo_bulk_samples):
        # Determine number of cells for the current sample
        if n_cells_per_sample_min == n_cells_per_sample_max:
            n_cells_current_sample = n_cells_per_sample_min
        else:
            n_cells_current_sample = random.randint(n_cells_per_sample_min, n_cells_per_sample_max)
        
        # Randomly sample cell indices (without replacement)
        sampled_indices = random.sample(range(adata_filtered.n_obs), k=n_cells_current_sample)
        
        # Extract expression data for sampled cells
        # This will be a slice of the original matrix (sparse or dense)
        sampled_X = adata_filtered.X[sampled_indices, :]
        
        # Aggregate expression
        if aggregation_method == "sum":
            aggregated_expression = sampled_X.sum(axis=0)
        else: # "mean"
            aggregated_expression = sampled_X.mean(axis=0)
        
        # For sparse matrices, sum/mean might return a 2D matrix (1, n_genes). Convert to 1D array.
        if hasattr(aggregated_expression, "A1"): # Handles numpy matrix
            aggregated_expression = aggregated_expression.A1
        elif hasattr(aggregated_expression, "toarray"): # Handles some sparse results not covered by A1
             aggregated_expression = aggregated_expression.toarray().flatten()
        
        pseudo_bulk_expressions.append(aggregated_expression)

        # Calculate cell type fractions
        sampled_cell_types_series = adata_filtered.obs[cell_type_col].iloc[sampled_indices]
        type_counts = sampled_cell_types_series.value_counts().reindex(unique_cell_types, fill_value=0.0)
        fractions = type_counts / n_cells_current_sample
        pseudo_bulk_fractions.append(fractions.values) # .values to get numpy array

        if (i + 1) % (n_pseudo_bulk_samples // 10 if n_pseudo_bulk_samples >=10 else 1) == 0 : # Print progress roughly 10 times
            print(f"Generated {i + 1}/{n_pseudo_bulk_samples} pseudo-bulk samples...")

    # Convert lists to final formats
    try:
        pseudo_bulk_X_np = np.array(pseudo_bulk_expressions)
    except Exception as e:
        print(f"Error converting pseudo_bulk_expressions to NumPy array: {e}")
        # Potentially inspect shapes if there's an issue with inconsistent gene numbers (should not happen here)
        # for idx, arr in enumerate(pseudo_bulk_expressions): print(f"Shape of expression sample {idx}: {arr.shape}")
        return None

    fractions_df = pd.DataFrame(pseudo_bulk_fractions, columns=unique_cell_types)

    # Create new AnnData object for pseudo-bulk data
    try:
        pseudo_bulk_adata = sc.AnnData(X=pseudo_bulk_X_np, obs=fractions_df, var=pd.DataFrame(index=all_gene_names))
        # pseudo_bulk_adata.var_names = all_gene_names # Alternative if var is not set in constructor
    except Exception as e:
        print(f"Error creating final AnnData object for pseudo-bulk data: {e}")
        return None

    print("\nPseudo-bulk AnnData object created successfully.")
    print(f"Shape of pseudo-bulk X (samples x genes): {pseudo_bulk_adata.X.shape}")
    print(f"Shape of pseudo-bulk obs (fractions): {pseudo_bulk_adata.obs.shape}")
    print(f"Columns in pseudo-bulk obs (cell types): {list(pseudo_bulk_adata.obs.columns)}")
    
    return pseudo_bulk_adata

if __name__ == '__main__':
    print("--- Example Usage of generate_pseudo_bulk_samples ---")

    # This is an example of how to use the function.
    # First, you'd need to load and prepare your single-cell data.
    # For example, using the 'load_and_prepare_data' function from a previous script.
    
    # --- Placeholder: Create a dummy filtered_adata for demonstration ---
    print("\nCreating placeholder filtered_adata for demonstration...")
    n_example_cells, n_example_genes = 1000, 500
    example_X_data = np.random.poisson(2, size=(n_example_cells, n_example_genes)).astype(np.float32) # More realistic counts
    
    # Ensure cell types are strings and categorical for proper handling
    cell_type_categories = ['TypeA', 'TypeB', 'TypeC', 'TypeD']
    example_obs_data = pd.DataFrame({
        'celltype': pd.Categorical(np.random.choice(cell_type_categories, size=n_example_cells))
    })
    example_var_data = pd.DataFrame(index=[f'gene_{j}' for j in range(n_example_genes)])
    
    filtered_adata_placeholder = sc.AnnData(X=example_X_data, obs=example_obs_data, var=example_var_data)
    print(f"Placeholder filtered_adata created with shape: ({filtered_adata_placeholder.n_obs}, {filtered_adata_placeholder.n_vars})")
    print(f"Cell types in placeholder: {filtered_adata_placeholder.obs['celltype'].value_counts().to_dict()}")
    # --- End of Placeholder ---

    if 'filtered_adata_placeholder' in locals(): # Check if placeholder was created
        # --- Define parameters for pseudo-bulk generation ---
        cell_type_column = "celltype"      # Column in .obs with cell type labels
        num_pseudo_bulk_samples = 50       # Number of pseudo-bulk samples to create
        min_cells_per_sample = 20
        max_cells_per_sample = 100
        agg_method = "sum"                 # Aggregation method: "sum" or "mean"
        # --- End of parameters ---
        
        print("\nStarting pseudo-bulk sample generation with placeholder data...")
        pseudo_bulk_dataset = generate_pseudo_bulk_samples(
            adata_filtered=filtered_adata_placeholder,
            cell_type_col=cell_type_column,
            n_pseudo_bulk_samples=num_pseudo_bulk_samples,
            n_cells_per_sample_min=min_cells_per_sample,
            n_cells_per_sample_max=max_cells_per_sample,
            aggregation_method=agg_method
        )

        if pseudo_bulk_dataset:
            print("\nPseudo-bulk generation with placeholder data successful!")
            print(f"Resulting pseudo-bulk AnnData shape: {pseudo_bulk_dataset.shape}")
            print("First 5 rows of fraction data (pseudo_bulk_dataset.obs):")
            print(pseudo_bulk_dataset.obs.head())
            print("\nFirst 5 genes for the first 3 pseudo-bulk samples (pseudo_bulk_dataset.X):")
            print(pseudo_bulk_dataset.X[:3, :5])
        else:
            print("\nPseudo-bulk generation with placeholder data failed.")
    else:
        print("\nSkipping pseudo-bulk generation example as placeholder data was not loaded.")
        print("To run this example, ensure the placeholder data loading section is active or integrate with actual data.")

    print("\n--- End of Example Usage ---")
    print("\nReminder: To use this script effectively, integrate it with your actual data loading and preparation pipeline.")
    print("You will need to provide an actual 'filtered_adata' object loaded with real single-cell data.")
```
