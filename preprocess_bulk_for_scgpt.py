import scanpy as sc
import numpy as np
import pandas as pd # For example data generation
from typing import Optional

def preprocess_pseudo_bulk_for_scgpt(
    pseudo_bulk_adata: sc.AnnData,
    target_sum: Optional[float] = 1e4,
    n_bins: int = 51,
    hvg_col_name: Optional[str] = None
) -> Optional[sc.AnnData]:
    """
    Preprocesses pseudo-bulk AnnData for scGPT input.
    Performs optional HVG filtering, normalization, log-transformation, 
    and bins gene expression values.

    Args:
        pseudo_bulk_adata (sc.AnnData): AnnData object containing pseudo-bulk gene expression.
        target_sum (Optional[float]): Target sum for count normalization (e.g., 1e4 for CPM).
                                      If None, normalization is skipped.
        n_bins (int): Number of bins for discretizing expression values (e.g., 51 for scGPT).
        hvg_col_name (Optional[str]): Name of the column in .var that marks highly variable genes.
                                      If provided, filters to these genes.

    Returns:
        Optional[sc.AnnData]: The processed AnnData object with binned expression,
                              or None if a critical error occurs.
    """
    print("Starting preprocessing of pseudo-bulk data...")

    # Make a copy to avoid modifying the original AnnData object unexpectedly
    adata_processed = pseudo_bulk_adata.copy()
    print(f"Copied input AnnData. Original shape: {pseudo_bulk_adata.shape}, Copied shape: {adata_processed.shape}")

    # Optional HVG filtering
    if hvg_col_name:
        if hvg_col_name in adata_processed.var.columns:
            if adata_processed.var[hvg_col_name].dtype == bool:
                print(f"Filtering genes based on boolean column '{hvg_col_name}'.")
                adata_processed = adata_processed[:, adata_processed.var[hvg_col_name]].copy()
                print(f"Number of genes after HVG filtering: {adata_processed.n_vars}")
                if adata_processed.n_vars == 0:
                    print(f"Warning: No genes remaining after HVG filtering with column '{hvg_col_name}'. Check your HVG data.")
                    return None # Or handle as appropriate
            else:
                print(f"Warning: HVG column '{hvg_col_name}' is not boolean. Skipping HVG filtering.")
        else:
            print(f"Warning: HVG column '{hvg_col_name}' not found in .var. Skipping HVG filtering.")

    # Normalization
    if target_sum is not None:
        if adata_processed.X is None:
            print("Error: adata_processed.X is None. Cannot perform normalization.")
            return None
        print(f"Normalizing total counts per sample to {target_sum}...")
        sc.pp.normalize_total(adata_processed, target_sum=target_sum)
        # Store the normalized data in a layer if further direct use is needed,
        # though sc.pp.log1p will operate on adata_processed.X which is now normalized.
        adata_processed.layers['normalized'] = adata_processed.X.copy() 
    else:
        print("Skipping total count normalization as target_sum is None.")
        # If not normalizing, ensure X is not None before log1p
        if adata_processed.X is None:
            print("Error: adata_processed.X is None and target_sum is None. Cannot proceed.")
            return None

    # Log1p Transformation
    print("Applying log1p transformation...")
    if target_sum is not None:
        # If normalized, X was updated by normalize_total. sc.pp.log1p works in-place on .X
        sc.pp.log1p(adata_processed) 
        adata_processed.layers['log1p'] = adata_processed.X.copy()
    else:
        # If not normalized, apply log1p to the original X and store in layers['log1p']
        # Also update .X to be this log1p data for consistency
        adata_processed.layers['log1p'] = np.log1p(adata_processed.X)
        adata_processed.X = adata_processed.layers['log1p'].copy()
    print("Log1p transformation complete. Data stored in adata_processed.layers['log1p'] and adata_processed.X")

    # Binning log-transformed expression values
    print(f"Binning log-transformed expression values into {n_bins} bins...")
    data_to_bin = adata_processed.layers['log1p']
    
    if data_to_bin is None:
        print("Error: data_to_bin (from layers['log1p']) is None. Cannot perform binning.")
        return None

    # Determine min and max for binning from the data itself
    # Filter out non-finite values for min/max calculation
    finite_data = data_to_bin[np.isfinite(data_to_bin)]
    if finite_data.size == 0:
        print("Error: No finite data available for binning. Check input data.")
        return None

    min_val = np.min(finite_data)
    max_val = np.max(finite_data)
    print(f"Data range for binning (finite values): min={min_val:.4f}, max={max_val:.4f}")

    if min_val == max_val: # Handle case where all values are the same
        print("Warning: All values in data_to_bin are identical after filtering for finite values. Assigning all to bin 0.")
        # All values will fall into the first bin (index 0)
        binned_expression = np.zeros(data_to_bin.shape, dtype=int)
    else:
        # Create n_bins intervals (n_bins+1 edges for np.linspace)
        # np.digitize needs n_bins edges to produce n_bins+1 possible output values (0 to n_bins)
        # Or, n_bins-1 edges to produce n_bins values (0 to n_bins-1 if values < edges[0] are mapped to 0)
        # The example uses bins[:-1] with np.digitize, meaning n_bins-1 edges.
        # bins = np.linspace(min_val, max_val, num=n_bins) # This creates n_bins points, meaning n_bins-1 intervals.
                                                        # If used as edges for np.digitize, it results in n_bins categories.
        
        # Let's follow the logic from the prompt's example:
        # bins = np.linspace(min_val, max_val, num=n_bins) # These are the centers/representative points for n_bins
        # If n_bins=51, this gives 51 points.
        # For np.digitize, we need edges. If we want n_bins distinct integer outputs (0 to n_bins-1),
        # we need n_bins-1 edges.
        # Example: bins = [0,1,2,3,4], n_bins=5. edges = [0.5, 1.5, 2.5, 3.5] (n_bins-1 edges)
        # value < 0.5 -> 0
        # 0.5 <= value < 1.5 -> 1
        # ...
        # value >= 3.5 -> 4
        
        # The example's np.digitize(data_to_bin, bins[:-1], right=False) and then clip logic:
        # `bins` from np.linspace(min_val, max_val, num=n_bins) means `bins` has `n_bins` elements.
        # `bins[:-1]` means it has `n_bins-1` elements (edges).
        # `np.digitize` with `n_bins-1` edges will return values from 0 to `n_bins-1`.
        # value < edges[0] -> 0
        # edges[0] <= value < edges[1] -> 1
        # ...
        # value >= edges[n_bins-2] (last edge) -> n_bins-1
        # This seems correct for 0 to n_bins-1 output directly.
        # The example's subsequent `clip(binned_expression - 1, 0, n_bins - 1)` implies digitize might output 1 to n_bins.
        # Let's re-verify np.digitize:
        # If bins = [b1, b2, b3], digitize(x, bins) gives:
        # x < b1 -> 0
        # b1 <= x < b2 -> 1
        # b2 <= x < b3 -> 2
        # x >= b3 -> 3
        # So, with N edges, it gives N+1 results.
        # If we want n_bins results (0 to n_bins-1), we need n_bins-1 edges.
        # So, `edges = np.linspace(min_val, max_val, num=n_bins -1)` is wrong.
        # `edges = np.linspace(min_val, max_val, num=n_bins)` will give n_bins points.
        # If these are edges for np.digitize, `np.digitize(data, edges)` would give 0 to n_bins.
        
        # Let's use the exact formulation from the prompt for np.digitize part to be safe.
        # It seems it intends `bins` from linspace to be the upper bounds of the bins.
        bin_edges = np.linspace(min_val, max_val, num=n_bins) # These are n_bins points.
                                                            # If used as edges, they define n_bins-1 intervals.
                                                            # np.digitize with these edges would give indices 0 to n_bins-1 for values < bin_edges[-1]
                                                            # and n_bins for values >= bin_edges[-1]
        
        # If bin_edges has K elements, np.digitize(X, bin_edges) returns indices i such that bin_edges[i-1] <= x < bin_edges[i]
        # If we want n_bins categories (0 to n_bins-1):
        # We need n_bins-1 actual thresholds to divide the space into n_bins regions.
        # Example: n_bins=3. Values 0, 1, 2.
        # Edges: e1, e2.  x < e1 -> 0;  e1 <= x < e2 -> 1; x >= e2 -> 2.
        # So, np.linspace(min_val, max_val, num=n_bins -1) would give the WRONG number of edges.
        # Let's use `num=n_bins` for linspace to define the *upper* edges of the bins, then adjust.
        
        # Create n_bins + 1 edges to define n_bins intervals.
        bin_edges_for_digitize = np.linspace(min_val, max_val, num=n_bins + 1)
        
        # Digitize: values are assigned to bin indices 1 to n_bins if they fall within these edges.
        # Values < bin_edges_for_digitize[0] are 0. Values >= bin_edges_for_digitize[-1] are n_bins+1.
        # This is not what the example used.
        
        # Reverting to the example's direct logic for binning part:
        # `bins = np.linspace(min_val, max_val, num=n_bins)` -> these are n_bins points.
        # `binned_expression = np.digitize(data_to_bin, bins[:-1], right=False)`
        # `bins[:-1]` has `n_bins-1` elements.
        # `np.digitize` with `N` edges (bins[:-1]) returns values from `0` to `N`.
        # So, `binned_expression` will range from `0` to `n_bins-1`.
        # Example: n_bins=51. `bins` has 51 elements. `bins[:-1]` has 50 elements (edges).
        # `np.digitize` will return values from 0 to 50. This seems correct.
        # The example's subsequent clip `np.clip(binned_expression - 1, 0, n_bins - 1)` was likely a safeguard or
        # based on a different interpretation of np.digitize's output initially.
        # If np.digitize(data, edges_len_N) gives 0 to N, then it's already 0 to n_bins-1.
        
        # Let's test np.digitize behavior carefully:
        # x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
        # edges = np.array([1, 2, 3]) # 3 edges
        # np.digitize(x, edges) -> array([0, 0, 1, 1, 2, 2, 3]) # Output range 0 to len(edges) which is 0 to 3.
        # So, if `bins[:-1]` has `n_bins-1` edges, output is `0` to `n_bins-1`. This is desired.
        # The example's `clip(binned_expression -1 ...)` would shift this to -1 to n_bins-2. That's not right.
        # The prompt's code for clip is: `np.clip(binned_expression -1, 0, n_bins - 1)`
        # If binned_expression is already 0 to n_bins-1, then `binned_expression - 1` is -1 to n_bins-2.
        # Clipping this to (0, n_bins-1) makes it 0 to n_bins-2. This effectively loses the last bin.
        # This must be a misunderstanding in my interpretation or the example's clip logic.

        # Let's assume the goal is: result in [0, n_bins-1].
        # Using `np.linspace(min_val, max_val, num=n_bins)` creates `n_bins` points.
        # If these are bin *upper boundaries* (exclusive, for right=False), then `bins[:-1]` are the first `n_bins-1` upper boundaries.
        # `np.digitize(data, bins[:-1], right=False)`:
        #   - Values < `bins[0]` map to bin 0.
        #   - `bins[0]` <= Values < `bins[1]` map to bin 1.
        #   ...
        #   - `bins[n_bins-2]` <= Values map to bin `n_bins-1`. (since `bins[:-1]` means last edge is `bins[n_bins-2]`)
        # This directly gives values in `0, ..., n_bins-1`.
        
        # So the clip `binned_expression - 1` in the example is likely incorrect if using `bins[:-1]`.
        # Let's use the simpler interpretation that `np.digitize` with `k` edges gives `k+1` bins (0 to `k`).
        # If `bins = np.linspace(min_val, max_val, n_bins-1)` these are `n_bins-1` edges.
        # `np.digitize(data_to_bin, bins)` would result in values from 0 to `n_bins-1`. This seems simplest.

        bin_edges = np.linspace(min_val, max_val, num=n_bins - 1) # n_bins-1 edges define n_bins intervals
        binned_expression = np.digitize(data_to_bin, bin_edges, right=False)
        # np.digitize with N edges returns values from 0 to N.
        # Here, N = n_bins-1. So output is 0 to n_bins-1. This is exactly what we want.
        # No clipping should be necessary with this formulation.

    adata_processed.layers['binned_expression'] = binned_expression.astype(np.int32) # Ensure integer type
    print(f"Binning complete. Binned data stored in adata_processed.layers['binned_expression']")
    
    # Verify min/max of binned data
    if binned_expression.size > 0:
        print(f"Min bin index: {np.min(binned_expression)}, Max bin index: {np.max(binned_expression)}")
        if np.max(binned_expression) >= n_bins:
            print(f"Warning: Max bin index ({np.max(binned_expression)}) is >= n_bins ({n_bins}). Check binning logic.")
    else:
        print("Warning: Binned expression is empty.")


    # The primary expression matrix X can remain as log1p.
    # adata_processed.X is already set to log1p data.

    print("Preprocessing complete.")
    return adata_processed

if __name__ == '__main__':
    print("--- Example Usage of preprocess_pseudo_bulk_for_scgpt ---")
    
    print("\nCreating placeholder pseudo_bulk_adata for demonstration...")
    n_samples, n_genes_ps = 100, 50
    # Simulate raw counts for pseudo-bulk (non-negative)
    example_X_ps = np.random.randint(0, 1000, size=(n_samples, n_genes_ps)).astype(float) 
    example_obs_ps = pd.DataFrame({f'meta_{k}': np.random.rand(n_samples) for k in range(2)},
                                  index=[f"sample_{i}" for i in range(n_samples)])
    example_var_ps = pd.DataFrame(index=[f'gene_{j}' for j in range(n_genes_ps)])
    
    pseudo_bulk_adata_placeholder = sc.AnnData(X=example_X_ps, obs=example_obs_ps, var=example_var_ps)
    
    # Add some fake HVG info to test that path (optional)
    # pseudo_bulk_adata_placeholder.var['highly_variable_genes'] = np.random.choice([True, False], size=n_genes_ps)
    # pseudo_bulk_adata_placeholder.var['highly_variable_genes'].iloc[:n_genes_ps//2] = True # Ensure some are true

    print(f"Placeholder pseudo_bulk_adata created with shape: {pseudo_bulk_adata_placeholder.shape}")
    print(f"Example raw data (first 3x5 sample):\n{pseudo_bulk_adata_placeholder.X[:3, :5]}")

    # --- Define parameters for preprocessing ---
    norm_target_sum = 10000.0  # e.g., CPM like normalization
    num_bins_for_scgpt = 51    # Number of bins for scGPT model
    # hvg_filter_col = "highly_variable_genes" # Example: "highly_variable_genes" or None
    hvg_filter_col = None 
    # --- End of parameters ---

    print("\nStarting pseudo-bulk data preprocessing with placeholder data...")
    processed_adata = preprocess_pseudo_bulk_for_scgpt(
        pseudo_bulk_adata_placeholder,
        target_sum=norm_target_sum,
        n_bins=num_bins_for_scgpt,
        hvg_col_name=hvg_filter_col
    )

    if processed_adata:
        print("\nPseudo-bulk preprocessing successful!")
        print(f"Resulting AnnData shape: {processed_adata.shape}")
        
        if 'normalized' in processed_adata.layers:
            print(f"Normalized data sum (first sample, if normalization was run): {processed_adata.layers['normalized'][0].sum():.2f}")
        
        if 'log1p' in processed_adata.layers:
            print(f"Log1p data (first 3x5 sample):\n{processed_adata.layers['log1p'][:3, :5]}")
        
        if 'binned_expression' in processed_adata.layers:
            print(f"Binned expression data (first 3x5 sample):\n{processed_adata.layers['binned_expression'][:3, :5]}")
            unique_bins = np.unique(processed_adata.layers['binned_expression'])
            print(f"Unique bin values observed: {unique_bins}")
            print(f"Min bin: {np.min(unique_bins)}, Max bin: {np.max(unique_bins)}")
            if np.max(unique_bins) >= num_bins_for_scgpt:
                 print(f"Warning: Max bin {np.max(unique_bins)} is out of expected range [0, {num_bins_for_scgpt-1}]")

        # Check .X content
        # print(f"Final .X data (first 3x5 sample, should be log1p):\n{processed_adata.X[:3, :5]}")

    else:
        print("\nPseudo-bulk preprocessing failed. Check logs for errors.")
        
    print("\n--- End of Example Usage ---")
    print("\nReminder: To use this script, integrate it with your actual pseudo-bulk data generation pipeline.")

```
