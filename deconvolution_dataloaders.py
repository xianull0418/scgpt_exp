import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scanpy as sc # For AnnData
import sys
from typing import Dict, Any, Optional

# Assuming scgpt is in the Python path
# sys.path.append("path_to_scGPT_repository") # If scgpt is not installed
try:
    from scgpt.tokenizer import GeneVocab # For loading/using vocab
except ImportError:
    print("Warning: scGPT.tokenizer.GeneVocab not found. Using a dummy class for script generation.")
    # Define a dummy GeneVocab if scGPT is not available, for script generation purposes
    class GeneVocab:
        def __init__(self, gene_list, specials=None):
            self.stoi = {}
            # Add genes first
            for i, gene in enumerate(gene_list):
                self.stoi[gene] = i
            
            # Add special tokens, ensuring they don't overwrite existing gene indices if names clash
            # and ensuring they get unique indices if not clashing.
            current_max_idx = len(gene_list) -1
            if specials:
                for special in specials:
                    if special not in self.stoi:
                        current_max_idx += 1
                        self.stoi[special] = current_max_idx
            
            self.itos = {i: gene for gene, i in self.stoi.items()}
            
            # Define common special tokens attributes, defaulting if not in input specials
            self._pad_token = "<pad>"
            self._cls_token = "<cls>"
            self._mask_token = "<mask>" # Example, not used in this script directly
            self._eos_token = "<eos>"   # Example
            self._eoc_token = "<eoc>"   # Example

            if self._pad_token not in self.stoi: self.stoi[self._pad_token] = len(self.stoi)
            if self._cls_token not in self.stoi: self.stoi[self._cls_token] = len(self.stoi)
            # Rebuild itos after potential additions
            self.itos = {i: gene for gene, i in self.stoi.items()}


        def __getitem__(self, token: str) -> int:
            return self.stoi.get(token, -1) # Or handle unknown token appropriately

        def __len__(self) -> int:
            return len(self.stoi)
        
        @property
        def pad_idx(self) -> int:
            idx = self.stoi.get(self._pad_token)
            if idx is None: raise ValueError(f"Pad token '{self._pad_token}' not found in vocab.")
            return idx

        @property
        def cls_idx(self) -> int:
            idx = self.stoi.get(self._cls_token)
            if idx is None: raise ValueError(f"CLS token '{self._cls_token}' not found in vocab.")
            return idx
        
        @property
        def mask_idx(self) -> Optional[int]:
            return self.stoi.get(self._mask_token)

        @property
        def eos_idx(self) -> Optional[int]:
            return self.stoi.get(self._eos_token)
        
        @property
        def eoc_idx(self) -> Optional[int]:
            return self.stoi.get(self._eoc_token)


        @classmethod
        def from_file(cls, filepath: str): # Dummy method
            print(f"Dummy GeneVocab.from_file called with {filepath}")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict): # Assuming stoi format like {"<pad>":0, "gene1":1}
                    # Extract genes and specials based on typical patterns
                    genes = [k for k,v in data.items() if not (k.startswith("<") and k.endswith(">"))]
                    specials = [k for k,v in data.items() if (k.startswith("<") and k.endswith(">"))]
                    return cls(genes, specials=specials)
                elif isinstance(data, list): # Assuming list of genes
                    return cls(data, specials=["<pad>", "<cls>", "<eoc>", "<mask>", "<eos>"]) # Add default specials
            except Exception as e:
                print(f"Error in dummy GeneVocab.from_file: {e}. Using default small vocab.")
            # Fallback to a very simple vocab if file loading fails or is not implemented
            return cls(["gene1", "gene2", "gene3"], specials=["<pad>", "<cls>"])


class PseudoBulkDataset(Dataset):
    def __init__(self, 
                 pseudo_bulk_adata: sc.AnnData, 
                 vocab: GeneVocab, 
                 max_seq_len: int, 
                 expression_layer: str = 'binned_expression', 
                 cls_value_bin: int = 0, 
                 pad_value: int = -2):
        """
        Custom PyTorch Dataset for pseudo-bulk deconvolution.

        Args:
            pseudo_bulk_adata: Preprocessed AnnData with binned expression and cell fractions.
            vocab: scGPT vocabulary object (GeneVocab).
            max_seq_len: Maximum sequence length (genes + CLS token).
            expression_layer: Layer in AnnData containing binned expression values.
            cls_value_bin: Bin value to assign to the CLS token's expression.
            pad_value: Value used for padding expression values.
        """
        self.adata = pseudo_bulk_adata
        self.genes = list(pseudo_bulk_adata.var_names)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.expression_layer = expression_layer
        self.cls_value_bin = cls_value_bin 
        self.pad_value = pad_value 

        if not self.genes:
            raise ValueError("pseudo_bulk_adata.var_names is empty. Cannot create dataset.")

        # Pre-calculate gene IDs (excluding CLS token for now)
        self.gene_ids_no_cls = [self.vocab[gene] for gene in self.genes]
        if any(gid == -1 for gid in self.gene_ids_no_cls): # Check if any gene was not found in vocab
            unknown_genes = [self.genes[i] for i, gid in enumerate(self.gene_ids_no_cls) if gid == -1]
            print(f"Warning: Some genes not found in vocab and will be mapped to -1: {unknown_genes[:5]}...")
            # Depending on model, -1 might cause issues if not handled by an embedding layer's padding_idx
        
        # Ensure CLS and PAD tokens are in vocab (properties will raise error if not)
        _ = self.vocab.cls_idx 
        _ = self.vocab.pad_idx

    def __len__(self) -> int:
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Get binned expression for the sample
        # Ensure it's a dense numpy array
        raw_expression_vector = self.adata.layers[self.expression_layer][idx]
        if hasattr(raw_expression_vector, "toarray"): # Check if sparse
            sample_binned_expression = raw_expression_vector.toarray().flatten()
        else:
            sample_binned_expression = np.asarray(raw_expression_vector).flatten()

        # 2. Get cell type fractions
        # Assuming fractions are in .obs and are already numeric
        sample_fractions_np = self.adata.obs.iloc[idx, :].values.astype(np.float32)
        sample_fractions = torch.tensor(sample_fractions_np, dtype=torch.float32)

        # 3. Prepare gene IDs and binned values with CLS token
        # Gene IDs: [CLS_ID, gene1_ID, gene2_ID, ...]
        current_gene_ids_list = [self.vocab.cls_idx] + self.gene_ids_no_cls
        # Binned values: [CLS_VALUE_BIN, bin_val1, bin_val2, ...]
        current_values_list = np.concatenate((np.array([self.cls_value_bin]), sample_binned_expression))
        
        current_seq_len = len(current_gene_ids_list)

        # 4. Padding
        padding_len = self.max_seq_len - current_seq_len
        
        if padding_len < 0: # Truncate if current_seq_len > max_seq_len
            # This should ideally not happen if max_seq_len is calculated correctly based on adata.var
            # print(f"Warning: current_seq_len ({current_seq_len}) > max_seq_len ({self.max_seq_len}). Truncating.")
            padded_gene_ids_np = np.array(current_gene_ids_list[:self.max_seq_len])
            padded_values_np = np.array(current_values_list[:self.max_seq_len])
            # For truncated sequences, the mask should be all False (no padding tokens)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        elif padding_len > 0:
            padded_gene_ids_np = np.pad(current_gene_ids_list, (0, padding_len), 
                                        mode='constant', constant_values=self.vocab.pad_idx)
            padded_values_np = np.pad(current_values_list, (0, padding_len), 
                                      mode='constant', constant_values=self.pad_value)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            padding_mask[current_seq_len:] = True # Mark padded positions as True
        else: # No padding needed, current_seq_len == max_seq_len
            padded_gene_ids_np = np.array(current_gene_ids_list)
            padded_values_np = np.array(current_values_list)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool) # All False

        return {
            "gene_ids": torch.tensor(padded_gene_ids_np, dtype=torch.long),
            "values": torch.tensor(padded_values_np, dtype=torch.float32), # Binned values are inputs to an embedding or float features
            "padding_mask": padding_mask, # True for padded positions
            "fractions": sample_fractions, # Target for deconvolution
        }

def create_dataloaders(
    train_adata: sc.AnnData, 
    valid_adata: sc.AnnData, 
    vocab: GeneVocab, 
    batch_size: int, 
    expression_layer: str = 'binned_expression',
    cls_value_bin: int = 0, 
    pad_value: int = -2, 
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training and validation.

    Args:
        train_adata: Preprocessed training AnnData object.
        valid_adata: Preprocessed validation AnnData object.
        vocab: scGPT vocabulary (GeneVocab).
        batch_size: Number of samples per batch.
        expression_layer: Layer in AnnData for binned expression.
        cls_value_bin: Bin value for CLS token.
        pad_value: Padding value for expression data.
        num_workers: Number of subprocesses for data loading.

    Returns:
        A tuple containing (train_loader, valid_loader).
    """
    if not train_adata.var_names.equals(valid_adata.var_names):
        raise ValueError("Train and validation AnnData objects must have the same var_names (genes) in the same order.")
    
    # max_seq_len includes the CLS token
    max_seq_len = len(train_adata.var_names) + 1 
    print(f"Max sequence length for DataLoader (genes + CLS token): {max_seq_len}")

    train_dataset = PseudoBulkDataset(
        train_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value
    )
    valid_dataset = PseudoBulkDataset(
        valid_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=True # Typically True for training for stable batch sizes
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=False
    )
    print(f"Created DataLoaders: Train batches={len(train_loader)}, Valid batches={len(valid_loader)}")
    return train_loader, valid_loader

if __name__ == '__main__':
    print("--- PyTorch Dataset and DataLoader Example for Deconvolution ---")

    # --- Configuration for Dummy Data ---
    N_TRAIN_SAMPLES, N_VALID_SAMPLES = 120, 40 # Reduced for faster example
    N_GENES = 50  # Reduced gene count for smaller sequences
    N_CELL_TYPES = 8
    BATCH_SIZE_EXAMPLE = 16 # Smaller batch size
    EXPRESSION_LAYER_NAME = 'binned_expression'
    # CLS token's expression value (which bin it falls into or a special value)
    CLS_TOKEN_VALUE_BIN_EXAMPLE = 0 
    # Value used for padding expression data points (genes)
    # This should ideally be a value that the model's embedding layer for values can ignore (e.g., padding_idx)
    # Or a value outside the typical range of binned data.
    EXPRESSION_PAD_VALUE_EXAMPLE = -2 
    # --- End of Configuration ---

    print("\n1. Creating dummy vocabulary...")
    gene_names_example = [f"gene_{i}" for i in range(N_GENES)]
    # Ensure standard special tokens are part of the vocab for the dummy GeneVocab
    special_tokens_example = ["<pad>", "<cls>", "<eoc>", "<mask>", "<eos>"] 
    example_vocab_instance = GeneVocab(gene_names_example, specials=special_tokens_example)
    print(f"Dummy Vocab created. Size: {len(example_vocab_instance)}. CLS ID: {example_vocab_instance.cls_idx}, PAD ID: {example_vocab_instance.pad_idx}")

    print("\n2. Creating dummy preprocessed AnnData objects...")
    def create_dummy_anndata(n_samples, var_names, n_cell_types, layer_name):
        # Simulate binned expression data (e.g., 0 to 50 for 51 bins)
        binned_data = np.random.randint(0, 51, size=(n_samples, len(var_names))).astype(np.float32)
        # Simulate cell type fractions (sum to 1)
        fractions_data = np.random.rand(n_samples, n_cell_types)
        fractions_data = fractions_data / fractions_data.sum(axis=1, keepdims=True)
        
        adata = sc.AnnData(
            X=np.random.rand(n_samples, len(var_names)).astype(np.float32), # Raw-like data in .X (not used by Dataset)
            layers={layer_name: binned_data},
            obs=pd.DataFrame(fractions_data, 
                             index=[f"sample_s{i}" for i in range(n_samples)],
                             columns=[f"CellType_{ct}" for ct in range(n_cell_types)]),
            var=pd.DataFrame(index=var_names)
        )
        return adata

    train_adata_dummy = create_dummy_anndata(N_TRAIN_SAMPLES, gene_names_example, N_CELL_TYPES, EXPRESSION_LAYER_NAME)
    valid_adata_dummy = create_dummy_anndata(N_VALID_SAMPLES, gene_names_example, N_CELL_TYPES, EXPRESSION_LAYER_NAME)
    print(f"Dummy train AnnData: {train_adata_dummy.shape}, Layer '{EXPRESSION_LAYER_NAME}' found.")
    print(f"Dummy valid AnnData: {valid_adata_dummy.shape}, Layer '{EXPRESSION_LAYER_NAME}' found.")
    print(f"Example fractions (first sample): {train_adata_dummy.obs.iloc[0].values}")

    print("\n3. Creating DataLoaders...")
    try:
        train_loader_example, valid_loader_example = create_dataloaders(
            train_adata_dummy,
            valid_adata_dummy,
            example_vocab_instance,
            batch_size=BATCH_SIZE_EXAMPLE,
            expression_layer=EXPRESSION_LAYER_NAME,
            cls_value_bin=CLS_TOKEN_VALUE_BIN_EXAMPLE,
            pad_value=EXPRESSION_PAD_VALUE_EXAMPLE,
            num_workers=0 
        )
        print("\nDataLoaders created successfully.")

        print("\n4. Inspecting a sample batch from train_loader...")
        sample_batch = next(iter(train_loader_example))
        for key, value in sample_batch.items():
            print(f"  Key: '{key}', Shape: {value.shape}, Dtype: {value.dtype}")
        
        # Detailed check of the first sample in the batch
        print("\n  Detailed check of first sample in batch:")
        first_sample_gene_ids = sample_batch['gene_ids'][0]
        first_sample_values = sample_batch['values'][0]
        first_sample_padding_mask = sample_batch['padding_mask'][0]
        
        print(f"    CLS token ID (expected {example_vocab_instance.cls_idx}): {first_sample_gene_ids[0].item()}")
        print(f"    CLS token value (expected {CLS_TOKEN_VALUE_BIN_EXAMPLE}): {first_sample_values[0].item()}")
        
        # Find first padding position to verify padding, if any
        # max_s_len is genes + CLS
        actual_seq_len_plus_cls = N_GENES + 1 
        if actual_seq_len_plus_cls < first_sample_gene_ids.shape[0]: # Check if padding exists
            print(f"    PAD token ID at first padded position (expected {example_vocab_instance.pad_idx}): {first_sample_gene_ids[actual_seq_len_plus_cls].item()}")
            print(f"    PAD value at first padded position (expected {EXPRESSION_PAD_VALUE_EXAMPLE}): {first_sample_values[actual_seq_len_plus_cls].item()}")
            print(f"    Padding mask at first padded position (expected True): {first_sample_padding_mask[actual_seq_len_plus_cls].item()}")
        else:
            print("    Sequence is not padded (max_seq_len equals actual_seq_len).")
            print(f"    Last gene ID in sequence: {first_sample_gene_ids[-1].item()}")
            print(f"    Last value in sequence: {first_sample_values[-1].item()}")
            print(f"    Padding mask at last position (expected False): {first_sample_padding_mask[-1].item()}")


    except Exception as e:
        print(f"\nAn error occurred during DataLoader example: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- End of DataLoader Example ---")
    print("Reminder: For actual use, replace dummy data with your real, preprocessed AnnData objects and a valid scGPT vocabulary.")
```
