import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scanpy as sc
import json
from pathlib import Path
import sys
import os
import time
import argparse
import random
from typing import Optional, Dict, Any, List, Union, Tuple

# --- Utility Function for Reproducibility ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Potentially set torch.backends.cudnn.deterministic and benchmark
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

# --- 1. Data Preparation (from prepare_scgpt_data.py) ---
def load_and_prepare_data(
    adata_path_str: str, 
    model_vocab_path_str: str # Path to the vocab.json from the model directory
) -> Tuple[Optional[sc.AnnData], Optional[Any], Optional[List[str]]]:
    """
    Loads a single-cell AnnData object and an scGPT model vocabulary, 
    then filters the AnnData object to include only genes common to both.

    Args:
        adata_path_str (str): Path to the .h5ad AnnData file.
        model_vocab_path_str (str): Path to the vocab.json file from the model.

    Returns:
        tuple: A tuple containing:
            - filtered_adata (sc.AnnData | None): AnnData filtered to common genes.
            - vocab_object (GeneVocab | None): Loaded GeneVocab object.
            - common_genes (list[str] | None): List of common gene names.
    """
    print(f"Loading AnnData from: {adata_path_str}")
    adata_path = Path(adata_path_str)
    if not adata_path.exists():
        print(f"Error: AnnData file not found at {adata_path_str}")
        return None, None, None
    
    try:
        adata = sc.read_h5ad(adata_path)
        print(f"Successfully loaded AnnData object with {adata.n_obs} cells and {adata.n_vars} genes.")
    except Exception as e:
        print(f"Error reading AnnData file at {adata_path_str}: {e}")
        return None, None, None

    vocab_file_path = Path(model_vocab_path_str)
    print(f"Loading vocabulary from: {vocab_file_path}")

    if not vocab_file_path.exists():
        print(f"Error: Vocabulary file (vocab.json) not found at {vocab_file_path}")
        return None, None, None

    try:
        # Use the GeneVocab class that will be defined later in this script
        vocab_object = GeneVocab.from_file(vocab_file_path)
        if isinstance(vocab_object.stoi, dict):
            model_gene_list_from_vocab = list(vocab_object.stoi.keys())
        else: # Fallback if stoi is not a dict (should not happen with proper GeneVocab)
            print("Warning: vocab_object.stoi is not a dictionary. Attempting to infer genes.")
            model_gene_list_from_vocab = [vocab_object.itos[i] for i in range(len(vocab_object))]

        print(f"Successfully loaded vocabulary with {len(model_gene_list_from_vocab)} genes/tokens.")
    except Exception as e:
        print(f"Error loading or parsing vocabulary file {vocab_file_path}: {e}")
        return None, None, None

    original_adata_genes = list(adata.var_names)
    print(f"Number of genes in original AnnData: {len(original_adata_genes)}")

    # Identify common genes (case-sensitive)
    common_genes = sorted(list(set(original_adata_genes) & set(model_gene_list_from_vocab)))
    
    if not common_genes:
        print("Error: No common genes found between AnnData and vocabulary.")
        print("Please check gene name conventions (e.g., ENSEMBL IDs vs. gene symbols) and species.")
        return None, vocab_object, None
        
    print(f"Number of common genes found: {len(common_genes)}")

    # Filter AnnData to retain only common genes
    adata_filtered = adata[:, common_genes].copy()
    print(f"Filtered AnnData to {adata_filtered.n_vars} common genes.")

    return adata_filtered, vocab_object, common_genes


# --- 2. Pseudo-bulk Generation (from create_pseudo_bulk.py) ---
def generate_pseudo_bulk_samples(
    adata_filtered: sc.AnnData,
    cell_type_col: str,
    n_pseudo_bulk_samples: int = 1000,
    n_cells_per_sample_min: int = 50,
    n_cells_per_sample_max: int = 200,
    aggregation_method: str = "sum"
) -> Optional[sc.AnnData]:
    print(f"Starting pseudo-bulk sample generation: {n_pseudo_bulk_samples} samples...")
    print(f"Params: min_cells={n_cells_per_sample_min}, max_cells={n_cells_per_sample_max}, aggregation='{aggregation_method}'")

    if cell_type_col not in adata_filtered.obs.columns:
        print(f"Error: Cell type column '{cell_type_col}' not found. Available: {list(adata_filtered.obs.columns)}")
        return None
    if aggregation_method not in ["sum", "mean"]:
        print(f"Error: Unknown aggregation_method '{aggregation_method}'. Choose 'sum' or 'mean'.")
        return None
    if n_cells_per_sample_max < n_cells_per_sample_min:
        print(f"Error: n_cells_per_sample_max < n_cells_per_sample_min.")
        return None
    if adata_filtered.n_obs < n_cells_per_sample_min:
        print(f"Error: Not enough cells ({adata_filtered.n_obs}) for min_cells_per_sample ({n_cells_per_sample_min}).")
        return None

    all_gene_names = list(adata_filtered.var_names)
    unique_cell_types = sorted(list(adata_filtered.obs[cell_type_col].astype('category').cat.categories))
    
    if not unique_cell_types:
        print(f"Error: No unique cell types found in column '{cell_type_col}'.")
        return None
    print(f"Found {len(unique_cell_types)} unique cell types: {unique_cell_types}")

    pseudo_bulk_expressions, pseudo_bulk_fractions = [], []
    is_sparse = isinstance(adata_filtered.X, (sc.sparse.csr_matrix, sc.sparse.csc_matrix))
    print(f"Input AnnData.X is {'sparse' if is_sparse else 'dense'}.")

    for i in range(n_pseudo_bulk_samples):
        n_cells = random.randint(n_cells_per_sample_min, n_cells_per_sample_max) \
            if n_cells_per_sample_min != n_cells_per_sample_max else n_cells_per_sample_min
        
        sampled_indices = random.sample(range(adata_filtered.n_obs), k=n_cells)
        sampled_X = adata_filtered.X[sampled_indices, :]
        
        agg_expr = sampled_X.sum(axis=0) if aggregation_method == "sum" else sampled_X.mean(axis=0)
        if hasattr(agg_expr, "A1"): agg_expr = agg_expr.A1
        elif hasattr(agg_expr, "toarray"): agg_expr = agg_expr.toarray().flatten()
        
        pseudo_bulk_expressions.append(agg_expr)

        sampled_cell_types = adata_filtered.obs[cell_type_col].iloc[sampled_indices]
        type_counts = sampled_cell_types.value_counts().reindex(unique_cell_types, fill_value=0.0)
        pseudo_bulk_fractions.append((type_counts / n_cells).values)

        if (i + 1) % (n_pseudo_bulk_samples // 10 if n_pseudo_bulk_samples >= 10 else 1) == 0:
            print(f"  Generated {i + 1}/{n_pseudo_bulk_samples} pseudo-bulk samples...")

    try:
        pseudo_bulk_X_np = np.array(pseudo_bulk_expressions)
    except Exception as e:
        print(f"Error converting expressions to NumPy array: {e}")
        return None
    fractions_df = pd.DataFrame(pseudo_bulk_fractions, columns=unique_cell_types, 
                                index=[f"pseudo_bulk_sample_{i}" for i in range(n_pseudo_bulk_samples)])
    var_df = pd.DataFrame(index=all_gene_names)
    
    try:
        pseudo_bulk_adata = sc.AnnData(X=pseudo_bulk_X_np, obs=fractions_df, var=var_df)
    except Exception as e:
        print(f"Error creating final AnnData for pseudo-bulk data: {e}")
        return None

    print(f"\nPseudo-bulk AnnData created: X shape {pseudo_bulk_adata.X.shape}, obs shape {pseudo_bulk_adata.obs.shape}")
    return pseudo_bulk_adata


# --- 3. Preprocessing Pseudo-bulk (from preprocess_bulk_for_scgpt.py) ---
def preprocess_pseudo_bulk_for_scgpt(
    pseudo_bulk_adata: sc.AnnData,
    target_sum: Optional[float] = 1e4,
    n_bins: int = 51,
    hvg_col_name: Optional[str] = None
) -> Optional[sc.AnnData]:
    print("Starting preprocessing of pseudo-bulk data...")
    adata_processed = pseudo_bulk_adata.copy()
    print(f"  Input shape: {adata_processed.shape}")

    if hvg_col_name:
        if hvg_col_name in adata_processed.var.columns and adata_processed.var[hvg_col_name].dtype == bool:
            print(f"  Filtering by HVG column '{hvg_col_name}'.")
            adata_processed = adata_processed[:, adata_processed.var[hvg_col_name]].copy()
            print(f"  Shape after HVG filter: {adata_processed.shape}")
            if adata_processed.n_vars == 0: print("Warning: No genes left after HVG filtering."); return None
        else: print(f"Warning: HVG column '{hvg_col_name}' not found or not boolean. Skipping HVG filtering.")

    if target_sum is not None and target_sum > 0:
        if adata_processed.X is None: print("Error: .X is None, cannot normalize."); return None
        print(f"  Normalizing total counts to {target_sum}.")
        sc.pp.normalize_total(adata_processed, target_sum=target_sum)
        adata_processed.layers['normalized'] = adata_processed.X.copy()
    else:
        print("  Skipping normalization (target_sum is None or <=0).")
        if adata_processed.X is None: print("Error: .X is None and skipping norm. Cannot proceed."); return None

    print("  Applying log1p transformation.")
    if 'normalized' in adata_processed.layers: # If normalized, .X was updated
        sc.pp.log1p(adata_processed) # Modifies .X in-place
        adata_processed.layers['log1p'] = adata_processed.X.copy()
    else: # If not normalized, apply to .X directly
        adata_processed.layers['log1p'] = np.log1p(adata_processed.X)
        adata_processed.X = adata_processed.layers['log1p'].copy() # Update .X to log1p

    print(f"  Binning log-transformed values into {n_bins} bins.")
    data_to_bin = adata_processed.layers['log1p']
    if data_to_bin is None: print("Error: layers['log1p'] is None. Cannot bin."); return None
    
    finite_data = data_to_bin[np.isfinite(data_to_bin)]
    if finite_data.size == 0: print("Error: No finite data for binning."); return None
    min_val, max_val = np.min(finite_data), np.max(finite_data)
    print(f"  Data range for binning: min={min_val:.4f}, max={max_val:.4f}")

    if min_val == max_val:
        print("Warning: All values for binning are identical. Assigning to bin 0.")
        binned_expression = np.zeros(data_to_bin.shape, dtype=int)
    else:
        bin_edges = np.linspace(min_val, max_val, num=n_bins - 1)
        binned_expression = np.digitize(data_to_bin, bin_edges, right=False)
    
    adata_processed.layers['binned_expression'] = binned_expression.astype(np.int32)
    print(f"  Binning complete. Min bin: {np.min(binned_expression)}, Max bin: {np.max(binned_expression)}")
    if np.max(binned_expression) >= n_bins: print("Warning: Max bin index >= n_bins. Check logic.")
    
    print("Preprocessing complete.")
    return adata_processed


# --- 4. Model Definition (from deconvolution_model_def.py) ---
# First, define GeneVocab and TransformerModel (can be dummy if scgpt not installed)
try:
    from scgpt.model import TransformerModel as scGPTTransformerModel
    from scgpt.tokenizer import GeneVocab as scGPTGeneVocab
    scgpt_available = True
except ImportError:
    print("Warning: scGPT library not found. Using DUMMY classes for TransformerModel and GeneVocab.")
    scgpt_available = False
    class scGPTTransformerModel(nn.Module): # Dummy
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.d_model = kwargs.get("d_model", 128)
            self.dummy_encoder = nn.Linear(kwargs.get("ntoken", 100), self.d_model)
            print(f"DUMMY scGPTTransformerModel initialized. d_model={self.d_model}")
        def load_state_dict(self, state_dict, strict=True): print("DUMMY load_state_dict called."); return [],[]
        def forward(self, src, values, src_key_padding_mask): # Match expected inputs
            # Simplified: use 'values' if it matches ntoken, or 'src' if it's sequence length
            # This dummy is very basic and won't reflect true scGPT behavior.
            # Let's assume 'values' is [B, SeqLen] and we want to project it to d_model.
            # For this dummy, let's assume ntoken in init was seq_len.
            if values.shape[1] == self.dummy_encoder.in_features:
                 simulated_emb = self.dummy_encoder(values) # [B, d_model]
            else: # Fallback if dimensions don't match
                 simulated_emb = torch.randn(values.shape[0], self.d_model, device=values.device)
            return {"cell_emb": simulated_emb} # Expected output structure

    class scGPTGeneVocab: # Dummy
        def __init__(self, gene_list, specials=None):
            self.stoi = {}
            idx = 0
            if specials:
                for s in specials: self.stoi[s] = idx; idx+=1
            for g in gene_list: 
                if g not in self.stoi: self.stoi[g] = idx; idx+=1
            self.itos = {i:s for s,i in self.stoi.items()}
            self._pad_token, self._cls_token = "<pad>", "<cls>"
            print(f"DUMMY scGPTGeneVocab initialized. Size: {len(self.stoi)}")
        def __getitem__(self, token): return self.stoi.get(token, -1)
        def __len__(self): return len(self.stoi)
        @property
        def pad_idx(self): return self.stoi.get(self._pad_token, 0)
        @property
        def cls_idx(self): return self.stoi.get(self._cls_token, 1)
        @classmethod
        def from_file(cls, fp):
            with open(fp, 'r') as f: data = json.load(f)
            if isinstance(data, dict): return cls(list(data.keys()))
            return cls(data, specials=["<pad>", "<cls>"])

# Make GeneVocab globally accessible, pointing to the real or dummy one
GeneVocab = scGPTGeneVocab
TransformerModel = scGPTTransformerModel

def create_deconvolution_model(
    model_dir_path: str, vocab_file_path: str, n_cell_types: int,
    scgpt_dropout_rate: Optional[float] = None, head_dropout_rate: float = 0.1,
    freeze_scgpt_base: bool = False, d_model_override: Optional[int] = None,
    nhead_override: Optional[int] = None, nlayers_override: Optional[int] = None,
    pad_value: int = -2
) -> Optional[nn.Module]:
    print(f"Creating deconvolution model from dir: {model_dir_path}, vocab: {vocab_file_path}")
    model_config_file = Path(model_dir_path) / "args.json"
    model_weights_file = Path(model_dir_path) / "best_model.pt"
    
    if not all([f.exists() for f in [model_config_file, model_weights_file, Path(vocab_file_path)]]):
        print("Error: One or more model/vocab files not found.")
        if not scgpt_available: print("scGPT not available, cannot use real model. Dummy will be used if example proceeds."); return None
        # If scgpt IS available but files missing, it's a definite error for real run.
        if scgpt_available: return None 
    
    try:
        if scgpt_available: # Load real model configs only if scGPT is real
            with open(model_config_file, "r") as f: model_configs = json.load(f)
        else: # Dummy configs for dummy model
            model_configs = {"embsize": 128, "nheads": 2, "nlayers": 2, "pad_token":"<pad>", "pad_value":-2,
                             "dropout":0.1, "input_emb_style":"continuous", "cell_emb_style":"cls", "n_cls":1, "nlayers_cls":1}
        vocab = GeneVocab.from_file(Path(vocab_file_path))
        ntoken = len(vocab)
        pad_token = model_configs.get("pad_token", "<pad>")
        print(f"Vocab size: {ntoken}, Pad token: {pad_token}")
    except Exception as e: print(f"Error loading config/vocab: {e}"); return None

    d_model = d_model_override if d_model_override else model_configs.get("embsize", 128)
    nhead = nhead_override if nhead_override else model_configs.get("nheads", 2)
    nlayers = nlayers_override if nlayers_override else model_configs.get("nlayers", 2)
    dropout = scgpt_dropout_rate if scgpt_dropout_rate is not None else model_configs.get("dropout", 0.1)
    
    scgpt_model_instance = TransformerModel(
        ntoken=ntoken, d_model=d_model, nhead=nhead, d_hid=model_configs.get("d_hid", d_model*4),
        nlayers=nlayers, vocab=vocab, dropout=dropout, pad_token=pad_token, 
        pad_value=model_configs.get("pad_value", pad_value),
        # Other params often found in args.json
        input_emb_style=model_configs.get("input_emb_style", "continuous"),
        n_input_bins=model_configs.get("n_input_bins", 0),
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
        n_cls=model_configs.get("n_cls",1),
        nlayers_cls=model_configs.get("nlayers_cls",1),
        use_fast_transformer=model_configs.get("use_fast_transformer", False) if scgpt_available else False, # Dummy doesn't use this
        pre_norm=model_configs.get("pre_norm", False)
    )
    print(f"TransformerModel instantiated (d_model={d_model}).")

    if scgpt_available and model_weights_file.exists(): # Load real weights if real model and file exists
        try:
            state_dict = torch.load(model_weights_file, map_location=torch.device('cpu'))
            missing, unexpected = scgpt_model_instance.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if unexpected: print(f"  Sample unexpected keys: {unexpected[:3]}")
        except Exception as e: print(f"Error loading weights: {e}")
    elif not scgpt_available:
        print("Skipping weights loading for DUMMY scGPT model.")
    else: # scGPT available but weights file does not exist
        print(f"Warning: Model weights file {model_weights_file} not found. Initializing from scratch.")


    scgpt_model_instance.deconv_head = nn.Sequential(
        nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(head_dropout_rate),
        nn.Linear(d_model // 2, n_cell_types), nn.Softmax(dim=-1)
    )
    print(f"Deconvolution head added (output_dim={n_cell_types}).")

    if freeze_scgpt_base:
        print("Freezing base model parameters.")
        for name, param in scgpt_model_instance.named_parameters():
            if 'deconv_head' not in name: param.requires_grad = False
            else: param.requires_grad = True # Ensure head is trainable
    else:
        print("All model parameters are trainable.")
        for param in scgpt_model_instance.parameters(): param.requires_grad = True
            
    return scgpt_model_instance


# --- 5. Dataloaders (from deconvolution_dataloaders.py) ---
class PseudoBulkDataset(Dataset):
    def __init__(self, pseudo_bulk_adata: sc.AnnData, vocab: GeneVocab, max_seq_len: int, 
                 expression_layer: str, cls_value_bin: int, pad_value: int):
        self.adata = pseudo_bulk_adata
        self.genes = list(pseudo_bulk_adata.var_names)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.expression_layer = expression_layer
        self.cls_value_bin = cls_value_bin
        self.pad_value = pad_value
        if not self.genes: raise ValueError("AnnData var_names is empty.")
        self.gene_ids_no_cls = [self.vocab[g] for g in self.genes]
        if any(gid == -1 for gid in self.gene_ids_no_cls):
            print(f"Warning: Some genes not in vocab: {[self.genes[i] for i,gid in enumerate(self.gene_ids_no_cls) if gid == -1][:5]}")

    def __len__(self): return self.adata.n_obs
    def __getitem__(self, idx):
        raw_expr = self.adata.layers[self.expression_layer][idx]
        expr_flat = raw_expr.toarray().flatten() if hasattr(raw_expr, "toarray") else np.asarray(raw_expr).flatten()
        
        fractions_np = self.adata.obs.iloc[idx, :].values.astype(np.float32)
        
        ids_list = [self.vocab.cls_idx] + self.gene_ids_no_cls
        vals_list = np.concatenate((np.array([self.cls_value_bin]), expr_flat))
        
        current_len = len(ids_list)
        padding_len = self.max_seq_len - current_len
        
        if padding_len < 0: # Truncate
            ids_final = np.array(ids_list[:self.max_seq_len])
            vals_final = np.array(vals_list[:self.max_seq_len])
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        elif padding_len > 0: # Pad
            ids_final = np.pad(ids_list, (0, padding_len), mode='constant', constant_values=self.vocab.pad_idx)
            vals_final = np.pad(vals_list, (0, padding_len), mode='constant', constant_values=self.pad_value)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[current_len:] = True
        else: # No change
            ids_final = np.array(ids_list)
            vals_final = np.array(vals_list)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            
        return {"gene_ids": torch.tensor(ids_final, dtype=torch.long),
                "values": torch.tensor(vals_final, dtype=torch.float32),
                "padding_mask": mask, 
                "fractions": torch.tensor(fractions_np, dtype=torch.float32)}

def create_dataloaders(
    train_adata: sc.AnnData, valid_adata: sc.AnnData, vocab: GeneVocab, batch_size: int, 
    expression_layer: str, cls_value_bin: int, pad_value: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    if not train_adata.var_names.equals(valid_adata.var_names):
        raise ValueError("Train/Validation AnnData must have same var_names.")
    max_seq_len = len(train_adata.var_names) + 1
    print(f"Max sequence length for DataLoader: {max_seq_len}")

    train_ds = PseudoBulkDataset(train_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value)
    valid_ds = PseudoBulkDataset(valid_adata, vocab, max_seq_len, expression_layer, cls_value_bin, pad_value)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=num_workers > 0, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=num_workers > 0, drop_last=False)
    print(f"DataLoaders created. Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    return train_loader, valid_loader


# --- 6. Training Utilities (from training_utils.py) ---
def get_loss_function(loss_type: str = "L1") -> nn.Module:
    if loss_type == "L1": print("Using L1 Loss."); return nn.L1Loss()
    if loss_type == "MSE": print("Using MSE Loss."); return nn.MSELoss()
    if loss_type == "KLDivergence": 
        print("Using KL Divergence Loss. Ensure model output is log-probs and target is probs.")
        return nn.KLDivLoss(reduction='batchmean')
    raise ValueError(f"Unsupported loss: {loss_type}")

def get_optimizer(model: nn.Module, lr: float, opt_type: str, weight_decay: float) -> optim.Optimizer:
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    if opt_type == "AdamW": print(f"Using AdamW (LR={lr}, WD={weight_decay})."); return optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    if opt_type == "Adam": print(f"Using Adam (LR={lr}, WD={weight_decay})."); return optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_type}")

def get_scheduler(optimizer: optim.Optimizer, sched_type: Optional[str], total_steps: Optional[int], warmup_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if sched_type is None or sched_type.lower() == "none": print("No LR scheduler."); return None
    if sched_type == "OneCycleLR":
        if total_steps is None: raise ValueError("OneCycleLR needs total_steps.")
        pct_start = float(warmup_steps) / float(total_steps) if warmup_steps > 0 and total_steps > 0 else 0.3
        print(f"Using OneCycleLR (MaxLR={optimizer.defaults['lr']}, TotalSteps={total_steps}, WarmupPct={pct_start:.3f}).")
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer.defaults['lr'], total_steps=total_steps, pct_start=pct_start)
    if sched_type == "LinearWarmup":
        if warmup_steps <= 0: print("LinearWarmup with 0 steps, effectively constant LR."); return optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        def lr_lambda(step): return float(step+1)/float(warmup_steps) if step < warmup_steps else 1.0
        print(f"Using LinearWarmup (LambdaLR) for {warmup_steps} steps, then constant LR."); return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if sched_type == "ReduceLROnPlateau": print("Using ReduceLROnPlateau."); return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
    raise ValueError(f"Unsupported scheduler: {sched_type}")


# --- 7. Fine-tuning Loop (from fine_tuning_loop.py) ---
def run_fine_tuning(
    model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: optim.Optimizer,
    loss_fn: nn.Module, n_epochs: int, device: torch.device, loss_type_str: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], checkpoint_dir: str, best_model_name: str
):
    print(f"Starting fine-tuning: {n_epochs} epochs on {device}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            gene_ids, values, padding_mask, true_fractions = \
                batch["gene_ids"].to(device), batch["values"].to(device), \
                batch["padding_mask"].to(device), batch["fractions"].to(device)
            
            optimizer.zero_grad()
            scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
            if "cell_emb" not in scgpt_output: raise KeyError("Output missing 'cell_emb'.")
            
            cell_embedding = scgpt_output["cell_emb"]
            if not hasattr(model, 'deconv_head'): raise AttributeError("Model missing 'deconv_head'.")
            predicted_fractions = model.deconv_head(cell_embedding)

            if loss_type_str == "KLDivergence":
                loss = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
            else:
                loss = loss_fn(predicted_fractions, true_fractions)
            
            loss.backward()
            optimizer.step()
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            total_train_loss += loss.item()
            if len(train_loader) > 4 and batch_idx % (len(train_loader)//4) == 0 and batch_idx > 0:
                 print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.3e}")

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                gene_ids, values, padding_mask, true_fractions = \
                    batch["gene_ids"].to(device), batch["values"].to(device), \
                    batch["padding_mask"].to(device), batch["fractions"].to(device)
                scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
                cell_embedding = scgpt_output["cell_emb"]
                predicted_fractions = model.deconv_head(cell_embedding)
                if loss_type_str == "KLDivergence":
                    val_loss_batch = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
                else:
                    val_loss_batch = loss_fn(predicted_fractions, true_fractions)
                total_val_loss += val_loss_batch.item()
        avg_val_loss = total_val_loss / len(valid_loader)
        history["val_loss"].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} Summary: Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.3e} | Time: {time.time()-epoch_start_time:.2f}s")

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(checkpoint_dir) / best_model_name)
            print(f"  Best model saved (Val Loss: {best_val_loss:.4f})")
            
    print(f"\nFine-tuning complete. Best validation loss: {best_val_loss:.4f}")
    return history

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scGPT Deconvolution Fine-tuning Script")
    # Data paths
    parser.add_argument("--adata_file_path", type=str, required=True, help="Path to input single-cell AnnData (.h5ad)")
    parser.add_argument("--model_dir_path", type=str, required=True, help="Path to pre-trained scGPT model directory")
    parser.add_argument("--vocab_file_path", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--output_dir", type=str, default="./deconv_finetune_output", help="Directory for checkpoints and results")
    
    # Pseudo-bulk generation
    parser.add_argument("--cell_type_col", type=str, required=True, help="Column in adata.obs for cell type annotations")
    parser.add_argument("--n_pseudo_bulk_samples", type=int, default=1000)
    parser.add_argument("--min_cells_per_sample", type=int, default=50)
    parser.add_argument("--max_cells_per_sample", type=int, default=200)
    parser.add_argument("--aggregation_method", type=str, default="sum", choices=["sum", "mean"])
    
    # Preprocessing
    parser.add_argument("--norm_target_sum", type=float, default=1e4, help="Target sum for normalization (0 or negative to skip)")
    parser.add_argument("--n_bins", type=int, default=51, help="Number of bins for expression discretization")
    parser.add_argument("--hvg_col", type=str, default=None, help="Optional: column in .var marking HVGs for filtering pseudo-bulk data")

    # Model params
    parser.add_argument("--freeze_scgpt_base", action='store_true', help="Freeze base scGPT model parameters")
    parser.add_argument("--head_dropout_rate", type=float, default=0.1)
    parser.add_argument("--scgpt_dropout_rate", type=float, default=None, help="Override scGPT's internal dropout (None uses original)")
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="L1", choices=["L1", "MSE", "KLDivergence"])
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "Adam"])
    parser.add_argument("--scheduler_type", type=str, default="OneCycleLR", choices=["OneCycleLR", "LinearWarmup", "ReduceLROnPlateau", "None"])
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_frac", type=float, default=0.1, help="Fraction of total steps for warmup")
    
    # Dataset/DataLoader params
    parser.add_argument("--cls_value_bin", type=int, default=0, help="Bin index for CLS token's value in Dataset")
    parser.add_argument("--expression_pad_value", type=int, default=-2, help="Padding value for expression data in Dataset")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    print("--- Configuration ---")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("---------------------\n")

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load and Prepare Single-Cell Data ---
    print("\n--- Step 1: Loading and Preparing Single-Cell Data ---")
    filtered_adata, vocab_obj, common_genes = load_and_prepare_data(args.adata_file_path, args.vocab_file_path)
    if filtered_adata is None or vocab_obj is None:
        print("Exiting due to error in data loading/preparation.")
        sys.exit(1)

    # --- 2. Generate Pseudo-bulk Samples ---
    print("\n--- Step 2: Generating Pseudo-bulk Samples ---")
    pseudo_bulk_adata = generate_pseudo_bulk_samples(
        filtered_adata, args.cell_type_col, args.n_pseudo_bulk_samples,
        args.min_cells_per_sample, args.max_cells_per_sample, args.aggregation_method
    )
    if pseudo_bulk_adata is None:
        print("Exiting due to error in pseudo-bulk sample generation.")
        sys.exit(1)

    # --- 3. Split and Preprocess Pseudo-bulk Data ---
    print("\n--- Step 3: Splitting and Preprocessing Pseudo-bulk Data ---")
    # Simple split (can be improved with sklearn's train_test_split for stratification if needed)
    n_total_samples = pseudo_bulk_adata.n_obs
    val_frac = 0.2 # Example: 20% for validation
    n_val_samples = int(n_total_samples * val_frac)
    n_train_samples = n_total_samples - n_val_samples
    
    # Shuffle indices before splitting
    shuffled_indices = np.random.permutation(n_total_samples)
    train_indices = shuffled_indices[:n_train_samples]
    valid_indices = shuffled_indices[n_train_samples:]

    train_pb_adata_raw = pseudo_bulk_adata[train_indices, :].copy()
    valid_pb_adata_raw = pseudo_bulk_adata[valid_indices, :].copy()
    print(f"Split pseudo-bulk data: Train {train_pb_adata_raw.n_obs}, Validation {valid_pb_adata_raw.n_obs}")

    train_processed_adata = preprocess_pseudo_bulk_for_scgpt(
        train_pb_adata_raw, args.norm_target_sum if args.norm_target_sum > 0 else None, 
        args.n_bins, args.hvg_col
    )
    valid_processed_adata = preprocess_pseudo_bulk_for_scgpt(
        valid_pb_adata_raw, args.norm_target_sum if args.norm_target_sum > 0 else None, 
        args.n_bins, args.hvg_col
    )
    if train_processed_adata is None or valid_processed_adata is None:
        print("Exiting due to error in preprocessing pseudo-bulk data.")
        sys.exit(1)
    
    # Ensure var_names are consistent after potential HVG filtering
    # If HVG filtering is done, it should be based on the full pseudo_bulk_adata before split,
    # or applied consistently to both train/valid based on train set's HVGs.
    # For simplicity here, if hvg_col is used, we assume it's applied and then we take common genes.
    if args.hvg_col:
        common_vars_after_hvg = list(train_processed_adata.var_names.intersection(valid_processed_adata.var_names))
        train_processed_adata = train_processed_adata[:, common_vars_after_hvg].copy()
        valid_processed_adata = valid_processed_adata[:, common_vars_after_hvg].copy()
        print(f"Ensured common genes after HVG: {len(common_vars_after_hvg)} genes.")
        if not common_vars_after_hvg: print("Error: No common genes after HVG filtering."); sys.exit(1)
        # Re-assign vocab_obj to use only these common genes (if GeneVocab needs specific genes)
        # This is tricky; the vocab_obj from original data loading should be used.
        # The PseudoBulkDataset will use var_names from processed_adata.
        # The vocab_obj should still contain all original common_genes.

    # --- 4. Model Setup ---
    print("\n--- Step 4: Setting up Deconvolution Model ---")
    # n_cell_types is determined by the columns in .obs of pseudo_bulk_adata (which are the cell type fractions)
    n_cell_types = train_processed_adata.obs.shape[1]
    print(f"Number of cell types for deconvolution head: {n_cell_types}")
    
    model = create_deconvolution_model(
        model_dir_path=args.model_dir_path,
        vocab_file_path=args.vocab_file_path,
        n_cell_types=n_cell_types,
        scgpt_dropout_rate=args.scgpt_dropout_rate,
        head_dropout_rate=args.head_dropout_rate,
        freeze_scgpt_base=args.freeze_scgpt_base
        # d_model_override, nhead_override, nlayers_override can be added if needed
    )
    if model is None:
        print("Exiting due to error in model creation.")
        sys.exit(1)
    model.to(device)

    # --- 5. Dataloaders ---
    print("\n--- Step 5: Creating DataLoaders ---")
    train_loader, valid_loader = create_dataloaders(
        train_processed_adata, valid_processed_adata, vocab_obj, args.batch_size,
        expression_layer='binned_expression', # This is where preprocess_pseudo_bulk_for_scgpt stores it
        cls_value_bin=args.cls_value_bin,
        pad_value=args.expression_pad_value,
        num_workers=args.num_workers
    )

    # --- 6. Training Components ---
    print("\n--- Step 6: Setting up Training Components ---")
    loss_fn = get_loss_function(args.loss_type)
    optimizer = get_optimizer(model, args.learning_rate, args.optimizer_type, args.weight_decay)
    
    total_training_steps = args.n_epochs * len(train_loader)
    warmup_training_steps = int(args.warmup_frac * total_training_steps)
    print(f"Total training steps: {total_training_steps}, Warmup steps: {warmup_training_steps}")
    
    scheduler = get_scheduler(optimizer, args.scheduler_type, total_training_steps, warmup_training_steps)

    # --- 7. Run Fine-tuning ---
    print("\n--- Step 7: Starting Fine-tuning ---")
    best_model_save_name = f"best_model_deconv_epochs{args.n_epochs}_lr{args.learning_rate}_{args.loss_type}.pt"
    
    history = run_fine_tuning(
        model, train_loader, valid_loader, optimizer, loss_fn, args.n_epochs, device,
        args.loss_type, scheduler, args.output_dir, best_model_save_name
    )

    # --- 8. Save Training History (Optional) ---
    history_file = Path(args.output_dir) / f"training_history_epochs{args.n_epochs}_lr{args.learning_rate}_{args.loss_type}.json"
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"\nTraining history saved to {history_file}")
    except Exception as e:
        print(f"\nError saving training history: {e}")

    print("\n--- Fine-tuning Script Finished ---")

```
