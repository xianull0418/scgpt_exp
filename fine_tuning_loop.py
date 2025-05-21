import torch
import torch.nn as nn # For loss example and dummy model
import numpy as np
from pathlib import Path
import os
import time
# from scipy.stats import pearsonr # For optional metrics, can be added later

# Assume model (scGPT base + deconv_head), dataloaders, optimizer, loss_fn, scheduler 
# are defined elsewhere and passed to this function.

def run_fine_tuning(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    n_epochs: int,
    device: torch.device,
    loss_type_str: str, # e.g., "L1", "MSE", "KLDivergence"
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_dir: str = "./checkpoints_finetune", # Changed default to avoid conflict
    best_model_name: str = "best_deconv_model_finetuned.pt" # Changed default
):
    """
    Runs the fine-tuning loop for the deconvolution model.
    Assumes 'model' is an nn.Module where model.deconv_head exists and
    the main model's forward pass (e.g., from scGPT's TransformerModel)
    returns a dictionary containing 'cell_emb'.
    """
    print(f"Starting fine-tuning for {n_epochs} epochs on device: {device}")
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "lr": []} # To store loss and LR per epoch

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            gene_ids = batch["gene_ids"].to(device)
            values = batch["values"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            true_fractions = batch["fractions"].to(device)

            optimizer.zero_grad()

            # Forward pass through scGPT base model
            # model() calls the forward method of the scGPT TransformerModel.
            # This should return a dictionary.
            scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
            
            if "cell_emb" not in scgpt_output:
                # This error indicates a mismatch in how the model is structured or called.
                # The 'create_deconvolution_model' script assumes scGPT's TransformerModel is the base.
                raise KeyError("Output from scGPT model does not contain 'cell_emb'. "
                               "Ensure `cell_emb_style` was 'cls' during scGPT model initialization "
                               "and that the model's forward pass returns it.")
            
            cell_embedding = scgpt_output["cell_emb"]

            # Pass cell embedding through the deconvolution head
            # model.deconv_head should have been attached in 'create_deconvolution_model'
            if not hasattr(model, 'deconv_head'):
                 raise AttributeError("Model does not have a 'deconv_head' attribute. Ensure it was attached.")
            predicted_fractions = model.deconv_head(cell_embedding)

            # Calculate loss
            if loss_type_str == "KLDivergence":
                # KLDivLoss expects log-probabilities for input, probabilities for target
                # deconv_head output is softmax (probabilities)
                # Add a small epsilon for numerical stability with torch.log
                loss = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
            else:
                loss = loss_fn(predicted_fractions, true_fractions)
            
            loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Step-wise schedulers (like OneCycleLR, or custom LambdaLR for linear warmup)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            total_train_loss += loss.item()
            
            # Log training progress a few times per epoch
            if len(train_loader) > 4 and batch_idx % (len(train_loader) // 4) == 0 and batch_idx > 0 :
                 current_lr_batch = optimizer.param_groups[0]['lr']
                 print(f"  Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f} | LR: {current_lr_batch:.6e}")


        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])


        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        # all_preds_val = [] # For more detailed metrics later
        # all_targets_val = [] # For more detailed metrics later
        with torch.no_grad():
            for batch in valid_loader:
                gene_ids = batch["gene_ids"].to(device)
                values = batch["values"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                true_fractions = batch["fractions"].to(device)

                scgpt_output = model(src=gene_ids, values=values, src_key_padding_mask=padding_mask)
                if "cell_emb" not in scgpt_output: # Same handling as in training
                     raise KeyError("Validation: Output from scGPT model does not contain 'cell_emb'.")
                
                cell_embedding = scgpt_output["cell_emb"]
                predicted_fractions = model.deconv_head(cell_embedding)
                
                if loss_type_str == "KLDivergence":
                    val_loss_batch = loss_fn(torch.log(predicted_fractions + 1e-10), true_fractions)
                else:
                    val_loss_batch = loss_fn(predicted_fractions, true_fractions)
                total_val_loss += val_loss_batch.item()
                
                # Store predictions and targets if you want to compute more complex metrics
                # all_preds_val.append(predicted_fractions.cpu().numpy())
                # all_targets_val.append(true_fractions.cpu().numpy())

        avg_val_loss = total_val_loss / len(valid_loader)
        history["val_loss"].append(avg_val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        current_lr_epoch_end = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{n_epochs} Summary: "
              f"Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr_epoch_end:.6e} | Duration: {epoch_duration:.2f}s")

        # Epoch-wise schedulers (like ReduceLROnPlateau)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
            # ReduceLROnPlateau might change the LR, so log it if it has verbose=True or check manually
            # print(f"  ReduceLROnPlateau: Current LR {optimizer.param_groups[0]['lr']:.6e}")


        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(checkpoint_dir) / best_model_name
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Best model saved to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  Error saving model: {e}")
            
    print(f"\nFine-tuning complete. Best validation loss: {best_val_loss:.4f}")
    return history


if __name__ == '__main__':
    print("--- Starting Fine-tuning Loop Example ---")

    # --- Mock necessary components for demonstration ---
    # This DummyModel needs to be compatible with how the run_fine_tuning function calls it.
    # Specifically, model(src, values, padding_mask) should return a dict with "cell_emb",
    # and model.deconv_head should exist.
    class DummyDeconvolutionModel(nn.Module):
        def __init__(self, d_model=64, n_genes_vocab=100, n_cell_types=5, pad_value=-2):
            super().__init__()
            self.d_model = d_model
            self.pad_value = pad_value # Not directly used in this simplified dummy's forward
            
            # Mock for the scGPT's core TransformerModel part
            # It should accept src, values, src_key_padding_mask
            # For simplicity, this dummy will use 'values' to generate 'cell_emb'.
            # A real TransformerModel would use src (gene_ids) to lookup embeddings,
            # then combine with 'values' (binned expression).
            self.mock_transformer_encoder = nn.Linear(n_genes_vocab, d_model) # input_size = sequence_length
            
            # The deconvolution head
            self.deconv_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, n_cell_types), nn.Softmax(dim=-1)
            )
        
        def forward(self, src, values, src_key_padding_mask):
            # src: [batch_size, seq_len] (gene_ids)
            # values: [batch_size, seq_len] (binned_expression)
            # src_key_padding_mask: [batch_size, seq_len] (bool, True for padding)
            
            # In a real scGPT model with cell_emb_style='cls', the CLS token's output embedding
            # from the transformer layers would be 'cell_emb'.
            # This dummy simulates that by taking the 'values' tensor and processing it.
            # Let's assume 'values' represents the features for the CLS token if we average,
            # or we can just project the whole sequence.
            # For this dummy, let's average the 'values' across the sequence dimension
            # before passing to the linear layer, to make it somewhat independent of seq_len.
            # We must be careful: src_key_padding_mask should be used to ignore padded values.
            
            # Create a mask for non-padded values based on src_key_padding_mask
            # Mask is True for padding, so invert for valid values
            valid_values_mask = ~src_key_padding_mask.unsqueeze(-1) # [B, S, 1]
            
            # Apply mask to values (set padded values to 0 for sum/mean)
            masked_values = values.unsqueeze(-1) * valid_values_mask # [B, S, 1]
            
            # Sum valid values and count them
            sum_values = masked_values.sum(dim=1) # [B, 1]
            num_valid_values = valid_values_mask.sum(dim=1) # [B, 1]
            num_valid_values = torch.clamp(num_valid_values, min=1.0) # Avoid division by zero
            
            # Mean of valid values (a very crude 'embedding' of the sequence)
            mean_sequence_value_features = sum_values / num_valid_values # [B,1]
            
            # To make it [B, d_model], this simple dummy needs more work.
            # Let's just use the mock_transformer_encoder on the raw 'values' for simplicity,
            # assuming 'values' has the right dimension (n_genes_vocab).
            # This is a BIG simplification.
            if values.shape[1] != self.mock_transformer_encoder.in_features:
                 # This will error if seq_len of data != n_genes_vocab in model def.
                 # This highlights the need for careful dummy data/model alignment.
                 # For the example, DUMMY_N_GENES_VOCAB in DataLoader should match n_genes_vocab here.
                 raise ValueError(f"Dummy model input dim mismatch: values.shape[1]={values.shape[1]}, expected {self.mock_transformer_encoder.in_features}")

            mock_cell_emb = self.mock_transformer_encoder(values) # [batch_size, d_model]
            
            return {"cell_emb": mock_cell_emb}

    class DummyDataLoader:
        def __init__(self, num_batches, batch_size, n_genes_vocab, n_cell_types, device, is_train=True):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.n_genes_vocab = n_genes_vocab # This is the number of features for the dummy model
            self.n_cell_types = n_cell_types
            self.device = device
            self.is_train = is_train

        def __len__(self):
            return self.num_batches

        def __iter__(self):
            for _ in range(self.num_batches):
                # Dummy data
                # gene_ids are not strictly used by this specific dummy model's forward, but are part of the spec
                gene_ids = torch.randint(0, self.n_genes_vocab, (self.batch_size, self.n_genes_vocab), device=self.device, dtype=torch.long)
                # values for this dummy model should be [batch_size, n_genes_vocab]
                values_for_dummy = torch.rand(self.batch_size, self.n_genes_vocab, device=self.device, dtype=torch.float)
                padding_mask = torch.zeros(self.batch_size, self.n_genes_vocab, device=self.device, dtype=torch.bool)
                # Example: pad the last 10 features for some samples
                if self.n_genes_vocab > 10 and self.is_train: 
                    padding_mask[0, -10:] = True 
                
                fractions = torch.softmax(torch.rand(self.batch_size, self.n_cell_types, device=self.device), dim=-1)
                yield {"gene_ids": gene_ids, "values": values_for_dummy, "padding_mask": padding_mask, "fractions": fractions}
    
    # --- Parameters for the dummy run ---
    DUMMY_N_EPOCHS = 2 # Reduced for quick test
    DUMMY_LR = 1e-3 # Slightly higher for faster change
    DUMMY_BATCH_SIZE = 8 
    DUMMY_N_GENES_VOCAB = 100 # Number of genes in vocab, also used as seq_len for dummy model input
    DUMMY_N_CELL_TYPES = 5
    DUMMY_D_MODEL = 32 # d_model for dummy model
    
    dummy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate dummy components
    example_model = DummyDeconvolutionModel(
        d_model=DUMMY_D_MODEL, 
        n_genes_vocab=DUMMY_N_GENES_VOCAB, 
        n_cell_types=DUMMY_N_CELL_TYPES
    ).to(dummy_device)
        
    example_train_loader = DummyDataLoader(10, DUMMY_BATCH_SIZE, DUMMY_N_GENES_VOCAB, DUMMY_N_CELL_TYPES, dummy_device)
    example_valid_loader = DummyDataLoader(5, DUMMY_BATCH_SIZE, DUMMY_N_GENES_VOCAB, DUMMY_N_CELL_TYPES, dummy_device, is_train=False)
    
    # Ensure all parameters of the dummy model are trainable
    example_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, example_model.parameters()), lr=DUMMY_LR)
    
    example_loss_fn = nn.L1Loss() 
    example_loss_type = "L1" # Matches example_loss_fn
    
    # Example using OneCycleLR scheduler
    total_steps_for_scheduler = DUMMY_N_EPOCHS * len(example_train_loader)
    example_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        example_optimizer, 
        max_lr=DUMMY_LR, 
        total_steps=total_steps_for_scheduler
    )
    # example_scheduler = None # Option for no scheduler

    print(f"\nCreated dummy components for device: {dummy_device}")
    print(f"  Dummy model: {type(example_model)}")
    print(f"  Dummy train_loader: {len(example_train_loader)} batches of size {DUMMY_BATCH_SIZE}")
    print(f"  Dummy valid_loader: {len(example_valid_loader)} batches of size {DUMMY_BATCH_SIZE}")
    print(f"  Optimizer: {type(example_optimizer)}")
    print(f"  Scheduler: {type(example_scheduler)}")
    print(f"  Loss function: {type(example_loss_fn)} (type: {example_loss_type})")


    # --- Run the fine-tuning loop ---
    print("\nCalling run_fine_tuning with dummy components...")
    try:
        training_history = run_fine_tuning(
            model=example_model,
            train_loader=example_train_loader,
            valid_loader=example_valid_loader,
            optimizer=example_optimizer,
            loss_fn=example_loss_fn,
            n_epochs=DUMMY_N_EPOCHS,
            device=dummy_device,
            loss_type_str=example_loss_type,
            scheduler=example_scheduler,
            checkpoint_dir="./dummy_checkpoints_finetune", # Unique dir for this test
            best_model_name="dummy_best_model_finetuned.pt"
        )
        print("\nFine-tuning loop example finished successfully.")
        print(f"Training history: {training_history}")
    except Exception as e:
        print(f"Error during dummy fine-tuning loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy checkpoint directory
        # import shutil
        # dummy_ckpt_path = Path("./dummy_checkpoints_finetune")
        # if dummy_ckpt_path.exists():
        #     shutil.rmtree(dummy_ckpt_path)
        #     print(f"Cleaned up {dummy_ckpt_path} directory.")
            
    print("\nReminder: This example uses dummy components. Actual fine-tuning requires a real model, data, and configurations.")

```
