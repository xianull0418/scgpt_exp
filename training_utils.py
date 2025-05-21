import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union

def get_loss_function(loss_type: str = "L1") -> nn.Module:
    """
    Returns a PyTorch loss function based on the specified type.
    Supported types: 'L1', 'MSE', 'KLDivergence'.

    Args:
        loss_type (str): Type of loss function.

    Returns:
        nn.Module: PyTorch loss function instance.
    """
    if loss_type == "L1":
        print("Using L1 Loss (Mean Absolute Error).")
        return nn.L1Loss()
    elif loss_type == "MSE":
        print("Using MSE Loss (Mean Squared Error).")
        return nn.MSELoss()
    elif loss_type == "KLDivergence":
        print("Using KL Divergence Loss.")
        print("  Note: For nn.KLDivLoss, model output should be log-probabilities (e.g., LogSoftmax)")
        print("  and target should be probabilities (e.g., Softmax output from target fractions).")
        print("  The deconvolution model head currently ends with Softmax.")
        print("  The training loop must apply log() to model output before passing to this loss.")
        return nn.KLDivLoss(reduction='batchmean') # 'batchmean' averages over batch and elements
                                                 # 'sum' might be preferred if per-sample loss magnitude is important
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Choose from 'L1', 'MSE', 'KLDivergence'.")

def get_optimizer(
    model: nn.Module, 
    learning_rate: float = 1e-4, 
    optimizer_type: str = "AdamW", 
    weight_decay: float = 1e-5
) -> optim.Optimizer:
    """
    Returns a PyTorch optimizer.
    Supported types: 'AdamW', 'Adam'.

    Args:
        model (nn.Module): The model whose parameters will be optimized.
        learning_rate (float): The learning rate.
        optimizer_type (str): Type of optimizer.
        weight_decay (float): Weight decay (L2 penalty).

    Returns:
        optim.Optimizer: PyTorch optimizer instance.
    """
    if optimizer_type == "AdamW":
        print(f"Using AdamW optimizer with LR={learning_rate}, Weight Decay={weight_decay}.")
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        print(f"Using Adam optimizer with LR={learning_rate}, Weight Decay={weight_decay}.")
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'AdamW' or 'Adam'.")

def get_scheduler(
    optimizer: optim.Optimizer, 
    scheduler_type: Optional[str] = "OneCycleLR", 
    total_steps: Optional[int] = None, # Required for OneCycleLR & LinearWarmup types
    warmup_steps: int = 0, # Used for OneCycleLR (via pct_start) and LinearWarmup
    # num_epochs: Optional[int] = None, # Alternative way to specify total_steps if steps_per_epoch is known
    # initial_lr: Optional[float] = None # Not directly used here as OneCycleLR takes max_lr from optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Returns a PyTorch learning rate scheduler.
    Supported types: 'OneCycleLR', 'LinearWarmup', 'ReduceLROnPlateau', None.

    Args:
        optimizer: The optimizer instance.
        scheduler_type: Type of scheduler.
        total_steps: Total number of training steps. Required for OneCycleLR and LinearWarmup.
        warmup_steps: Number of steps for warmup. Used by OneCycleLR (calculated as pct_start) and LinearWarmup.
        
    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Scheduler instance or None.
    """
    if scheduler_type is None or scheduler_type.lower() == "none":
        print("No learning rate scheduler will be used.")
        return None
        
    if scheduler_type == "OneCycleLR":
        if total_steps is None:
            raise ValueError("For OneCycleLR, 'total_steps' must be provided.")
        
        # pct_start is the fraction of total_steps for warmup.
        # If warmup_steps is given, calculate pct_start. Otherwise, use OneCycleLR default (0.3).
        pct_start = float(warmup_steps) / float(total_steps) if warmup_steps > 0 and total_steps > 0 else 0.3
        
        # Ensure pct_start is within a reasonable range for OneCycleLR (e.g., >0 and <1 if warmup is used)
        if not (0 < pct_start < 1.0) and warmup_steps > 0 :
             print(f"Warning: pct_start ({pct_start:.3f}) for OneCycleLR is unusual. Ensure warmup_steps and total_steps are sensible.")
        if pct_start == 0 and warmup_steps > 0: # Avoid division by zero if warmup_steps is very small relative to total_steps
            pct_start = 0.001 # A very small warmup
        
        print(f"Using OneCycleLR scheduler with Max LR (from optimizer)={optimizer.defaults['lr']}, Total Steps={total_steps}, Warmup Pct={pct_start:.3f}.")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults['lr'], # OneCycleLR uses this as the peak LR.
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos', # Common choice, can be 'linear'
            div_factor=25,         # Determines initial LR = max_lr / div_factor
            final_div_factor=1e4   # Determines min LR = initial_lr / final_div_factor
        )
    elif scheduler_type == "LinearWarmup":
        # This provides a linear warmup from a small LR to optimizer's initial LR.
        # After warmup, it can decay or stay constant. Here: constant.
        if warmup_steps <= 0:
            print("LinearWarmup selected, but warmup_steps is 0. Scheduler will effectively be constant LR.")
            # Return a scheduler that does nothing or a constant one
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: 1.0)


        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps) # Step from near 0 to 1
            return 1.0 # Constant LR (optimizer's initial_lr) after warmup

        print(f"Using LambdaLR for linear warmup over {warmup_steps} steps, then constant LR.")
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "ReduceLROnPlateau":
        print("Using ReduceLROnPlateau scheduler. Monitors a validation metric (e.g., loss).")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',    # Reduce LR when the monitored quantity has stopped decreasing
            factor=0.1,    # Factor by which the learning rate will be reduced
            patience=10,   # Number of epochs/steps with no improvement after which LR will be reduced
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Choose 'OneCycleLR', 'LinearWarmup', 'ReduceLROnPlateau', or None.")

if __name__ == '__main__':
    print("--- Loss, Optimizer, and Scheduler Setup Example ---")

    # Dummy model for instantiation (e.g., a simple linear layer)
    # In a real scenario, this would be your scGPT-based deconvolution model
    dummy_model = nn.Linear(in_features=100, out_features=10) # Example: 100 input features, 10 output classes/fractions
    print(f"\nDummy model created: {type(dummy_model)}")

    # --- 1. Loss Function Example ---
    print("\n--- Loss Function Examples ---")
    l1_loss_fn = get_loss_function(loss_type="L1")
    print(f"  Instantiated L1 Loss: {type(l1_loss_fn)}")

    mse_loss_fn = get_loss_function(loss_type="MSE")
    print(f"  Instantiated MSE Loss: {type(mse_loss_fn)}")
    
    kl_loss_fn = get_loss_function(loss_type="KLDivergence")
    print(f"  Instantiated KLDiv Loss: {type(kl_loss_fn)}")
    
    # Example of how KLDiv might be used in a training step:
    # model_output_raw = dummy_model(torch.randn(3, 100)) # Raw logits
    # model_output_softmax = torch.softmax(model_output_raw, dim=-1)
    # model_output_logsoftmax = torch.log(model_output_softmax + 1e-10) # Add epsilon for stability
    # # Target fractions should sum to 1 (probabilities)
    # target_fractions = torch.softmax(torch.rand(3, 10), dim=-1) 
    # loss_kl_example = kl_loss_fn(model_output_logsoftmax, target_fractions)
    # print(f"  Example KLDiv loss calculation (conceptual): {loss_kl_example.item() if 'loss_kl_example' in locals() else 'skipped'}")


    # --- 2. Optimizer Example ---
    print("\n--- Optimizer Examples ---")
    example_learning_rate = 5e-5
    adamw_optimizer = get_optimizer(dummy_model, learning_rate=example_learning_rate, optimizer_type="AdamW", weight_decay=0.01)
    print(f"  Instantiated AdamW Optimizer: {type(adamw_optimizer)}")
    
    adam_optimizer = get_optimizer(dummy_model, learning_rate=1e-3, optimizer_type="Adam", weight_decay=0.0)
    print(f"  Instantiated Adam Optimizer: {type(adam_optimizer)}")


    # --- 3. Scheduler Example ---
    print("\n--- Scheduler Examples ---")
    # These parameters would typically come from your training configuration
    total_training_steps_example = 2000 # Example: 20 epochs * 100 batches/epoch
    warmup_training_steps_example = 200  # Example: 10% of total steps for warmup
    
    # Using the AdamW optimizer for scheduler examples
    one_cycle_scheduler = get_scheduler(
        adamw_optimizer, 
        scheduler_type="OneCycleLR", 
        total_steps=total_training_steps_example,
        warmup_steps=warmup_training_steps_example 
    )
    print(f"  Instantiated OneCycleLR Scheduler: {type(one_cycle_scheduler)}")

    linear_warmup_scheduler = get_scheduler(
        adam_optimizer, # Using the other optimizer for variety
        scheduler_type="LinearWarmup",
        total_steps=total_training_steps_example, # total_steps is not strictly used by this simple LinearWarmup after warmup phase
        warmup_steps=warmup_training_steps_example
    )
    print(f"  Instantiated LinearWarmup (LambdaLR) Scheduler: {type(linear_warmup_scheduler)}")
    
    plateau_scheduler = get_scheduler(adamw_optimizer, scheduler_type="ReduceLROnPlateau")
    print(f"  Instantiated ReduceLROnPlateau Scheduler: {type(plateau_scheduler)}")
    
    no_scheduler = get_scheduler(adamw_optimizer, scheduler_type=None)
    print(f"  No scheduler instance: {type(no_scheduler)}")

    print("\n--- Example components instantiated successfully. ---")
    print("These utility functions can now be integrated into the main fine-tuning script.")
```
