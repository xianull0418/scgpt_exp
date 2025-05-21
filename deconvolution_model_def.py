import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# Assuming scgpt is in the Python path
# sys.path.append("path_to_scGPT_repository") # If scgpt is not installed
try:
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab # For loading vocab
except ImportError:
    # Fallback for environments where scgpt might not be fully installed/discoverable initially
    # This allows the script to be created, but it won't run without scGPT.
    print("Warning: scGPT library not found. Please ensure it's installed or in PYTHONPATH for full functionality.")
    # Define dummy classes if scGPT is not found, so the script is syntactically valid.
    class TransformerModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Dummy TransformerModel initialized because scGPT was not found.")
            # Add a basic layer to allow state_dict loading/saving for the dummy example
            self.dummy_layer = nn.Linear(10,10) 
        def load_state_dict(self, state_dict, strict=True):
            print("Dummy TransformerModel.load_state_dict called.")
            # Handle minimal state_dict for dummy example
            if 'dummy_layer.weight' in state_dict and 'dummy_layer.bias' in state_dict:
                 super().load_state_dict({'dummy_layer.weight': state_dict['dummy_layer.weight'], 
                                          'dummy_layer.bias': state_dict['dummy_layer.bias']}, strict=False)
            return [], [] # missing_keys, unexpected_keys
        def forward(self, *args, **kwargs):
            print("Dummy TransformerModel.forward called.")
            return {'cell_emb': torch.randn(1, kwargs.get("d_model", 128))} # d_model from dummy_args

    class GeneVocab:
        def __init__(self, genes_list, specials=None):
            self.stoi = {gene: i for i, gene in enumerate(genes_list)}
            if specials:
                for i, special_token in enumerate(specials):
                    if special_token not in self.stoi: # Add if not already present
                        self.stoi[special_token] = len(self.stoi)
            self.itos = {i: gene for gene, i in self.stoi.items()}
            self._default_index = -1 # Placeholder
            print("Dummy GeneVocab initialized.")

        @classmethod
        def from_file(cls, file_path: Path):
            print(f"Dummy GeneVocab.from_file called with {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict): # Assuming stoi format like {"<pad>":0, "gene1":1}
                return cls(list(data.keys())) # Simplified for dummy
            elif isinstance(data, list): # Assuming list of genes
                return cls(data)
            raise ValueError("Unsupported dummy vocab format")
        
        def get_default_index(self):
            return self._default_index

        def __len__(self):
            return len(self.itos)
        
        def __getitem__(self, token):
            return self.stoi.get(token, self._default_index)


# --- User-defined placeholder paths ---
# model_dir_placeholder = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/" # Should contain args.json, best_model.pt
# vocab_file_placeholder = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/vocab.json" # Or separate path if not in model_dir
# --- End of User-defined placeholder paths ---

def create_deconvolution_model(
    model_dir_path: str,
    vocab_file_path: str,
    n_cell_types: int,
    scgpt_dropout_rate: Optional[float] = None,
    head_dropout_rate: float = 0.1,
    freeze_scgpt_base: bool = False,
    d_model_override: Optional[int] = None,
    nhead_override: Optional[int] = None,
    nlayers_override: Optional[int] = None,
    pad_value: int = -2 # Default pad value used in some scGPT setups
) -> Optional[nn.Module]:
    '''
    Creates a deconvolution model by loading a pre-trained scGPT model
    and adding a task-specific deconvolution head.
    '''
    print(f"Creating deconvolution model...")
    print(f"Loading pre-trained model from directory: {model_dir_path}")
    print(f"Loading vocabulary from: {vocab_file_path}")

    model_config_file = Path(model_dir_path) / "args.json"
    model_weights_file = Path(model_dir_path) / "best_model.pt"
    vocab_path = Path(vocab_file_path)

    if not model_config_file.exists():
        print(f"Error: Model config file 'args.json' not found at {model_config_file}")
        return None
    if not model_weights_file.exists():
        print(f"Error: Model weights file 'best_model.pt' not found at {model_weights_file}")
        return None
    if not vocab_path.exists():
        print(f"Error: Vocabulary file 'vocab.json' not found at {vocab_path}")
        return None

    try:
        with open(model_config_file, "r") as f:
            model_configs: Dict[str, Any] = json.load(f)
        print(f"Loaded model configurations from {model_config_file}")
    except Exception as e:
        print(f"Error loading model config {model_config_file}: {e}")
        return None
        
    try:
        vocab = GeneVocab.from_file(vocab_path)
        ntoken = len(vocab)
        # Ensure pad_token is correctly identified from vocab or args.json
        pad_token = model_configs.get("pad_token", "<pad>") 
        if pad_token not in vocab.stoi: # Check if pad_token exists in vocab's string-to-index mapping
             print(f"Warning: Pad token '{pad_token}' defined in args.json (or default) not found in vocabulary. Attempting to use vocab's default or first special.")
             # Attempt to find a pad token if not explicitly defined or if default is missing
             if "<pad>" in vocab.stoi: pad_token = "<pad>"
             elif vocab.stoi: pad_token = vocab.itos[0] # Fallback, not ideal
             else:
                 print("Error: Cannot determine pad token from vocabulary.")
                 return None
        print(f"Loaded vocabulary from {vocab_path}. Vocab size (ntoken): {ntoken}, Pad token: {pad_token}")
    except Exception as e:
        print(f"Error loading vocab {vocab_path}: {e}")
        return None

    # Extract parameters from model_configs, with overrides and defaults
    d_model = d_model_override if d_model_override is not None else model_configs.get("embsize", 512)
    nhead = nhead_override if nhead_override is not None else model_configs.get("nheads", 4)
    nlayers = nlayers_override if nlayers_override is not None else model_configs.get("nlayers", 4)
    
    d_hid = model_configs.get("d_hid", d_model * 4) # Transformer default: 4*d_model
    
    current_dropout = model_configs.get("dropout", 0.1) # Default dropout from scGPT
    if scgpt_dropout_rate is not None:
        current_dropout = scgpt_dropout_rate # Override if specified
        
    # Other parameters from args.json that TransformerModel might need
    nlayers_cls = model_configs.get("nlayers_cls", 3)
    n_cls = model_configs.get("n_cls", 1) # Often 1 for tasks using a CLS token
    input_emb_style = model_configs.get("input_emb_style", "continuous") # "continuous", "binned"
    n_input_bins = model_configs.get("n_input_bins", 0) # Relevant if input_emb_style is "binned"
    cell_emb_style = model_configs.get("cell_emb_style", "cls") # How cell embeddings are generated
    use_fast_transformer = model_configs.get("use_fast_transformer", True) # Check args.json
    pre_norm = model_configs.get("pre_norm", False) # Check args.json
    
    # pad_value can also be in args.json, overriding the function default
    final_pad_value = model_configs.get("pad_value", pad_value)

    print(f"Final Model params: ntoken={ntoken}, d_model={d_model}, nhead={nhead}, d_hid={d_hid}, nlayers={nlayers}, dropout={current_dropout}")
    print(f"Pad token: {pad_token}, Pad value: {final_pad_value}, n_input_bins: {n_input_bins}")

    # Instantiate scGPT model
    try:
        scgpt_model = TransformerModel(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=nlayers_cls,
            n_cls=n_cls,
            vocab=vocab, # Pass the loaded vocab object
            dropout=current_dropout,
            pad_token=pad_token,
            pad_value=final_pad_value, 
            input_emb_style=input_emb_style,
            n_input_bins=n_input_bins,
            cell_emb_style=cell_emb_style,
            use_fast_transformer=use_fast_transformer,
            pre_norm=pre_norm
            # Add other relevant params from model_configs if TransformerModel expects them
        )
        print("scGPT TransformerModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating TransformerModel: {e}")
        return None

    # Load pre-trained weights
    try:
        state_dict = torch.load(model_weights_file, map_location=torch.device('cpu'))
        # Use strict=False to allow for the new deconvolution head not being in the checkpoint,
        # and to ignore other potential mismatches (e.g. if model was saved with a different head).
        missing_keys, unexpected_keys = scgpt_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained weights from {model_weights_file}.")
        if missing_keys:
            print(f"  Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            # Filter out keys that might belong to a pre-existing head in the checkpoint
            filtered_unexpected_keys = [k for k in unexpected_keys if not k.startswith("deconv_head") and not k.startswith("value_head")]
            if filtered_unexpected_keys:
                 print(f"  Warning: Unexpected keys found in state_dict (excluding typical head names): {filtered_unexpected_keys}")

    except Exception as e:
        print(f"Error loading model weights from {model_weights_file}: {e}")
        return None

    # Define the deconvolution head
    # Input to the head is typically the CLS token embedding from scGPT (d_model)
    # Output is n_cell_types probabilities
    scgpt_model.deconv_head = nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.ReLU(),
        nn.Dropout(head_dropout_rate),
        nn.Linear(d_model // 2, n_cell_types),
        nn.Softmax(dim=-1)  # Softmax for probability distribution over cell types
    )
    print(f"Deconvolution head added: Linear({d_model}, {d_model//2}) -> ReLU -> Dropout({head_dropout_rate}) -> Linear({d_model//2}, {n_cell_types}) -> Softmax")

    # Freeze base model parameters if requested
    if freeze_scgpt_base:
        print("Freezing pre-trained scGPT base model parameters...")
        frozen_count = 0
        trainable_count = 0
        for name, param in scgpt_model.named_parameters():
            if 'deconv_head' not in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True # Ensure head parameters are trainable
                trainable_count +=1
        print(f"Parameters frozen: {frozen_count} parameters. Trainable (head): {trainable_count} parameters.")
    else:
        print("All model parameters (scGPT base + deconv_head) will be trainable.")
        for param in scgpt_model.parameters(): # Ensure all are trainable
            param.requires_grad = True
            
    return scgpt_model

if __name__ == '__main__':
    print("--- Deconvolution Model Creation Example ---")

    # Define placeholder paths (these should be replaced by actual paths)
    # For the example, we check if these are the default placeholders or if they've been changed.
    user_model_dir = "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/"
    user_vocab_file = "YOUR_PATH_TO_VOCAB_JSON_FILE" # Can be same as model_dir + "/vocab.json"

    # Fallback to dummy files if placeholders are not changed
    if user_model_dir == "YOUR_PATH_TO_PRETRAINED_MODEL_DIR/" or \
       user_vocab_file == "YOUR_PATH_TO_VOCAB_JSON_FILE":
        
        print("\nPlaceholders for model/vocab paths not set. Using dummy files for demonstration.")
        dummy_model_dir = Path("./dummy_scgpt_model_dir")
        dummy_model_dir.mkdir(exist_ok=True)
        
        dummy_vocab_file = dummy_model_dir / "vocab.json"
        dummy_args_file = dummy_model_dir / "args.json"
        dummy_weights_file = dummy_model_dir / "best_model.pt"

        # Create dummy args.json
        dummy_args_content = {
            "embsize": 64, "nheads": 2, "nlayers": 1, "dropout": 0.1, "d_hid": 128,
            "n_input_bins": 0, "input_emb_style": "continuous", "cell_emb_style": "cls",
            "pad_token": "<pad>", "pad_value": -2, "n_cls": 1, "nlayers_cls":1,
            "use_fast_transformer": False, "pre_norm": False
        }
        with open(dummy_args_file, "w") as f:
            json.dump(dummy_args_content, f)

        # Create dummy vocab.json
        dummy_vocab_content = {"<pad>": 0, "geneA": 1, "geneB": 2, "<cls>": 3, "<eoc>": 4}
        with open(dummy_vocab_file, "w") as f:
            json.dump(dummy_vocab_content, f)

        # Create a dummy best_model.pt (minimal state_dict)
        try:
            # Use the (potentially dummy) GeneVocab and TransformerModel to create a savable state_dict
            temp_vocab = GeneVocab.from_file(dummy_vocab_file)
            minimal_scgpt_model = TransformerModel(
                ntoken=len(temp_vocab), d_model=dummy_args_content["embsize"], 
                nhead=dummy_args_content["nheads"], d_hid=dummy_args_content["d_hid"],
                nlayers=dummy_args_content["nlayers"], vocab=temp_vocab,
                pad_token=dummy_args_content["pad_token"], pad_value=dummy_args_content["pad_value"]
            )
            torch.save(minimal_scgpt_model.state_dict(), dummy_weights_file)
            print(f"Dummy model files (args, vocab, weights) created in {dummy_model_dir}")
            
            # Update paths to use these dummy files for the example run
            model_dir_to_use = str(dummy_model_dir)
            vocab_file_to_use = str(dummy_vocab_file)
            can_run_example = True
        except Exception as e:
            print(f"Critical error: Could not create dummy model files for example: {e}")
            print("This might happen if scGPT or its dependencies are not correctly installed/available.")
            print("Skipping model creation demonstration.")
            can_run_example = False
    else:
        model_dir_to_use = user_model_dir
        vocab_file_to_use = user_vocab_file
        can_run_example = True
        print(f"\nUsing user-provided paths: model_dir='{model_dir_to_use}', vocab_file='{vocab_file_to_use}'")


    if can_run_example:
        num_cell_types_example = 12  # Example: 12 cell types for deconvolution
        
        print(f"\nAttempting to create deconvolution model with n_cell_types={num_cell_types_example}...")
        deconv_model_instance = create_deconvolution_model(
            model_dir_path=model_dir_to_use,
            vocab_file_path=vocab_file_to_use,
            n_cell_types=num_cell_types_example,
            freeze_scgpt_base=True, # Example: freeze the base
            head_dropout_rate=0.2,
            # Example of overriding a model parameter:
            # d_model_override=dummy_args_content["embsize"] if 'dummy_args_content' in locals() else 128 
        )

        if deconv_model_instance:
            print("\n--- Deconvolution Model Instance ---")
            print(deconv_model_instance)
            
            print("\n--- Trainable Parameters Check ---")
            total_params = 0
            trainable_params = 0
            for name, param in deconv_model_instance.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    print(f"  Trainable: {name} (Size: {param.numel()})")
                    trainable_params += param.numel()
            print(f"Total model parameters: {total_params}")
            print(f"Trainable model parameters: {trainable_params}")
            if freeze_scgpt_base and trainable_params > 0 and total_params > trainable_params:
                 print("Base model appears frozen as expected.")
            elif not freeze_scgpt_base and total_params == trainable_params:
                 print("Full model is trainable as expected.")

        else:
            print("\nDeconvolution model creation failed. Review logs for details.")
            print("If using real paths, ensure they are correct and files are valid.")
            print("If using dummy files, this might indicate an issue with the dummy setup or scGPT availability.")

    print("\n--- End of Deconvolution Model Creation Example ---")
```
