import transformers
from .adaptive_qwen3_hijack import observer_Qwen3Attention_forward

def replace_qwen3_adaptive():
    """
    Monkey patch Qwen3 Attention ONLY to capture attention scores.
    NO compression, NO cache modification.
    """
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
    except ImportError:
        print("Warning: Qwen3 not found in transformers. Skipping monkey patch.")
        return

    print("Applying Qwen3 Observer Patch (Attention Only)...")
    
    # Replace Attention Forward only
    qwen3_modeling.Qwen3Attention.forward = observer_Qwen3Attention_forward
    
    # Model.forward and prepare_inputs_for_generation remain untouched
    # because we are not changing the cache structure or length.
