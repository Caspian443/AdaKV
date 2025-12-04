import transformers
from .adaptive_deepseek_v3_hijack import observer_DeepseekV3Attention_forward

def replace_deepseek_v3_adaptive():
    """
    Monkey patch DeepSeek-V3 Attention ONLY to capture attention scores.
    NO compression, NO cache modification.
    """
    try:
        import transformers.models.deepseek_v3.modeling_deepseek_v3 as deepseek_v3_modeling
    except ImportError:
        print("Warning: DeepSeek-V3 not found in transformers. Skipping monkey patch.")
        return

    print("Applying DeepSeek-V3 Observer Patch (Attention Only)...")
    
    # Replace Attention Forward only
    deepseek_v3_modeling.DeepseekV3Attention.forward = observer_DeepseekV3Attention_forward
    
    # Model.forward and prepare_inputs_for_generation remain untouched
    # because we are not changing the cache structure or length.

