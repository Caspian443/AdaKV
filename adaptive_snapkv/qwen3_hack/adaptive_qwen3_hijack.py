import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.utils import logging, is_flash_attn_2_available
from adaptive_snapkv.monkeypatch.snapkv_utils import init_adaptive_snapkv

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

def observer_Qwen3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    # 1. Init Hook Point (AttnScoreCapturer will be attached to self.kv_cluster)
    init_adaptive_snapkv(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    bsz, q_len, _ = hidden_states.size()

    # 2. QKV Proj + Norm (Qwen3 Specific)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 3. RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        # 4. [CRITICAL] Probe: Calculate and save scores
        # Only calculating heatmap during Prefill (q_len > 1)
        if q_len > 1:
             # The 'update_kv' in get_attention.py is defined as: calculate score, then return original KV.
             # So this will NOT compress, just observe.
             key_states, value_states = self.kv_cluster.update_kv(key_states, query_states, value_states)
        
        # 5. Update Cache (Standard logic, append without eviction)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 6. Flash Attention (Standard Logic)
    # Transformers shape: (Batch, Head, Seq, Dim)
    # FlashAttn func shape: (Batch, Seq, Head, Dim)
    
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = 0.0 if not self.training else self.attention_dropout
    
    attn_output = flash_attn_func(
        query_states, key_states, value_states, 
        dropout_p=dropout_rate, softmax_scale=self.scaling, causal=True
    )
    
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None
