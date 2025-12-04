import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
        eager_attention_forward,
        DeepseekV3Attention
    )
except ImportError:
    pass # Will handle check in patch.py

from transformers.utils import logging
from adaptive_snapkv.monkeypatch.snapkv_utils import init_adaptive_snapkv

logger = logging.get_logger(__name__)

def observer_DeepseekV3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # NOTE: adakv init
    init_adaptive_snapkv(self)

    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:  # support using interleaved weights for efficiency
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    # NOTE: adakv probe
    # Only calculating heatmap during Prefill (seq_length > 1)
    # We use the 'update_kv' from AttnScoreCapturer which simply calculates score and returns.
    # It will NOT modify key_states/value_states in a way that affects subsequent logic 
    # (because it returns them as is, and we don't use the return value to overwrite unless we want to).
    if seq_length > 1:
        # DeepSeek V3 calculates attention as Score = Q * K^T. 
        # query_states here is (B, H, S, D_qk)
        # key_states here is (B, H, S, D_qk)
        # This matches the expectation of manual_calcul_attn_score which expects (B, H, S, D)
        # We pass them to update_kv to trigger score calculation
        
        # Note: manual_calcul_attn_score handles GQA logic. 
        # DeepSeek V3 typically has num_heads == num_kv_heads (MLA behaves like MHA in projection), 
        # or if GQA is used on top of MLA (rare). 
        # DeepSeekV3Attention init says: self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # So it supports GQA.
        
        self.kv_cluster.update_kv(key_states, query_states, value_states)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

