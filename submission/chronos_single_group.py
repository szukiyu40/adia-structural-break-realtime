"""Retained singleton group-attention specialization from the previous official GPU runtime."""
from chronos.chronos2.layers import AttentionOutput, MHA

class SingleGroupMHA(MHA):

    def forward(self, hidden_states, mask, encoder_states=None, position_ids=None, output_attentions=False):
        assert not self.training and (not self.use_rope)
        assert hidden_states.shape[1] == 1 and encoder_states is None
        assert not output_attentions
        return AttentionOutput(hidden_states=self.o(self.v(hidden_states)), attn_weights=None)

def _simplify_single_group_attention(model):
    assert len(model.encoder_model.encoder.block) == 12
    for block in model.encoder_model.encoder.block:
        attention = block.layer[1].self_attention
        assert not attention.use_rope and (not attention.training)
        assert type(attention) is MHA
        attention.__class__ = SingleGroupMHA
