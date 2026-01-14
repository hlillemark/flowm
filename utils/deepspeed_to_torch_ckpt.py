# Deepspeed saves the state dict differently: need to use the mp_rank_00_model_states.pt to turn it back 

from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# requires weights_only=False change in this function 
convert_zero_checkpoint_to_fp32_state_dict(
    "source",
    "target"
)

