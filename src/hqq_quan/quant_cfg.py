from hqq.core.quantize import BaseQuantizeConfig


def get_quant_config_deit(model):
    quant_config = {}
    n_blocks = len(model.blocks)

    qkv_cfg  = BaseQuantizeConfig(nbits=4, group_size=48)
    proj_cfg = BaseQuantizeConfig(nbits=5, group_size=64)
    fc1_cfg  = BaseQuantizeConfig(nbits=4, group_size=48)
    fc2_cfg  = BaseQuantizeConfig(nbits=4, group_size=48)

    for i in range(n_blocks):
        quant_config[f"blocks.{i}.attn.qkv"]  = qkv_cfg
        quant_config[f"blocks.{i}.attn.proj"] = proj_cfg
        quant_config[f"blocks.{i}.mlp.fc1"]   = fc1_cfg
        quant_config[f"blocks.{i}.mlp.fc2"]   = fc2_cfg

    return quant_config



def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers

    qkv_cfg = BaseQuantizeConfig(nbits=4, group_size=32, quant_zero=True, quant_scale=True)
    o_cfg   = BaseQuantizeConfig(nbits=8, group_size=64, quant_zero=True, quant_scale=True)
    up_cfg  = BaseQuantizeConfig(nbits=4, group_size=32, quant_zero=True, quant_scale=True)
    gate_cfg= BaseQuantizeConfig(nbits=4, group_size=32, quant_zero=True, quant_scale=True)
    down_cfg= BaseQuantizeConfig(nbits=8, group_size=64, quant_zero=True, quant_scale=True)

    for i in range(n_layers):
        quant_config[f"model.layers.{i}.self_attn.q_proj"] = qkv_cfg
        quant_config[f"model.layers.{i}.self_attn.k_proj"] = qkv_cfg
        quant_config[f"model.layers.{i}.self_attn.v_proj"] = qkv_cfg
        quant_config[f"model.layers.{i}.self_attn.o_proj"] = o_cfg

        quant_config[f"model.layers.{i}.mlp.gate_proj"] = gate_cfg
        quant_config[f"model.layers.{i}.mlp.up_proj"]   = up_cfg
        quant_config[f"model.layers.{i}.mlp.down_proj"] = down_cfg

    return quant_config

