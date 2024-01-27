import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.layers import trunc_normal_
from model.VIT1 import VisionTransformer
from timm.models.vision_transformer import default_cfgs
from model.decoder import DecoderLinear
from model.decoder import MaskTransformer
from model.Segmenter import Segmenter
from timm.models.helpers import load_pretrained, load_custom_pretrained
from model import checkpoint_filter_fn
def create_vit(model_cfg_al):
    model_cfg = {}

    model_cfg["image_size"] = (model_cfg_al["image_size_0"], model_cfg_al["image_size_1"])
    backbone = model_cfg_al["backbone"]
    model_cfg["patch_size"] = (model_cfg_al["patch_size"], model_cfg_al["patch_size"])
    model_cfg["n_layers"] = model_cfg_al["n_layers"]
    model_cfg["d_model"] = model_cfg_al["d_model"]
    model_cfg["d_model"] = model_cfg_al["d_model"]
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]
    #normalization = model_cfg.pop("normalization")

    model_cfg["n_heads"] = model_cfg_al["n_heads"]
    model_cfg["n_cls"] = model_cfg_al["n_classes"]
    model_cfg["dropout"] = 0.1
    model_cfg["drop_path_rate"] = model_cfg_al["drop_path_rate"]
    model_cfg["distilled"] = False
    model_cfg["channels"] = 3

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )
    print(type(default_cfg))
    default_cfg["input_size"] = (3, 480, 640)

    model = VisionTransformer(**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        print(1111)
        load_custom_pretrained(model, default_cfg)

    return model




def create_decoder(encoder, decoder_cfg_al):
    decoder_cfg = {}
    name = decoder_cfg_al["name"]
    decoder_cfg["n_cls"] = decoder_cfg_al["n_cls"]
    decoder_cfg["patch_size"] = encoder.patch_size
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["n_layers"] = decoder_cfg_al["n_layers"]

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder_cfg["drop_path_rate"] = decoder_cfg_al["drop_path_rate"]
        decoder_cfg["dropout"] = 0.1
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    print(model_cfg)
    model_cfg = model_cfg

    decoder_cfg = model_cfg
    decoder_cfg["n_cls"] = model_cfg["n_classes"]
    decoder_binary_cfg = model_cfg


    encoder = create_vit(model_cfg)
    encoder1 = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    decoder_binary_cfg["n_cls"] = 2
    decoder_binary = create_decoder(encoder, decoder_binary_cfg)
    model = Segmenter(encoder, encoder1, decoder, decoder_binary, n_cls=model_cfg["n_classes"])

    return model