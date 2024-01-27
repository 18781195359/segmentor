import torch
import torch.nn as nn
import torch.nn.functional as F

from model import padding, unpadding
from timm.models.layers import trunc_normal_


class Fusion_Module(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_layer = nn.Linear(2*d_model, d_model)
    def forward(self, x_rgb, x_tir):
        x_sum = x_rgb + x_tir
        x_mul = x_rgb * x_tir
        return self.fusion_layer(torch.cat((x_sum, x_mul), dim=2))

class Fusion_Module_all(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fusion_layer = nn.Linear(d_model, 2*d_model)
        self.fusion_layer1 = nn.Linear(2*d_model, 4*d_model)
        self.fusion_layer2 = nn.Linear(4*d_model, d_model)
    def forward(self, x_before):
        return self.fusion_layer2(self.fusion_layer1(self.fusion_layer(x_before))) + x_before

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        encoder1,
        decoder,
        decoder_binary,
        n_cls
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.encoder1 = encoder1
        self.decoder = decoder
        self.decoder_binary = decoder_binary
        self.fusion = Fusion_Module(encoder.d_model)
        self.fusion1 = Fusion_Module_all(encoder.d_model)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im, im_tir):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x_rgb = self.encoder(im, return_features=True)
        x_tir = self.encoder1(im_tir, return_features=True)
        x = self.fusion(x_rgb, x_tir)
        
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        class_for_decoder = x[:, -self.n_cls:]
        class_for_binary = x[:,-(self.n_cls + 2):-(self.n_cls)]
        x = x[:, :-self.n_cls-2]
        
        
        # x_binary = x_binary[:, num_extra_tokens:]
        masks_binary = self.decoder_binary(x, (H, W), class_for_binary)
        masks_binary = F.interpolate(masks_binary, size=(H, W), mode="bilinear")
        masks_binary= unpadding(masks_binary, (H_ori, W_ori))
        
        x = self.fusion1(x)
        masks = self.decoder(x, (H, W), class_for_decoder)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks, masks_binary

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
