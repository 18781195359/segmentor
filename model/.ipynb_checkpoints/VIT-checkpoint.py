import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from model import init_weights
from model import resize_pos_embed
from model.blocks import Block
from timm.models.vision_transformer import _load_weights
from model import Backbone_ResNet152_in3
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size[0]//self.patch_size[0],image_size[1]//self.patch_size[1]
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=(self.patch_size[0],self.patch_size[1]), stride=(self.patch_size[0],self.patch_size[1])
        )

        self.bn = nn.BatchNorm2d(embed_dim)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)



    def forward(self, im):
        B, C, H, W = im.shape
        #x = self.layer3_rgb(self.layer2_rgb(self.layer1_rgb(im))).flatten(2).transpose(1, 2)
        x = self.bn(self.relu(self.proj(im))).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                image_size,
                patch_size,
                n_layers,
                d_model,#1200
                d_ff,
                n_heads,
                n_cls,
                dropout=0.1,
                drop_path_rate=0.0,
                distilled=False,
                channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )

        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)
        self.final_linear = nn.Linear(2 * d_model, d_model, bias=True)

    def forward(self, im_rgb, im_tir, return_features=False):
        B, _, H, W = im_rgb.shape
        PS = self.patch_size

        x_rgb = self.patch_embed(im_rgb)
        x_tir = self.patch_embed(im_tir)
        x_final = self.final_linear(torch.cat((x_rgb, x_tir), dim=2))

        #x_final = x_rgb+ x_tir
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x_rgb = torch.cat((cls_tokens, dist_tokens, x_rgb), dim=1)
            x_tir = torch.cat((cls_tokens, dist_tokens, x_tir), dim=1)
            x_final = torch.cat((cls_tokens, dist_tokens, x_final), dim=1)
        else:
            x_final = torch.cat((cls_tokens, x_final), dim=1)
            x_rgb = torch.cat((cls_tokens, x_rgb), dim=1)
            x_tir = torch.cat((cls_tokens, x_tir), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled

        if x_final.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS[0], W // PS[1]),
                num_extra_tokens,
            )

        x_final = x_final + pos_embed
        x_rgb = x_rgb + pos_embed
        x_tir = x_tir + pos_embed

        x_tir = self.dropout(x_tir)
        x_rgb = self.dropout(x_rgb)
        x_final = self.dropout(x_final)

        index_epcho = 0
        x_final_for_binary = None
        x_tir_for_binary = None
        x_rgb_for_binary = None
        for blk in self.blocks:
            index_epcho += 1
            temp_rgb = x_rgb
            temp_tir = x_tir
            x_tir = blk(temp_tir + temp_rgb)
            x_rgb = blk(torch.sigmoid(temp_tir * temp_rgb))
            x_final = blk(x_final)
            if index_epcho == 6:
                x_rgb_for_binary = x_rgb
                x_tir_for_binary = x_tir
                x_final_for_binary = x_final
        # x_final = self.norm(self.final_linear(torch.cat((x_rgb, x_tir), dim=2)) + x_final)
        x_final = self.norm(self.final_linear(torch.cat((x_rgb, x_tir), dim=2)) + x_final)
        x_final_for_binary = self.norm(self.final_linear(torch.cat((x_rgb_for_binary, x_tir_for_binary), dim=2)) + x_final_for_binary)

        if return_features:
            return x_final,x_final_for_binary

        if self.distilled:
            x_final, x_dist = x_final[:, 0], x_final[:, 1]
            x_final = self.head(x_final)
            x_dist = self.head_dist(x_dist)
            x_final = (x_final + x_dist) / 2
        else:
            x_final = x_final[:, 0]
            x_final = self.head(x_final)

            x_final_for_binary = x_final_for_binary[:, 0]
            x_final_for_binary = self.head(x_final_for_binary)

        return x_final, x_final_for_binary

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        print(checkpoint_path)
        _load_weights(self, checkpoint_path, prefix)




