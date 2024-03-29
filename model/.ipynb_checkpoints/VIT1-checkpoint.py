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

        # self.bn = nn.BatchNorm2d(embed_dim)
        # #self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(0.1)



    def forward(self, im):
        B, C, H, W = im.shape
        #x = self.layer3_rgb(self.layer2_rgb(self.layer1_rgb(im))).flatten(2).transpose(1, 2)
        x = self.proj(im).flatten(2).transpose(1, 2)
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
                torch.randn(1, self.patch_embed.num_patches + 1 , d_model)
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
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls + 2, d_model))
        
        self.apply(init_weights)
        self.final_linear = nn.Linear(2 * d_model, d_model, bias=True)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
            x = torch.cat((x, self.cls_emb), dim=1)
            
        else:
            x = torch.cat((cls_tokens, x), dim=1)
            x = torch.cat((x, self.cls_emb), dim=1)     
        
        pos_embed = self.pos_embed
        pos_embed = torch.cat((pos_embed, self.cls_emb),dim=1)
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS[0], W // PS[1]),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        print(checkpoint_path)
        _load_weights(self, checkpoint_path, prefix)