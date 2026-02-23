import jittor as jt
from jittor import nn
import numpy as np
from block_jt import StepAdaBlock
from utils import trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        x = self.proj(x)                       
        x = x.flatten(2).transpose(0, 2, 1)    
        return x


class StandardAdaViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 ada_head=True, ada_layer=False, ada_token=False,
                 keep_layers=1, ada_token_start_layer=0, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.keep_layers = keep_layers

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim), dtype='float32')
        self.pos_embed = jt.zeros((1, num_patches + 1, embed_dim), dtype='float32')
        self.pos_drop  = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.Sequential(*[
            StepAdaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                ada_head=(ada_head and i >= keep_layers),
                ada_layer=(ada_layer and i >= keep_layers),
                is_token_select=(ada_token and i >= ada_token_start_layer),
            )
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) \
                    if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,   0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).float32()
        x = jt.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.float32()
        x = self.pos_drop(x)

        attn_list, hidden_list = [], []
        head_select_list, layer_select_list, token_select_list = [], [], []
        head_logits_list, layer_logits_list = [], []

        for blk in self.blocks.layers.values():
            x, attn, h_sel, l_sel, t_sel, h_logits, l_logits = blk(x)
            attn_list.append(attn)
            hidden_list.append(x)
            if h_sel    is not None: head_select_list.append(h_sel)
            if l_sel    is not None: layer_select_list.append(l_sel)
            if t_sel    is not None: token_select_list.append(t_sel)
            if h_logits is not None: head_logits_list.append(h_logits)
            if l_logits is not None: layer_logits_list.append(l_logits)

        def to_tensor(lst):
            return jt.stack(lst, dim=1) if lst else None

        head_select = to_tensor(head_select_list)
        if head_select is not None:
            if head_select.ndim == 4 and head_select.shape[-1] == 1:
                head_select = head_select.reshape(
                    head_select.shape[0],
                    head_select.shape[1],
                    head_select.shape[2])

        layer_select = to_tensor(layer_select_list)
        token_select = to_tensor(token_select_list)
        head_logits  = to_tensor(head_logits_list)
        layer_logits = to_tensor(layer_logits_list)

        x = self.norm(x)
        logit_dict = dict(
            head_select_logits  = head_logits,
            layer_select_logits = layer_logits,
        )
        return (x[:, 0], head_select, layer_select, token_select,
                attn_list, hidden_list, logit_dict)

    def execute(self, x, ret_attn_list=False):
        x, head_sel, layer_sel, token_sel, attn_list, hidden_list, logits = \
            self.forward_features(x)
        x = self.head(x)

        if ret_attn_list:
            return x, head_sel, layer_sel, token_sel, attn_list, hidden_list, logits
        return x, head_sel, layer_sel, token_sel, logits