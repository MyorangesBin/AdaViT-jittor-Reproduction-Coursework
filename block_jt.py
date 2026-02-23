import jittor as jt
from jittor import nn
from utils import DropPath

def gumble_sigmoid(logits, tau=1, hard=False, threshold=0.5, training=True):
    if training:
        U = jt.rand_like(logits)
        gumbles = -jt.log(-jt.log(U + 1e-20) + 1e-20)
        y_soft = ((logits + gumbles) / tau).sigmoid()
    else:
        y_soft = logits.sigmoid()
    if hard:
        y_hard = jt.where(y_soft > threshold,
                          jt.ones_like(logits), jt.zeros_like(logits))
        ret = y_hard - y_soft.stop_grad() + y_soft
    else:
        ret = y_soft
    return ret


class SimpleTokenSelect(nn.Module):
    def __init__(self, dim_in, tau=3.0, is_hard=True, threshold=0.5):
        # ★ tau 从 5.0 降到 3.0，sigmoid 更陡，选择更果断
        super().__init__()
        self.mlp_head  = nn.Linear(dim_in, 1)
        self.tau       = tau
        self.is_hard   = is_hard
        self.threshold = threshold

    def execute(self, x, attn, attn_pre_softmax, token_select=None):
        logits = self.mlp_head(x[:, 1:])
        if token_select is None:
            token_select = gumble_sigmoid(
                logits, self.tau, self.is_hard, self.threshold, self.is_training())
            b = x.shape[0]
            cls_token_mask = jt.ones((b, 1, 1))
            token_select   = jt.concat([cls_token_mask, token_select], dim=1)
            token_select   = token_select.transpose(0, 2, 1)   # B,1,N
        attn_policy = jt.matmul(token_select.transpose(0, 2, 1), token_select)
        attn        = attn * attn_policy.unsqueeze(1)
        attn_sum    = attn.sum(dim=-1, keepdims=True)
        attn        = attn / (attn_sum + 1e-6)
        return attn, token_select.squeeze(1)


class BlockHeadSelect(nn.Module):
    def __init__(self, dim_in, num_heads, tau=3.0, is_hard=True, threshold=0.5):
        # ★ tau 从 5.0 降到 3.0
        super().__init__()
        self.mlp_head  = nn.Linear(dim_in, num_heads)
        self.tau       = tau
        self.is_hard   = is_hard
        self.threshold = threshold
        self.head_dim  = dim_in // num_heads

    def execute(self, x):
        logits = self.mlp_head(x)
        sample = gumble_sigmoid(
            logits, self.tau, self.is_hard, self.threshold, self.is_training())
        bsize  = x.shape[0]
        width_select = sample.unsqueeze(-1)\
                             .expand(bsize, -1, self.head_dim)\
                             .reshape(bsize, -1, 1)
        return sample, width_select, logits


class BlockLayerSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer=2, tau=3.0, is_hard=True, threshold=0.5):
        super().__init__()
        self.mlp_head  = nn.Linear(dim_in, num_sub_layer)
        self.tau       = tau
        self.is_hard   = is_hard
        self.threshold = threshold

    def execute(self, x):
        logits = self.mlp_head(x)
        sample = gumble_sigmoid(
            logits, self.tau, self.is_hard, self.threshold, self.is_training())
        return sample, logits


class DynaLinear(nn.Linear):
    def execute(self, input, width_select=None):
        if width_select is None:
            return super().execute(input)
        if width_select.ndim == 3 and width_select.shape[-1] == 1 \
                and width_select.shape[1] > 1:
            result = super().execute(input)
            mask   = width_select.transpose(0, 2, 1)
            return result * mask
        if width_select.ndim == 3 and width_select.shape[1] == 1 \
                and width_select.shape[-1] > 1:
            return super().execute(input * width_select)
        return super().execute(input)


class AdaAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0., ada_token=False):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = dim // num_heads
        self.scale        = self.head_dim ** -0.5
        self.query        = DynaLinear(dim, dim, bias=qkv_bias)
        self.key          = DynaLinear(dim, dim, bias=qkv_bias)
        self.value        = DynaLinear(dim, dim, bias=qkv_bias)
        self.proj         = DynaLinear(dim, dim)
        self.attn_drop    = nn.Dropout(attn_drop)
        self.proj_drop    = nn.Dropout(proj_drop)
        self.token_select = SimpleTokenSelect(dim) if ada_token else None

    def execute(self, x, width_select=None, head_select=None, token_select=None):
        B, N, C = x.shape
        q = self.query(x, width_select=width_select)\
                .reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.key(x,   width_select=width_select)\
                .reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.value(x,  width_select=width_select)\
                .reshape(B, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        attn             = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_pre_softmax = attn
        attn             = attn.softmax(dim=-1)
        if head_select is not None:
            attn = attn * head_select.view(B, self.num_heads, 1, 1)
        attn_origin = attn
        if self.token_select is not None:
            attn, token_select = self.token_select(
                x, attn, attn_pre_softmax, token_select=token_select)
        attn = self.attn_drop(attn)
        x    = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        if width_select is not None:
            x = self.proj(x, width_select=width_select.transpose(0, 2, 1))
        else:
            x = self.proj(x)
        if token_select is not None:
            x = x * token_select.unsqueeze(-1)
        return self.proj_drop(x), attn_origin, token_select


class AdaMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = DynaLinear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = DynaLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x, width_select=None):
        x = self.fc1(x, width_select=width_select)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)   # fc2 不传 width_select
        return self.drop(x)


class StepAdaBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 ada_head=False, ada_layer=False, is_token_select=False, **kwargs):
        super().__init__()
        self.norm1        = nn.LayerNorm(dim)
        self.attn         = AdaAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, ada_token=is_token_select)
        self.drop_path    = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2        = nn.LayerNorm(dim)
        self.mlp          = AdaMlp(in_features=dim,
                                    hidden_features=int(dim * mlp_ratio), drop=drop)
        self.head_select  = BlockHeadSelect(dim, num_heads) if ada_head  else None
        self.layer_select = BlockLayerSelect(dim, 2)        if ada_layer else None

    def execute(self, x):
        policy_token = x[:, 0]
        head_sel, width_sel, head_logits = None, None, None
        if self.head_select is not None:
            head_sel, width_sel, head_logits = self.head_select(policy_token)
        layer_sel, layer_logits = None, None
        if self.layer_select is not None:
            layer_sel, layer_logits = self.layer_select(policy_token)

        msa_keep = layer_sel[:, 0].view(-1, 1, 1) if layer_sel is not None else 1.0
        attn_x, attn_origin, token_sel = self.attn(
            self.norm1(x), width_select=width_sel, head_select=head_sel)
        x = x + self.drop_path(attn_x) * msa_keep

        mlp_keep = layer_sel[:, 1].view(-1, 1, 1) if layer_sel is not None else 1.0
        mlp_x = self.mlp(self.norm2(x))
        if token_sel is not None:
            mlp_x = mlp_x * token_sel.unsqueeze(-1)
        x = x + self.drop_path(mlp_x) * mlp_keep

        return x, attn_origin, head_sel, layer_sel, token_sel, head_logits, layer_logits