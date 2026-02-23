import jittor as jt
from jittor import nn
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jt.rand(shape)
    random_tensor = random_tensor.floor()
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def trunc_normal_(var, mean=0., std=1., a=-2., b=2.):
    shape = tuple(int(s) for s in var.shape)
    
    tmp = np.random.normal(mean, std, size=shape + (8,)).astype(np.float32)
    valid = (tmp >= a) & (tmp <= b)
    
    idx = valid.argmax(axis=-1, keepdims=True)
    result = np.take_along_axis(tmp, idx, axis=-1).squeeze(-1)
   
    result = np.clip(result, a, b).astype(np.float32)
    var.assign(result)