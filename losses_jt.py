import jittor as jt
from jittor import nn

def binary_entropy(prob, eps=1e-7):
    neg_entro = prob * prob.clamp(min_v=eps).log() + \
                (1 - prob) * (1 - prob).clamp(min_v=eps).log()
    return -neg_entro

def soft_cross_entropy(predicts, targets):
    student_likelihood = nn.log_softmax(predicts, dim=-1)
    targets_prob       = nn.softmax(targets, dim=-1)
    return (-targets_prob * student_likelihood).sum(-1).mean()

class AdaLoss(nn.Module):
    def __init__(self, base_criterion,
                 head_target_ratio=0.66, layer_target_ratio=0.7, token_target_ratio=0.75,
                 loss_ratio=0.5, entropy_weight=0.3):
        super().__init__()
        self.base_criterion = base_criterion
        self.head_target    = head_target_ratio
        self.layer_target   = layer_target_ratio
        self.token_target   = token_target_ratio
        self.loss_ratio     = loss_ratio
        self.entropy_weight = entropy_weight

    def execute(self, outputs, y):
        x, head_select, layer_select, token_select = outputs[:4]
        logits_dict = outputs[-1]

        base_loss  = self.base_criterion(x, y)
        head_loss  = self._compute_usage_loss(
            head_select,  self.head_target,  logits_dict.get('head_select_logits'))
        layer_loss = self._compute_usage_loss(
            layer_select, self.layer_target, logits_dict.get('layer_select_logits'))
        token_loss = self._compute_usage_loss(
            token_select, self.token_target, None)

        total_loss = base_loss + self.loss_ratio * (head_loss + layer_loss + token_loss)

        return total_loss, {
            "base_loss" : float(base_loss.data[0]),
            "head_loss" : float(head_loss.data[0]),
            "layer_loss": float(layer_loss.data[0]),
            "token_loss": float(token_loss.data[0]),
        }

    def _compute_usage_loss(self, select_mask, target_ratio, logits):
        if select_mask is None:
            return (jt.zeros(1) * 0.0).sum()

        usage_mean = select_mask.float32().mean()
        flops_loss = (usage_mean - target_ratio).pow(2)

        entropy_loss = (jt.zeros(1) * 0.0).sum()
        if logits is not None and self.entropy_weight > 0:
            entropy_loss = binary_entropy(logits.sigmoid()).mean()

        return flops_loss - self.entropy_weight * entropy_loss