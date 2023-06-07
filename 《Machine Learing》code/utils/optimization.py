import logging
import re, math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import torch.nn.init as init

logger = logging.getLogger(__name__)


def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal_(p.data)


def set_optimizer(model, args, num_warmup_steps, num_training_steps, last_epoch=-1):
    plm = hasattr(model.input_layer, 'plm_model')
    if plm and args.layerwise_decay <= 0.:
        for n, p in model.named_parameters():
            if 'plm_model' in n:
                p.requires_grad = False
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ['bias', 'LayerNorm.weight']

    other_params = [
        p for n, p in model.named_parameters() if 'plm_model' not in n
    ]
    LayerNorm_params = [n for n, p in model.named_parameters() if 'plm_model' not in n and 'norm' in n]
    if plm and 0. < args.layerwise_decay <= 0.5: 
        grouped_params = [
            {'params': list(set([p for n, p in params if 'plm_model' in n and not any(nd in n for nd in no_decay)])), 'lr': args.layerwise_decay * args.lr, 'weight_decay': args.l2},
            {'params': list(set([p for n, p in params if 'plm_model' in n and any(nd in n for nd in no_decay)])), 'lr': args.layerwise_decay * args.lr, 'weight_decay': 0.0},
            {'params': list(set([p for n, p in params if 'plm_model' not in n and not any(nd in n for nd in no_decay)])), 'weight_decay': args.l2},
            {'params': list(set([p for n, p in params if 'plm_model' not in n and any(nd in n for nd in no_decay)])), 'weight_decay': 0.0},
        ]
        print('Use seperate lr %f for pretrained model ...' % (args.lr * args.layerwise_decay))
    elif plm and 0.5 < args.layerwise_decay < 1.: 
        pattern = r'encoder\.layer\.(.*?)\.'
        num_layers = int(model.input_layer.plm_model.config.num_hidden_layers)
        groups = {"decay": defaultdict(list), "no_decay": defaultdict(list)}
        for n, p in params:
            res = re.search(pattern, n) if 'plm_model' in n else None
            depth = int(res.group(1)) if res is not None else 0 if 'plm_model' in n else num_layers
            if any(nd in n for nd in no_decay):
                groups["no_decay"][int(depth)].append(p)
            else:
                groups["decay"][int(depth)].append(p)
        grouped_params = []
        for d in groups["decay"]:
            lr = args.lr * (args.layerwise_decay ** (num_layers - d))
            grouped_params.append({'params': list(set(groups["decay"][d])), 'lr': lr, 'weight_decay': args.l2})
        for d in groups["no_decay"]:
            lr = args.lr * (args.layerwise_decay ** (num_layers - d))
            grouped_params.append({'params': list(set(groups["no_decay"][d])), 'lr': lr, 'weight_decay': 0.0})
        print('Use layerwise decay (rate %f) lr %f for pretrained model ...' % (args.layerwise_decay, args.lr))
    else:
        grouped_params = [
            {'params': list(set([p for n, p in params if not any(nd in n for nd in no_decay)])), 'weight_decay': args.l2},
            {'params': list(set([p for n, p in params if any(nd in n for nd in no_decay)])), 'weight_decay': 0.0},
        ]
        print('Use the same lr %f for all parameters ...' % (args.lr))
    optimizer = AdamW(grouped_params, lr=args.lr, max_grad_norm=args.max_norm)
    schedule_func = schedule_dict[args.lr_schedule]
    scheduler = schedule_func(optimizer, num_warmup_steps, num_training_steps, last_epoch=last_epoch)
    return optimizer, scheduler

def get_ratsql_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return max(0.0, math.sqrt((num_training_steps - current_step) / float(num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule(optimizer, *args, last_epoch=-1):

    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

schedule_dict = {
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup,
    "ratsql": get_ratsql_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, max_grad_norm=-1, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, max_grad_norm=max_grad_norm, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
