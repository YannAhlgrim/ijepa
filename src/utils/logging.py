# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def _format_value(value):
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "-".join(_format_value(v) for v in value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def build_run_name(args):
    meta_args = args.get("meta", {})
    data_args = args.get("data", {})
    opt_args = args.get("optimization", {})
    mask_args = args.get("mask", {})
    val_args = args.get("validation", {})

    model_name = meta_args.get("model_name", "model")
    tag = meta_args.get("tag")
    representation_type = meta_args.get("representation_type")
    head_type = meta_args.get("head_type")
    patch_size = mask_args.get("patch_size", meta_args.get("patch_size"))
    crop_size = data_args.get("crop_size", meta_args.get("crop_size"))
    batch_size = data_args.get("batch_size")
    optimizer = opt_args.get("optimizer", "opt")
    lr = opt_args.get("lr")
    weight_decay = opt_args.get("weight_decay")
    epochs = opt_args.get("epochs")

    submission_model_name = meta_args.get("submission_model_name")
    parts = [submission_model_name] if submission_model_name else []
    parts.append(model_name)

    def add(prefix, value):
        formatted = _format_value(value)
        if formatted is not None:
            parts.append(f"{prefix}{formatted}")

    add("p", patch_size)
    add("c", crop_size)
    if tag:
        parts.append(str(tag))
    add("bs", batch_size)
    add("rep", representation_type)
    add("head", head_type)
    parts.append(str(optimizer).lower())
    add("lr", lr)
    add("wd", weight_decay)
    add("ep", epochs)
    add("sched", opt_args.get("lr_schedule"))
    add("ms", opt_args.get("step_milestones"))
    add("sg", opt_args.get("step_gamma"))
    add("wu", opt_args.get("warmup"))
    add("mom", opt_args.get("momentum"))
    add("leta", opt_args.get("lars_eta"))
    add("leps", opt_args.get("lars_eps"))
    add("ipe", opt_args.get("ipe_scale"))
    add("cs", data_args.get("crop_scale"))
    add("lf", data_args.get("label_fraction"))
    add("eval", val_args.get("eval_every"))
    add("seed", meta_args.get("seed"))

    return "-".join([p for p in parts if p])


def _extract_run_name_from_checkpoint(meta_args):
    checkpoint_path = meta_args.get("checkpoint_path")
    if checkpoint_path:
        folder = os.path.dirname(checkpoint_path)
    else:
        folder = meta_args.get("checkpoint_folder")
    if not folder:
        return None
    return os.path.basename(os.path.normpath(folder))


def resolve_log_dir(args, stage="train"):
    log_args = args.setdefault("logging", {})
    auto_folder = log_args.get("auto_folder", True)
    if auto_folder or not log_args.get("folder"):
        run_name = log_args.get("run_name")
        if stage == "eval" and not run_name:
            meta_args = args.get("meta", {})
            run_name = _extract_run_name_from_checkpoint(meta_args)
        if not run_name:
            run_name = build_run_name(args)
        base_dir = "experiment_logs"
        if stage == "eval":
            folder = os.path.join(base_dir, "eval-wilds", run_name)
        else:
            folder = os.path.join(base_dir, run_name)
        log_args["folder"] = folder
    os.makedirs(log_args["folder"], exist_ok=True)
    return log_args["folder"]


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats
