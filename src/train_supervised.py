import os
import random
import resource
import shutil
import sys
import json
import time
import yaml
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.datasets.wilds import make_iwildcam
from src.helper import init_model
from src.models.head import ViTClassifier
from src.models.loss import make_criterion
from src.transforms import make_transforms, make_transform_eval
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter, resolve_log_dir
from src.utils.optimizers import LARS
from src.eval_wilds import main as eval_wilds_main

# Repository root used to store downloadable checkpoints in the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _set_seed(seed):
    """Seed all RNGs so multi-seed runs are reproducible and distinct."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _peak_host_ram_gb():
    """Peak resident set size (RSS) of this process, in GB.

    Uses resource.getrusage(RUSAGE_SELF).ru_maxrss, which on Linux is reported
    in kilobytes. This is the process high-water mark; with tasks_per_node=1 it
    reflects the whole training worker. Compare against the SLURM mem request
    (e.g. 180G) to right-size future jobs.
    """
    try:
        maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(maxrss_kb) / (1024.0 * 1024.0)
    except (ValueError, OSError):
        return None


def _peak_gpu_mem_gb(device):
    """Peak allocated and reserved GPU memory (GB) since the last reset."""
    if not torch.cuda.is_available():
        return None, None
    try:
        alloc = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
        return float(alloc), float(reserved)
    except (RuntimeError, ValueError):
        return None, None


def _compute_class_counts(train_dataset, num_classes):
    """Extract per-class sample counts from the iWildCam training subset.

    Supports the WildsToTorchWrapper around a WILDS subset or a torch Subset.
    """
    if not hasattr(train_dataset, "dataset"):
        raise TypeError("Cannot determine class counts from train_dataset")

    underlying = train_dataset.dataset
    if isinstance(underlying, torch.utils.data.Subset):
        base_labels = np.asarray(underlying.dataset.y_array)
        indices = np.asarray(underlying.indices)
        labels = base_labels[indices]
    elif hasattr(underlying, "y_array"):
        labels = np.asarray(underlying.y_array)
    else:
        raise TypeError("Cannot determine class counts from train_dataset")

    counts = np.bincount(labels, minlength=num_classes)
    return counts.astype(np.float32)


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def distributed_average(value, device):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def distributed_sum(value, device):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _find_metric(metrics, key):
    if isinstance(metrics, dict):
        if key in metrics:
            return metrics[key]
        for value in metrics.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    if isinstance(metrics, list):
        for item in metrics:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _collect_eval_rows(eval_root, metric_key):
    rows = []
    for dirpath, _, filenames in os.walk(eval_root):
        for fname in filenames:
            if not fname.endswith("_metrics.json"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r") as f:
                    metrics = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            value = _find_metric(metrics, metric_key)
            if value is None:
                continue
            run_name = os.path.basename(os.path.dirname(path))
            rows.append((float(value), run_name, path))
    return rows


class EarlyStopping:
    def __init__(
        self,
        enabled=True,
        patience=10,
        min_delta=0.0,
        min_epochs=0,
        restore_best_weights=True,
    ):
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_state = None

    def state_dict(self):
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "restore_best_weights": self.restore_best_weights,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.enabled = state.get("enabled", self.enabled)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.min_epochs = state.get("min_epochs", self.min_epochs)
        self.restore_best_weights = state.get(
            "restore_best_weights", self.restore_best_weights
        )
        self.best_metric = state.get("best_metric", self.best_metric)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self.bad_epochs = state.get("bad_epochs", self.bad_epochs)

    def step(self, epoch, metric, model_module):
        if not self.enabled:
            return False, False

        improved = metric < (self.best_metric - self.min_delta)
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
            if self.restore_best_weights:
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model_module.state_dict().items()
                }
            return True, False

        self.bad_epochs += 1
        should_stop = (
            epoch + 1
        ) >= self.min_epochs and self.bad_epochs >= self.patience
        return False, should_stop

    def restore(self, model_module, device):
        if self.restore_best_weights and self.best_state is not None:
            model_module.load_state_dict(self.best_state)
            model_module.to(device)


def evaluate(model, loader, criterion, device, use_bfloat16):
    model.eval()
    loss_sum = 0.0
    n_correct = 0.0
    n_total = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_bfloat16, dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            preds = outputs.argmax(dim=1)
            n_correct += float((preds == labels).sum().item())
            n_total += float(batch_size)
            loss_sum += float(loss.item()) * float(batch_size)

    global_loss_sum = distributed_sum(loss_sum, device)
    global_correct = distributed_sum(n_correct, device)
    global_total = distributed_sum(n_total, device)

    val_loss = global_loss_sum / max(global_total, 1.0)
    val_acc = global_correct / max(global_total, 1.0)
    return val_loss, val_acc


def main(args, resume_preempt=False):
    del resume_preempt

    world_size, rank = init_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for supervised distributed training")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    m_args = args["meta"]
    o_args = args["optimization"]
    d_args = args["data"]
    mk_args = args["mask"]
    l_args = args["logging"]
    v_args = args["validation"]
    es_args = o_args["early_stopping"]

    seed = int(m_args.get("seed", _GLOBAL_SEED))
    _set_seed(seed)
    logger.info(f"Using seed {seed}")
    accum_steps = o_args.get("gradient_accumulation_steps", 1)
    use_gradient_checkpointing = m_args.get("use_gradient_checkpointing", True)

    folder = resolve_log_dir(args, stage="train")
    tag = l_args["write_tag"]

    with open(os.path.join(folder, "params-supervised.yaml"), "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(args, f)

    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    best_path = os.path.join(folder, f"{tag}-best.pth.tar")
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%.6f", "train_loss"),
        ("%.6f", "val_loss"),
        ("%.6f", "val_acc"),
        ("%.6e", "lr"),
        ("%.6f", "best_val_loss"),
        ("%d", "best_epoch"),
        ("%d", "early_stop"),
    )

    encoder, _ = init_model(
        device=device,
        patch_size=mk_args["patch_size"],
        crop_size=d_args["crop_size"],
        model_name=m_args["model_name"],
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    representation_type = m_args.get("representation_type", "last_avgpool")
    head_type = m_args.get("head_type", "linear")
    model = ViTClassifier(
        encoder,
        m_args["num_classes"],
        m_args["embed_dim"],
        representation_type=representation_type,
        head_type=head_type,
    ).to(device)

    if o_args["freeze_weights"]:
        logger.info("Freezing encoder weights (Linear Probing mode)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
    else:
        logger.info("Training full model (Fine-tuning mode)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = o_args["optimizer"].lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=o_args["lr"], weight_decay=o_args["weight_decay"]
        )
    elif optimizer_name == "lars":
        optimizer = LARS(
            params,
            lr=o_args["lr"],
            weight_decay=o_args["weight_decay"],
            momentum=o_args.get("momentum", 0.9),
            eta=o_args.get("lars_eta", 0.001),
            eps=o_args.get("lars_eps", 1e-8),
            exclude_bias_and_norm=o_args.get("lars_exclude_bias_and_norm", True),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=o_args["lr"],
            momentum=o_args.get("momentum", 0.9),
            weight_decay=o_args["weight_decay"],
        )

    scheduler = None
    lr_schedule = o_args.get("lr_schedule", "cosine").lower()
    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=o_args.get("step_milestones", [15, 30, 45]),
            gamma=o_args.get("step_gamma", 0.1),
        )
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=o_args["epochs"], eta_min=o_args["final_lr"]
        )

    train_transform = make_transforms(
        crop_size=d_args["crop_size"],
        horizontal_flip=d_args["use_horizontal_flip"],
        color_distortion=d_args["use_color_distortion"],
        color_jitter=d_args["color_jitter_strength"],
        gaussian_blur=d_args["use_gaussian_blur"],
        use_random_resized_crop=d_args.get("use_random_resized_crop", True),
    )
    val_transform = make_transform_eval(
        crop_size=d_args["crop_size"],
    )

    label_fraction = d_args.get("label_fraction", None)
    if label_fraction is not None:
        logger.info(
            f"Using label_fraction={label_fraction} for the training split (seed={seed})"
        )

    train_dataset, train_loader, train_sampler = make_iwildcam(
        transform=train_transform,
        split="train",
        batch_size=d_args["batch_size"],
        root_path=d_args["root_path"],
        rank=rank,
        world_size=world_size,
        collator=None,
        num_workers=d_args["num_workers"],
        pin_mem=d_args["pin_mem"],
        drop_last=True,
        label_fraction=label_fraction,
        seed=seed,
    )

    _, val_loader, val_sampler = make_iwildcam(
        transform=val_transform,
        split="val",
        batch_size=d_args["batch_size"],
        root_path=d_args["root_path"],
        rank=rank,
        world_size=world_size,
        collator=None,
        num_workers=d_args["num_workers"],
        pin_mem=d_args["pin_mem"],
        drop_last=False,
    )

    loss_name = o_args.get("loss", "cross_entropy")
    class_counts = _compute_class_counts(train_dataset, m_args["num_classes"])
    criterion = make_criterion(loss_name, class_counts=class_counts, device=device)
    logger.info(
        f"Using loss: {loss_name} (class_counts sum={int(class_counts.sum())}, "
        f"nonzero={int((class_counts > 0).sum())})"
    )
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    eval_every = int(v_args["eval_every"])

    early_stopper = EarlyStopping(
        enabled=es_args["enabled"],
        patience=es_args["patience"],
        min_delta=es_args["min_delta"],
        min_epochs=es_args["min_epochs"],
        restore_best_weights=es_args["restore_best_weights"],
    )

    start_epoch = 0

    checkpoint_to_load = None
    resuming_interrupted = False
    if os.path.exists(latest_path):
        checkpoint_to_load = latest_path
        resuming_interrupted = True
    elif m_args["load_checkpoint"]:
        r_file = m_args["read_checkpoint"]
        checkpoint_folder = m_args["checkpoint_folder"]
        if os.path.isabs(r_file):
            checkpoint_to_load = r_file
        else:
            checkpoint_to_load = os.path.join(checkpoint_folder, r_file)

    if checkpoint_to_load is not None and not os.path.exists(checkpoint_to_load):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_load}")

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        checkpoint = torch.load(checkpoint_to_load, map_location="cpu")
        if resuming_interrupted and "model" in checkpoint:
            model.module.load_state_dict(checkpoint["model"])
            if "opt" in checkpoint:
                optimizer.load_state_dict(checkpoint["opt"])
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", 0))
            early_stopper.load_state_dict(checkpoint.get("early_stopping", {}))
            logger.info(
                f"Resuming training from {checkpoint_to_load} at epoch {start_epoch}"
            )
        else:
            encoder_state = checkpoint.get("encoder")
            if encoder_state is None and "model" in checkpoint:
                encoder_state = {
                    k.replace("encoder.", "", 1): v
                    for k, v in checkpoint["model"].items()
                    if k.startswith("encoder.")
                }
            if encoder_state is None:
                raise KeyError(
                    f"No encoder weights found in checkpoint: {checkpoint_to_load}"
                )
            encoder_state = strip_module_prefix(encoder_state)
            msg = model.module.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(
                f"Loaded pre-trained encoder from {checkpoint_to_load} with msg: {msg}"
            )

    def save_checkpoint(epoch, train_loss, val_loss, val_acc, is_best=False):
        save_dict = {
            "model": model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": args,
            "early_stopping": early_stopper.state_dict(),
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=epoch))
            if is_best:
                torch.save(save_dict, best_path)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.perf_counter()
    epochs_run = 0
    early_stopped = False
    for epoch in range(start_epoch, o_args["epochs"]):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epochs_run = epoch + 1

        model.train()
        if o_args["freeze_weights"]:
            model.module.encoder.eval()

        loss_meter = AverageMeter()

        optimizer.zero_grad()
        for itr, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                enabled=m_args["use_bfloat16"], dtype=torch.bfloat16
            ):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            (loss / accum_steps).backward()

            if (itr + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_meter.update(loss.item(), n=labels.size(0))

            if itr % log_freq == 0 and rank == 0:
                logger.info(
                    f"Epoch {epoch + 1} [{itr}/{len(train_loader)}] Train Loss: {loss_meter.avg:.4f}"
                )

        train_loss = distributed_average(loss_meter.avg, device)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == o_args["epochs"])
        if do_eval:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_bfloat16=m_args["use_bfloat16"],
            )
            is_best, should_stop = early_stopper.step(epoch + 1, val_loss, model.module)
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            is_best, should_stop = False, False

        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            logger.info(
                f"Epoch {epoch + 1} done | train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f} best_val_loss={early_stopper.best_metric:.6f}"
            )

        csv_logger.log(
            epoch + 1,
            train_loss,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
            early_stopper.best_metric,
            early_stopper.best_epoch,
            int(should_stop),
        )

        save_checkpoint(epoch + 1, train_loss, val_loss, val_acc, is_best=is_best)

        stop_tensor = torch.tensor([int(should_stop)], device=device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(stop_tensor, src=0)

        if bool(stop_tensor.item()):
            early_stopped = True
            if rank == 0:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}. Best val_loss={early_stopper.best_metric:.6f} @ epoch {early_stopper.best_epoch}"
                )
            break

    train_time_seconds = time.perf_counter() - train_start_time
    peak_host_ram_gb = _peak_host_ram_gb()
    peak_gpu_alloc_gb, peak_gpu_reserved_gb = _peak_gpu_mem_gb(device)
    if rank == 0:
        logger.info(
            f"Training finished: epochs_run={epochs_run} "
            f"early_stopped={early_stopped} "
            f"train_time={_format_hms(train_time_seconds)} "
            f"({train_time_seconds:.1f}s) "
            f"peak_host_ram_gb={peak_host_ram_gb} "
            f"peak_gpu_alloc_gb={peak_gpu_alloc_gb} "
            f"peak_gpu_reserved_gb={peak_gpu_reserved_gb}"
        )

    if early_stopper.enabled and early_stopper.restore_best_weights:
        if rank == 0:
            logger.info("Restoring best model weights before exit")
        early_stopper.restore(model.module, device)
        if rank == 0:
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "epoch": early_stopper.best_epoch,
                    "val_loss": early_stopper.best_metric,
                    "args": args,
                },
                best_path,
            )

    if rank == 0:
        # Evaluate on all official WILDS leaderboard splits for submission.
        # WILDS-iWildCam: "id_test" == Test ID, "test" == Test OOD.
        eval_splits = [
            ("val", "iwildcam_val"),
            ("id_val", "iwildcam_id_val"),
            ("id_test", "iwildcam_id_test"),
            ("test", "iwildcam_test"),
        ]

        def _make_eval_args(split, write_tag):
            return {
                "meta": {
                    "seed": seed,
                    "model_name": m_args["model_name"],
                    "embed_dim": m_args["embed_dim"],
                    "num_classes": m_args["num_classes"],
                    "patch_size": mk_args.get("patch_size", m_args.get("patch_size", 16)),
                    "crop_size": d_args.get("crop_size", m_args.get("crop_size", 224)),
                    "use_bfloat16": m_args.get("use_bfloat16", True),
                    "representation_type": representation_type,
                    "head_type": head_type,
                    "checkpoint_path": best_path,
                    "force_single_process": True,
                },
                "data": {
                    "batch_size": d_args.get("batch_size", 128),
                    "root_path": d_args.get("root_path", "./wilds_data"),
                    "num_workers": d_args.get("num_workers", 8),
                    "pin_mem": d_args.get("pin_mem", True),
                    "split": split,
                    "download": True,
                },
                "logging": {
                    "write_tag": write_tag,
                    "auto_folder": True,
                },
            }

        eval_results = {}
        eval_folder = None
        for split, write_tag in eval_splits:
            eval_args = _make_eval_args(split, write_tag)
            result = eval_wilds_main(args=eval_args)
            eval_results[split] = result
            if eval_folder is None:
                eval_folder = eval_args.get("logging", {}).get("folder")

        # Common run-level info folded into every metrics JSON + the params summary.
        run_info = {
            "seed": seed,
            "label_fraction": label_fraction if label_fraction is not None else 1.0,
            "train_time_seconds": float(train_time_seconds),
            "train_time_hms": _format_hms(train_time_seconds),
            "epochs_run": int(epochs_run),
            "configured_epochs": int(o_args["epochs"]),
            "best_epoch": int(early_stopper.best_epoch),
            "early_stopped": bool(early_stopped),
            "best_val_loss": float(early_stopper.best_metric),
            "peak_host_ram_gb": peak_host_ram_gb,
            "peak_gpu_alloc_gb": peak_gpu_alloc_gb,
            "peak_gpu_reserved_gb": peak_gpu_reserved_gb,
        }

        # Fold run_info into each split's metrics JSON so an aggregator can read
        # metrics + seed + timing + epochs from a single file per split.
        for split, result in eval_results.items():
            if not result:
                continue
            metrics_path = result.get("metrics_path")
            if not metrics_path or not os.path.exists(metrics_path):
                continue
            try:
                with open(metrics_path, "r") as f:
                    metrics_obj = json.load(f)
                metrics_obj["run_info"] = run_info
                metrics_obj["split"] = split
                with open(metrics_path, "w") as f:
                    json.dump(metrics_obj, f, indent=2, sort_keys=True)
            except (OSError, json.JSONDecodeError):
                logger.warning(f"Could not augment metrics JSON for split {split}")

        if eval_folder:
            try:
                params_out = yaml.safe_load(yaml.dump(args))
                params_out.setdefault("meta", {})["representation_type"] = representation_type
                params_out.setdefault("meta", {})["head_type"] = head_type
                params_out["results"] = {
                    "best_val_loss": float(early_stopper.best_metric),
                    "best_epoch": int(early_stopper.best_epoch),
                    "best_checkpoint": best_path,
                    "eval_checkpoint": (
                        os.path.join(eval_folder, "best_model.pth.tar")
                        if eval_folder else best_path
                    ),
                    "seed": seed,
                    "train_time_seconds": float(train_time_seconds),
                    "train_time_hms": _format_hms(train_time_seconds),
                    "epochs_run": int(epochs_run),
                    "configured_epochs": int(o_args["epochs"]),
                    "early_stopped": bool(early_stopped),
                    "peak_host_ram_gb": peak_host_ram_gb,
                    "peak_gpu_alloc_gb": peak_gpu_alloc_gb,
                    "peak_gpu_reserved_gb": peak_gpu_reserved_gb,
                    "eval_metrics_val": (
                        eval_results.get("val", {}).get("metrics")
                        if eval_results.get("val")
                        else None
                    ),
                    "eval_metrics_id_val": (
                        eval_results.get("id_val", {}).get("metrics")
                        if eval_results.get("id_val")
                        else None
                    ),
                    "eval_metrics_id_test": (
                        eval_results.get("id_test", {}).get("metrics")
                        if eval_results.get("id_test")
                        else None
                    ),
                    "eval_metrics_test": (
                        eval_results.get("test", {}).get("metrics")
                        if eval_results.get("test")
                        else None
                    ),
                }
                with open(os.path.join(eval_folder, "params-supervised.yaml"), "w") as f:
                    yaml.dump(params_out, f)
                with open(os.path.join(eval_folder, "params.yaml"), "w") as f:
                    yaml.dump(params_out, f)

                # Preserve the trained checkpoint so the WILDS submission generator can
                # re-load it. Also copy to the project root if a submission name is set.
                eval_checkpoint_path = os.path.join(eval_folder, "best_model.pth.tar")
                try:
                    shutil.copy(best_path, eval_checkpoint_path)
                    logger.info(f"Best checkpoint copied to {eval_checkpoint_path}")
                except OSError:
                    logger.warning("Could not copy best checkpoint to eval folder")

                submission_model_name = m_args.get("submission_model_name")
                if submission_model_name:
                    root_checkpoint_path = os.path.join(
                        PROJECT_ROOT, f"{submission_model_name}_seed{seed}.pth.tar"
                    )
                    try:
                        shutil.copy(best_path, root_checkpoint_path)
                        logger.info(f"Best checkpoint copied to {root_checkpoint_path}")
                    except OSError:
                        logger.warning(
                            f"Could not copy best checkpoint to {root_checkpoint_path}"
                        )
            except OSError:
                logger.warning("Could not write supervised params to eval folder")

        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except OSError:
                logger.warning("Could not remove supervised run folder")


if __name__ == "__main__":
    raise RuntimeError(
        "Use main_distributed_supervised.py to launch this script with a config file."
    )
