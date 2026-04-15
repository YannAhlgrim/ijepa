import os
import sys
import yaml
import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import numpy as np

from src.datasets.wilds import make_iwildcam
from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms
from src.models.head import ViTClassifier  # Import our new class
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args):
    # -- Init Distributed
    world_size, rank = init_distributed()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # -- Extract Config Params
    m_args = args["meta"]
    o_args = args["optimization"]
    d_args = args["data"]

    # -- 1. Initialize Encoder
    encoder, _ = init_model(
        device=device,
        patch_size=args.get("mask", {}).get("patch_size", 14),
        crop_size=d_args["crop_size"],
        model_name=m_args["model_name"],
    )

    # -- 2. Wrap in Classification Head
    embed_dim = m_args.get("embed_dim")
    model = ViTClassifier(encoder, m_args["num_classes"], embed_dim).to(device)

    # -- 3. Handle Freezing (Linear Probing vs Fine-Tuning)
    if o_args["freeze_weights"]:
        logger.info("Freezing encoder weights (Linear Probing mode)")
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False
    else:
        logger.info("Training full model (Fine-tuning mode)")

    # -- 4. Load Pre-trained Weights
    if m_args["load_checkpoint"]:
        load_path = os.path.join(args["logging"]["folder"], m_args["read_checkpoint"])
        checkpoint = torch.load(load_path, map_location="cpu")
        msg = model.encoder.load_state_dict(checkpoint["encoder"], strict=False)
        logger.info(f"Loaded encoder from {load_path} with msg: {msg}")

    # -- 5. Data Setup
    transform = make_transforms(crop_size=d_args["crop_size"])
    _, loader, sampler = make_iwildcam(
        transform=transform,
        split="train",
        batch_size=d_args["batch_size"],
        root_path=d_args["root_path"],
        rank=rank,
        world_size=world_size,
        collator=None,  # No mask collator needed
    )

    # -- 6. Optimizer Selection
    params = [p for p in model.parameters() if p.requires_grad]
    if o_args["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=o_args["lr"], weight_decay=o_args["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            params, lr=o_args["lr"], momentum=0.9, weight_decay=o_args["weight_decay"]
        )

    criterion = nn.CrossEntropyLoss().to(device)
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    # -- 7. Training Loop
    for epoch in range(o_args["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        loss_meter = AverageMeter()

        for itr, (imgs, labels, _) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(
                enabled=m_args["use_bfloat16"], dtype=torch.bfloat16
            ):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            if itr % 10 == 0 and rank == 0:
                logger.info(
                    f"Epoch {epoch} [{itr}/{len(loader)}] Loss: {loss_meter.avg:.4f}"
                )


if __name__ == "__main__":
    main()
