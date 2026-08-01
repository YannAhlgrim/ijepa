import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax (Ren et al., NeurIPS 2020).

    Adjusts the logits by the log of the training class frequencies so that the
    resulting softmax is calibrated to the training distribution, assigning
    larger gradients to rare classes. This is particularly useful for the
    iWildCam 2020-WILDS benchmark where the headline metric is macro F1 and
    species are heavily imbalanced.

    Parameters
    ----------
    class_counts : array-like of shape (num_classes,)
        Per-class training sample counts. Order must match the class indices.
    device : torch.device
        Device on which the frequency buffer will live.
    """

    def __init__(self, class_counts, device):
        super().__init__()
        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)
        # Small epsilon to avoid log(0) for classes absent from the training set.
        self.register_buffer(
            "log_class_freqs",
            torch.log(class_counts + 1e-6).to(device),
        )

    def forward(self, logits, targets):
        # Equivalent to: -log( exp(z_y) * n_y / sum_i exp(z_i) * n_i )
        return F.cross_entropy(logits + self.log_class_freqs, targets)


def make_criterion(loss_name, class_counts, device):
    """Build a classification criterion by name.

    Parameters
    ----------
    loss_name : str
        One of ``cross_entropy`` or ``balanced_softmax``.
    class_counts : array-like or None
        Per-class training counts; required for ``balanced_softmax``.
    device : torch.device
        Target device for the loss module.

    Returns
    -------
    nn.Module
        The requested loss module, moved to ``device``.
    """
    loss_name = str(loss_name).lower().strip() if loss_name else "cross_entropy"
    if loss_name == "balanced_softmax":
        if class_counts is None:
            raise ValueError(
                "class_counts are required for balanced_softmax loss"
            )
        return BalancedSoftmaxLoss(class_counts, device).to(device)
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss().to(device)
    raise ValueError(f"Unknown loss: {loss_name}")
