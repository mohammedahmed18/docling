import logging
from typing import List, Optional

from docling.datamodel.accelerator_options import AcceleratorDevice

_log = logging.getLogger(__name__)


def decide_device(
    accelerator_device: str, supported_devices: Optional[List[AcceleratorDevice]] = None
) -> str:
    """
    Resolve the device based on the acceleration options and the available devices in the system.

    Rules:
    1. AUTO: Check for the best available device on the system.
    2. User-defined: Check if the device actually exists, otherwise fall-back to CPU
    """
    import torch

    # Cache hardware status first for use below
    cuda_built = torch.backends.cuda.is_built()
    cuda_avail = torch.cuda.is_available() if cuda_built else False
    mps_built = torch.backends.mps.is_built()
    mps_avail = torch.backends.mps.is_available() if mps_built else False

    has_cuda = cuda_built and cuda_avail
    has_mps = mps_built and mps_avail

    # Filter by supported_devices early
    if supported_devices is not None:
        if has_cuda and AcceleratorDevice.CUDA not in supported_devices:
            _log.info(
                "Removing CUDA from available devices because it is not in supported_devices"
            )
            has_cuda = False
        if has_mps and AcceleratorDevice.MPS not in supported_devices:
            _log.info(
                "Removing MPS from available devices because it is not in supported_devices"
            )
            has_mps = False

    # No matter what, CPU is always valid
    device = "cpu"

    # --- Main decision tree ---
    # Handle 'auto'
    if accelerator_device == AcceleratorDevice.AUTO.value:
        if has_cuda:
            device = "cuda:0"
        elif has_mps:
            device = "mps"
    # Handle CUDA (any cuda string)
    elif accelerator_device.startswith("cuda"):
        if has_cuda:
            # Try to get the index, minimally split and check
            colon = accelerator_device.find(":")
            if colon == -1:
                # Just "cuda"
                device = "cuda:0"
            else:
                idx_str = accelerator_device[colon + 1 :]
                if idx_str.isdigit():
                    cuda_index = int(idx_str)
                    cuda_count = torch.cuda.device_count()
                    if cuda_index < cuda_count:
                        device = f"cuda:{cuda_index}"
                    else:
                        _log.warning(
                            "CUDA device 'cuda:%d' is not available. Fall back to 'CPU'.",
                            cuda_index,
                        )
                else:
                    _log.warning(
                        "Invalid CUDA device format '%s'. Fall back to 'CPU'",
                        accelerator_device,
                    )
        else:
            _log.warning("CUDA is not available in the system. Fall back to 'CPU'")

    elif accelerator_device == AcceleratorDevice.MPS.value:
        if has_mps:
            device = "mps"
        else:
            _log.warning("MPS is not available in the system. Fall back to 'CPU'")

    elif accelerator_device == AcceleratorDevice.CPU.value:
        device = "cpu"

    else:
        _log.warning(
            "Unknown device option '%s'. Fall back to 'CPU'", accelerator_device
        )

    _log.info("Accelerator device: '%s'", device)
    return device
