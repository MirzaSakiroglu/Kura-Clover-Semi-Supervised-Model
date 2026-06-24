"""
src.utils.device.py
Device Setter script
BoMeyering 2025
"""

import torch
import logging
import omegaconf
from omegaconf import OmegaConf
from src.utils.loggers import rank_log
from src.distributed import is_main_process

def set_torch_device(conf: OmegaConf) -> OmegaConf:
    """_summary_

    Args:
        conf (OmegaConf): _description_

    Returns:
        OmegaConf: _description_
    """

    logger = logging.getLogger()
    if not isinstance(conf, omegaconf.dictconfig.DictConfig):
        raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
    
    if 'device' in conf:
        # Check if requested device is CUDA
        if conf.device == 'cuda':
            if torch.cuda.is_available():
                device_str = f"cuda:{conf.local_rank}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                rank_log(conf.is_main, logger.info, "CUDA not available but Apple Silicon GPU found. Falling back to MPS.")
                device_str = "mps" # MPS typically doesn't use rank indexes on Mac
            else:
                rank_log(conf.is_main, logger.info, "CUDA/MPS not available. Falling back to CPU computation.")
                device_str = "cpu"
                
            conf.device = device_str
            
        elif conf.device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                conf.device = "mps"
            else:
                rank_log(conf.is_main, logger.info, "MPS requested but not available. Falling back to CPU.")
                conf.device = "cpu"
                
        elif conf.device != 'cpu':
            rank_log(conf.is_main, logger.info, "Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda', 'mps']. Falling back to 'cpu'.")
            conf.device = "cpu"
    else:
        rank_log(conf.is_main, logger.info, "No value set for 'conf.device'. Setting to 'cpu'.")
        conf.device = "cpu"