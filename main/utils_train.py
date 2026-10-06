# Adapted from DynamiCrafter (https://github.com/Doubiiu/DynamiCrafter), Apache License 2.0.
# Modified for UniVerse: plain DDP by default; checkpoint loading with U-Net input-channel expansion
# and key checks.
import os
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict

UNET_PREFIX = "model.diffusion_model."
## U-Net input conv: [320, 8, 3, 3] in ViewCrafter, [320, 10, 3, 3] in UniVerse (+ inpainting / style mask)
UNET_INPUT_CONV = "model.diffusion_model.input_blocks.0.0.weight"


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    ## plain DDP by default; set `lightning.strategy` for others (e.g. a deepspeed config)
    if "strategy" in lightning_config:
        return lightning_config.strategy
    return "ddp"

def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint: %s"%pretrained_ckpt)

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        if "state_dict" in pl_sd:
            state_dict = pl_sd["state_dict"]
        elif "module" in pl_sd:
            ## deepspeed
            state_dict = OrderedDict((key[16:], value) for key, value in pl_sd["module"].items())
        else:
            state_dict = pl_sd

        model_state_dict = model.state_dict()
        filtered_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape != model_state_dict[k].shape:
                target = model_state_dict[k]
                if k == UNET_INPUT_CONV and v.dim() == target.dim() == 4 and v.shape[0] == target.shape[0] \
                        and v.shape[2:] == target.shape[2:] and v.shape[1] < target.shape[1]:
                    ## copy the existing input channels, zero-init the new ones (e.g. ViewCrafter 8 -> UniVerse 10)
                    mainlogger.info(f">>> Expanding {k}: {list(v.shape)} -> {list(target.shape)} (new channels zero-initialised)")
                    new_entry = torch.zeros_like(target)
                    new_entry[:, :v.shape[1], ...] = v
                    v = new_entry
                else:
                    raise RuntimeError(f"Shape mismatch for '{k}': checkpoint {list(v.shape)} vs model {list(target.shape)}. "
                                       f"Only the U-Net input conv ({UNET_INPUT_CONV}) can be expanded with new input channels.")
            filtered_state_dict[k] = v

        ## strict=False below tolerates missing or extra auxiliary keys, but not a checkpoint that does not
        ## fit the model (e.g. another key prefix) or lacks U-Net weights
        if not any(k in model_state_dict for k in filtered_state_dict):
            example = f" (e.g. checkpoint key '{next(iter(filtered_state_dict))}' vs model key '{next(iter(model_state_dict))}')" \
                if filtered_state_dict else ""
            raise RuntimeError(f"None of the {len(filtered_state_dict)} keys of {pretrained_ckpt} matches the model{example}. "
                               f"Wrong key prefix?")
        missing_unet = [k for k in model_state_dict if k.startswith(UNET_PREFIX) and k not in filtered_state_dict]
        if missing_unet:
            raise RuntimeError(f"{pretrained_ckpt} lacks {len(missing_unet)} U-Net weights of the model, e.g. {missing_unet[:3]}")

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        mainlogger.info(f">>> model checkpoint loaded: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
        if missing:
            mainlogger.warning(f"    missing (not in checkpoint, kept as initialised): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            mainlogger.warning(f"    unexpected (in checkpoint, ignored): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
