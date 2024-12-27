from utils import *
from trainer import Trainer
import torch
import gc

device = 'cuda:0'
all_tokens = load_pile_deduped_pythia_random_sampled().reshape((-1, 1024))
N_1 = 30
pythia1 = HookedTransformer.from_pretrained("pythia-1.3b", checkpoint_index=N_1, device = device)

for N_2 in [36, 40, 42, 45, 50, 60, 65, 70, 75, 80, 90]:
    pythia2 = HookedTransformer.from_pretrained("pythia-1.3b", checkpoint_index=N_2, device = device)

    default_cfg = {
        "seed": 69,
        "batch_size": 4096,
        "buffer_mult": 128,
        "lr": 5e-5,
        "num_tokens": 320_000_000, # current pile dataset has 320 million tokens. going for 1 epoch due to time constraints
        "l1_coeff": 2.5,
        "beta1": 0.9,
        "beta2": 0.999,
        "d_in": pythia1.cfg.d_model,
        "dict_size": 2**14, # this is the size of hidden dim
        "seq_len": pythia1.cfg.n_ctx,
        "enc_dtype": "fp32",
        "model_name": "pythia1.3b_automateds",
        "site": "resid_pre",
        "device": "cuda:0",
        "model_batch_size": 4,
        "log_every": 50,
        "save_every": 30000,
        "dec_init_norm": 0.08,
        "hook_point": "blocks.20.hook_resid_pre",
        "wandb_project": "pythia crosscoder 1.3b training",
        "wandb_entity": "xiaoxiaoanddali-imperial-college-london",
    }
    cfg = arg_parse_update_cfg(default_cfg)

    trainer = Trainer(cfg, pythia1, pythia2, all_tokens)
    trainer.train()

    # Clear memory    
    del pythia2, trainer
    torch.cuda.empty_cache()
    gc.collect()