# %%
from utils import *
from trainer import Trainer
# %%
device = 'cuda:0'

N_1 = 30
pythia1 = HookedTransformer.from_pretrained("pythia-19m", checkpoint_index=N_1, device = device)

N_2 = 60
pythia2 = HookedTransformer.from_pretrained("pythia-19m", checkpoint_index=N_2, device = device)

# base_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b", 
#     device=device, 
# )

# chat_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b-it", 
#     device=device, 
# )

# %%
all_tokens = load_pile_lmsys_mixed_tokens()

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 1024,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 20_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": pythia1.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 2048,
    "enc_dtype": "fp32",
    "model_name": "pythia-19m",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.3.hook_resid_pre",
    "wandb_project": "crosscoder-training",
    "wandb_entity": "xiaoxiaoanddali-imperial-college-london",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, pythia1, pythia2, all_tokens)
trainer.train()
# %%