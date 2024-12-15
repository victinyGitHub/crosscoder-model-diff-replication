# %%
from utils import *
from trainer import Trainer
# %%
device = 'cuda:0'

N_1 = 30
pythia1 = HookedTransformer.from_pretrained("pythia-1.3b", checkpoint_index=N_1, device = device)

N_2 = 90
pythia2 = HookedTransformer.from_pretrained("pythia-1.3b", checkpoint_index=N_2, device = device)

# base_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b", 
#     device=device, 
# )

# chat_model = HookedTransformer.from_pretrained(
#     "gemma-2-2b-it", 
#     device=device, 
# )

# %%
all_tokens = load_pile_deduped_pythia_random_sampled()

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 320_000_000, # current pile dataset has 320 million tokens. going for 1 epoch due to time constraints
    "l1_coeff": 3,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": pythia1.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 2048,
    "enc_dtype": "fp32",
    "model_name": "pythia1.3b30v90l18",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.18.hook_resid_pre",
    "wandb_project": "pythia crosscoder 1.3b training",
    "wandb_entity": "xiaoxiaoanddali-imperial-college-london",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, pythia1, pythia2, all_tokens)
trainer.train()
# %%