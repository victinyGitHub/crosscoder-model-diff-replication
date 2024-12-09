from utils import *
from trainer import Trainer

device = 'cuda:0'

N_1 = 30
pythia1 = HookedTransformer.from_pretrained("pythia-19m", checkpoint_index=N_1, device = device)

print(pythia1.cfg)

data = load_dataset(
    "EleutherAI/pile-deduped-pythia-random-sampled", 
    split="train"
)
data.set_format(type="torch", columns=["Tokens"])
all_tokens = data["Tokens"]
print(f"Number of tokens: {len(all_tokens)}")