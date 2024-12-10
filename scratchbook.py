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
print(data.shape)
# Get a small sample of tokens and decode them
sample_size = 10
indices = torch.arange(0, len(all_tokens))
sampled_tokens = [all_tokens[i] for i in indices]

print("\nSample texts from the dataset:")
print("-" * 50)
for i, tokens in enumerate(sampled_tokens):
    text = pythia1.to_string(tokens)
    print(f"\nSample {i+1}:")
    print(text)
    print("-" * 50)
