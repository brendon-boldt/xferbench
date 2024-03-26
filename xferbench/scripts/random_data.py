import numpy as np
import datasets  # type: ignore

rng = np.random.default_rng(0)

# Allow for special tokens
VOCAB_SIZE = 30000 - 10
CONTEXT_SIZE = 256
N_TOKENS = 100_000_000

n_examples = N_TOKENS // CONTEXT_SIZE + 1

input_ids = rng.choice(VOCAB_SIZE, (n_examples, CONTEXT_SIZE))

ds = datasets.Dataset.from_dict({"input_ids": input_ids})
ds.save_to_disk("data/baselines/rand")
