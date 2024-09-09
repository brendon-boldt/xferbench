# Developed from: https://github.com/toizzy/tilt-transfer/blob/master/corpora/constructed_corpora/hierarchical_parens.py

from collections import deque
import sys

import numpy as np
import datasets  # type: ignore

open_prob = 0.4

N_TOKENS = 100_000_000
CONTEXT_SIZE = 256
# Account for potential special tokens
VOCAB_SIZE = 30_000 - 10

# Zipf-Mandlebrot law
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4176592/
alpha = 1
beta = 2.7

def main() -> None:
    data = np.empty((N_TOKENS // CONTEXT_SIZE + 1) * CONTEXT_SIZE, dtype=np.int32)

    ps = (np.arange(VOCAB_SIZE) + 1 + beta) ** -alpha
    ps /= sum(ps)

    if len(sys.argv) >= 2:
        seed = int(sys.argv[1])
    else:
        seed = 0

    # Note: The seed was added after the dataset used in the paper was generated,
    # so exact reproduction is not gauranteed.
    rng = np.random.default_rng(seed)

    open_deque: deque[int] = deque()
    open_decision = rng.choice([0, 1], len(data))
    samples = rng.choice(VOCAB_SIZE, len(data), p=ps)
    for i in range(len(data)):
        if open_decision[i] or len(open_deque) == 0:
            data[i] = samples[i]
            open_deque.append(data[i])
        else:
            last_open = open_deque.pop()
            data[i] = last_open

    data = data.reshape(-1, CONTEXT_SIZE)

    ds = datasets.Dataset.from_dict({"input_ids": data})
    ds.save_to_disk(f"data/var/dyck-seed-{seed}")

if __name__ == "__main__":
    main()
