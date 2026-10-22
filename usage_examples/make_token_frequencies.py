"""Token frequency table (counts per token ID) of a tokenizer over a sample of its pretraining corpus.

It is the tokenizer prior of the ``tokstat`` region metrics (GFMBENCH_TOKEN_FREQ): how common each token
is in genomic DNA, independent of any benchmark task. CPU only.

Usage:
    python usage_examples/make_token_frequencies.py --kind bpe --tokenizer_path TOK.json --out bpe.npy
    python usage_examples/make_token_frequencies.py --kind nucleotide --out single.npy
"""
import argparse
from pathlib import Path

import numpy as np

from ham_dna_tokenizer.mlm_training import load_sequence_encoder

VOCABULARY_SIZE = 4096


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=["hamiltonian", "bpe", "nucleotide"])
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--shard", default="/mnt/lustre/data/projects/sense/common/data/pretrain/GenBank_DNABERT2/shards/train_000.txt",
                        help="text file with one DNA sequence per line")
    parser.add_argument("--num_lines", type=int, default=3000)
    parser.add_argument("--out", required=True, help="output .npy of counts, length %d" % VOCABULARY_SIZE)
    args = parser.parse_args()

    encode = load_sequence_encoder(args.kind, Path(args.tokenizer_path) if args.tokenizer_path else None)
    counts = np.zeros(VOCABULARY_SIZE, dtype=np.int64)
    bases = 0
    with open(args.shard) as handle:
        for line_number, line in enumerate(handle):
            if line_number >= args.num_lines:
                break
            sequence = line.strip().upper()
            if not sequence:
                continue
            ids, _ = encode(sequence)
            np.add.at(counts, np.asarray(ids), 1)
            bases += len(sequence)
    np.save(args.out, counts)
    used = int((counts > 0).sum())
    print(f"{args.kind}: {bases:,} bases -> {counts.sum():,} tokens; {used} of {VOCABULARY_SIZE} token IDs seen -> {args.out}")


if __name__ == "__main__":
    main()
