#!/usr/bin/env python3
import argparse
import json
import random

def main():
    parser = argparse.ArgumentParser(description="Split a single JSON Lines manifest into train and eval sets.")
    parser.add_argument("--manifest_in", type=str, required=True,
                        help="Path to the input JSON Lines manifest (e.g. preprocessed_manifest.json).")
    parser.add_argument("--train_out", type=str, required=True,
                        help="Path to the output training JSON Lines file.")
    parser.add_argument("--eval_out", type=str, required=True,
                        help="Path to the output evaluation JSON Lines file.")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="Fraction of data to use for training (default=0.9). The rest is for eval.")
    args = parser.parse_args()

    # Read all lines from the input manifest
    with open(args.manifest_in, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle to randomize which samples go into train vs. eval
    random.shuffle(lines)

    # Compute the split index
    split_index = int(len(lines) * args.split_ratio)

    train_lines = lines[:split_index]
    eval_lines = lines[split_index:]

    # Write the training split
    with open(args.train_out, 'w', encoding='utf-8') as f_train:
        for line in train_lines:
            f_train.write(line)

    # Write the evaluation split
    with open(args.eval_out, 'w', encoding='utf-8') as f_eval:
        for line in eval_lines:
            f_eval.write(line)

    print(f"Total entries: {len(lines)}")
    print(f"Training manifest: {len(train_lines)} entries -> {args.train_out}")
    print(f"Evaluation manifest: {len(eval_lines)} entries -> {args.eval_out}")

if __name__ == "__main__":
    main()
