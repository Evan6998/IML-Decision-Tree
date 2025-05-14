import argparse
import numpy as np
import typing


def entropy(categories: dict[typing.Any, int]):
    total_number = sum(categories.values())
    entropy = 0
    for _, count in categories.items():
        prob = count / total_number
        entropy += -1 * prob * np.log2(prob)
    return entropy


def error(categories: dict[typing.Any, int]):
    total_number = sum(categories.values())
    mode_occurance = max(categories.values())
    return (total_number - mode_occurance) / total_number


def inspect(filename: str, output: str):
    data = np.genfromtxt(filename, delimiter="\t", skip_header=1)
    label = data[:, -1]
    unique, counts = np.unique(label, return_counts=True)
    categories = dict(zip(unique, counts))
    with open(output, 'w') as fout:
        fout.write(f"entropy: {entropy(categories)}\n")
        fout.write(f"error: {error(categories)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="path to the input .tsv file")
    parser.add_argument("output", type=str, help="path to the output file")
    args = parser.parse_args()

    inspect(args.input, args.output)
