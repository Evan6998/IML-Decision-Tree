import argparse
import typing
import numpy as np
# import matplotlib.pyplot as plt

type FeatureType = typing.Any
type LabelType = typing.Any


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree.
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired
    """

    def __init__(self):
        # self.left: Node | None = None
        # self.right: Node | None = None
        self.children: dict[FeatureType, Node] = {}

        self.attr: typing.SupportsIndex | None = None
        self.vote: LabelType | None = None


def train(train_file: str, max_depth: int) -> Node:
    data = np.genfromtxt(train_file, delimiter="\t", skip_header=1)
    return tree_recurse(data, 0, max_depth)


def all_columns_identical(dataset: np.ndarray[typing.Any, typing.Any]):
    # exclude label column
    features = dataset[:, :-1]
    return bool(np.all(features == features[:, [0]]))


def is_base_case(dataset: np.ndarray[typing.Any, typing.Any]) -> bool:
    empty_dataset = dataset.size == 0
    all_label_same = len(np.unique(dataset[:, -1])) == 1
    return empty_dataset or all_label_same or all_columns_identical(dataset)


def mode_of_label(dataset: np.ndarray[typing.Any, typing.Any]) -> LabelType:
    labels = dataset[:, -1]
    values, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    candidates = values[counts == max_count]
    return max(candidates)  # break ties by choosing the larger value


def entropy(label: np.ndarray[typing.Any, typing.Any]):
    unique, counts = np.unique(label, return_counts=True)
    categories = dict(zip(unique, counts))
    total_number = sum(categories.values())
    entropy: float = 0
    for _, count in categories.items():
        prob = count / total_number
        entropy += -1 * prob * np.log2(prob)
    return entropy


data = np.array(
    [
        [1, 1, 2, 1, 1],
        [1, 2, 2, 2, 2],
        [1, 2, 2, 1, 1],
        [1, 3, 3, 2, 2],
        [1, 3, 3, 1, 2],
        [1, 3, 1, 1, 1],
        [2, 1, 3, 1, 3],
        [2, 1, 1, 2, 2],
        [2, 1, 1, 1, 1],
        [2, 2, 3, 2, 2],
        [2, 2, 2, 1, 1],
        [2, 3, 3, 2, 3],
        [2, 3, 3, 1, 3],
        [2, 3, 2, 2, 2],
        [2, 3, 2, 1, 1],
        [2, 3, 1, 2, 2],
    ]
)

mapping: dict[typing.SupportsIndex, dict[int, str]] = {
    0: {1: "Rain", 2: "No Rain"},
    1: {1: "Before", 2: "During", 3: "After"},
    2: {2: "Both", 3: "Backpack", 1: "Lunchbox"},
    3: {1: "Tired", 2: "Not Tired"},
    4: {1: "Drive", 2: "Bus", 3: "Bike"},
}


def mutual_information(
    dataset: np.ndarray[typing.Any, typing.Any], feature_idx: typing.SupportsIndex
):
    label = dataset[:, -1]
    h_label = entropy(label)
    x_d = dataset[:, feature_idx]
    unique, counts = np.unique(x_d, return_counts=True)
    feature_values = dict(zip(unique, counts))

    h_label_given_feature: float = 0
    for v, count in feature_values.items():
        print(v, count)
        dataset_given_v = dataset[dataset[:, feature_idx] == v]
        print(f"{dataset_given_v=}")
        ent = entropy(dataset_given_v[:, -1])
        print(f"{ent=}")
        h_label_given_feature += count / dataset.shape[0] * ent

    i_xy = h_label - h_label_given_feature
    return i_xy


def best_attribute(dataset: np.ndarray[typing.Any, typing.Any]) -> typing.SupportsIndex:
    best_mi = 0
    best_idx = 0
    for idx in range(dataset.shape[1] - 1):
        if (mi := mutual_information(dataset, idx)) > best_mi:
            best_mi = mi
            best_idx = idx
    return best_idx


def tree_recurse(dataset: np.ndarray[typing.Any, typing.Any], cur_depth: int, max_depth: int) -> Node:
    q = Node()
    print(f"The shape of dataset is {dataset.shape}:{type(dataset.shape)}")
    if not is_base_case(dataset) and cur_depth < max_depth:
        q.attr = best_attribute(dataset)
        print(f"best column: {q.attr}")
        attr_column = dataset[:, q.attr]
        v_x_d = np.unique(attr_column)
        for v in v_x_d:
            print(f"Processing value {v} in column {attr_column}")
            D_v = dataset[attr_column == v]
            edge = mapping[q.attr][v]
            q.children[edge] = tree_recurse(D_v, cur_depth + 1, max_depth)
            # q.children[v] = tree_recurse(D_v)
    # q.vote = mode_of_label(dataset)
    q.vote = mapping[4][mode_of_label(dataset)]
    return q


def predict(): ...


def print_tree(node: Node):
    pass


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_input", type=str, help="path to training input .tsv file"
    )
    parser.add_argument("test_input", type=str, help="path to the test input .tsv file")
    parser.add_argument(
        "max_depth", type=int, help="maximum depth to which the tree should be built"
    )
    parser.add_argument(
        "train_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the training data should be written",
    )
    parser.add_argument(
        "test_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the test data should be written",
    )
    parser.add_argument(
        "metrics_out",
        type=str,
        help="path of the output .txt file to which metrics such as train and test error should be written",
    )
    parser.add_argument(
        "print_out",
        type=str,
        help="path of the output .txt file to which the printed tree should be written",
    )
    args = parser.parse_args()

    # Here's an example of how to use argparse
    print_out = args.print_out

    # Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)

    root = train(args.train_input, args.max_depth)
