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
        self.distribution: dict[FeatureType, int] = {}

        self.attr: typing.SupportsIndex | None = None
        self.vote: LabelType | None = None


def train(train_file: str, max_depth: int) -> Node:
    data = np.genfromtxt(train_file, delimiter="\t", skip_header=1)
    return tree_recurse(data, 0, max_depth)


def all_columns_identical(dataset: np.ndarray[typing.Any, typing.Any]):
    # exclude label column
    features = dataset[:, :-1]
    return bool(np.all([len(np.unique(col)) == 1 for col in features.T]))  # type: ignore


def is_base_case(dataset: np.ndarray[typing.Any, typing.Any]) -> bool:
    empty_dataset = dataset.size == 0
    all_label_same = len(np.unique(dataset[:, -1])) == 1
    all_col_iden = all_columns_identical(dataset)
    return empty_dataset or all_label_same or all_col_iden


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


# data = np.array(
#     [
#         [1, 1, 2, 1, 1],
#         [1, 2, 2, 2, 2],
#         [1, 2, 2, 1, 1],
#         [1, 3, 3, 2, 2],
#         [1, 3, 3, 1, 2],
#         [1, 3, 1, 1, 1],
#         [2, 1, 3, 1, 3],
#         [2, 1, 1, 2, 2],
#         [2, 1, 1, 1, 1],
#         [2, 2, 3, 2, 2],
#         [2, 2, 2, 1, 1],
#         [2, 3, 3, 2, 3],
#         [2, 3, 3, 1, 3],
#         [2, 3, 2, 2, 2],
#         [2, 3, 2, 1, 1],
#         [2, 3, 1, 2, 2],
#     ]
# )

# mapping: dict[typing.SupportsIndex, dict[int, str]] = {
#     0: {1: "Rain", 2: "No Rain"},
#     1: {1: "Before", 2: "During", 3: "After"},
#     2: {2: "Both", 3: "Backpack", 1: "Lunchbox"},
#     3: {1: "Tired", 2: "Not Tired"},
#     4: {1: "Drive", 2: "Bus", 3: "Bike"},
# }


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
        dataset_given_v = dataset[dataset[:, feature_idx] == v]
        ent = entropy(dataset_given_v[:, -1])
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


def calculate_distribution(dataset: np.ndarray[typing.Any, typing.Any]):
    set_of_label: set[int] = set([0,1])
    label = dataset[:, -1]
    unique, counts = np.unique(label, return_counts=True)
    from collections import OrderedDict
    result = dict(zip(unique, counts))
    # print(f"{set_of_label=}, {set(result.keys())=}")
    for y in set_of_label - set(result.keys()):
        result[y] = 0
    # print(result)
    return OrderedDict(sorted(result.items()))


def tree_recurse(
    dataset: np.ndarray[typing.Any, typing.Any], cur_depth: int, max_depth: int
) -> Node:
    q = Node()
    # print(f"The shape of dataset is {dataset.shape}:{type(dataset.shape)}")
    if not is_base_case(dataset) and cur_depth < max_depth:
        q.attr = best_attribute(dataset)
        # print(f"best column: {q.attr}")
        attr_column = dataset[:, q.attr]
        v_x_d = np.unique(attr_column)
        for v in v_x_d:
            # print(f"Processing value {v} in column {attr_column}")
            D_v = dataset[attr_column == v]
            # edge = mapping[q.attr][v]
            # q.children[edge] = tree_recurse(D_v, cur_depth + 1, max_depth)
            q.children[v] = tree_recurse(D_v, cur_depth + 1, max_depth)
    q.vote = mode_of_label(dataset)
    q.distribution = calculate_distribution(dataset)
    # q.vote = mapping[4][mode_of_label(dataset)]
    return q


def predict(root: Node, x: np.ndarray[typing.Any, typing.Any]):
    current_node = root
    while True:
        if current_node.attr is not None:
            # internal node
            feature_val = x[current_node.attr]
            if feature_val in current_node.children:
                current_node = current_node.children[feature_val]
            else:
                return current_node.vote
        else:
            # leaf node
            return current_node.vote


def error_rate(
    predict: np.ndarray[typing.Any, typing.Any],
    label: np.ndarray[typing.Any, typing.Any],
) -> float:
    assert predict.shape == label.shape, (
        "Predictions and labels must have the same shape"
    )
    return float(1.0 - np.mean(predict == label))  # type: ignore


def predict_all(root: Node, filename: str):
    dataset = np.genfromtxt(filename, delimiter="\t", skip_header=1)
    features = dataset[:, :-1]
    result: list[LabelType] = []
    for row in features:
        result.append(predict(root, row))
    return result, error_rate(np.array(result), dataset[:, -1])


def generate_tree(
    node: Node,
    header: list[str],
    last_feature: typing.SupportsIndex | None,
    last_val: FeatureType | None,
    depth: int,
) -> typing.Generator[str, None, None]:
    def stringify_distribution(distribution: dict[FeatureType, int]):
        temp_list: list[str] = []
        for y, count in distribution.items():
            temp_list.append(f"{count} {int(y)}")
        output_string = f"[{'/'.join(temp_list)}]"
        return output_string

    if last_feature is None or last_val is None:
        # root
        output_string = stringify_distribution(node.distribution)
        yield output_string
    else:
        output_string = f"{'| ' * depth}{header[last_feature]} = {int(last_val)}: {stringify_distribution(node.distribution)}"
        yield output_string
    for feature_val, child in node.children.items():
        yield from generate_tree(child, header, node.attr, feature_val, depth + 1)


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

    result, train_err = predict_all(root, args.train_input)
    print(f"error(train): {train_err}")
    with open(args.train_out, "w") as fout:
        for label in result:
            fout.write(str(label))
            fout.write("\n")

    result, test_err = predict_all(root, args.test_input)
    print(f"error(test): {test_err}")
    with open(args.test_out, "w") as fout:
        for label in result:
            fout.write(str(label))
            fout.write("\n")

    with open(args.metrics_out, "w") as fout:
        fout.write(f"error(train): {train_err}\n")
        fout.write(f"error(test): {test_err}\n")

    with open(args.print_out, "w") as fout:
        for s in generate_tree(root, list(np.genfromtxt(args.train_input, dtype=str, max_rows=1)), None, None, 0):
            fout.write(s)
            fout.write("\n")
