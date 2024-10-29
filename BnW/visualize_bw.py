import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

CMAP = mpl.colors.ListedColormap(
    [
        "white",
        "cornflowerblue",
        "tomato",
        "lime",
        "yellow",
        "lightgrey",
        "magenta",
        "orange",
        "aqua",
        "maroon",
        "black",
    ]
)
NORM = mpl.colors.BoundaryNorm(np.arange(-0.5, CMAP.N), CMAP.N)


def create_parser():
    # create an argparser for training or evaluation
    parser = argparse.ArgumentParser(description="Visualize the black and white images")
    parser.add_argument(
        "--type",
        type=str,
        help="Path to the image",
        required=True,
        choices=["training", "evaluation"],
    )
    return parser


def plot_single_image(filepath: Path, cmap, norm, iter, skip):
    with open(filepath, "r") as f:
        data = json.load(f)

    # see how many train inputs there are
    train_inputs = len(data["train"])
    # see how many test inputs there are
    test_inputs = len(data["test"])

    the_key = None

    def press(event):
        nonlocal the_key
        the_key = event.key

    # create a figure with 2 subplots for each input
    fig, axs = plt.subplots(2, train_inputs + test_inputs, figsize=(20, 20))
    fig.tight_layout()

    # plot each training input
    for i, train_input in enumerate(data["train"]):
        x = np.array(train_input["input"])
        axs[0, i].pcolor(x, edgecolors="darkgrey", linewidth=2, cmap=cmap, norm=norm)
        axs[0, i].axis("off")
        axs[0, i].set_title(f"Input {i}")
        axs[0, i].set_aspect("equal")

        x = np.array(train_input["output"])
        axs[1, i].pcolor(x, edgecolors="darkgrey", linewidth=2, cmap=cmap, norm=norm)
        axs[1, i].axis("off")
        axs[1, i].set_title(f"Output {i}")
        axs[1, i].set_aspect("equal")

    # plot each test input
    for i, test_input in enumerate(data["test"]):
        x = np.array(test_input["input"])
        x[x > 0] = 10
        axs[0, i + train_inputs].pcolor(
            x, edgecolors="darkgrey", linewidth=2, cmap=cmap, norm=norm
        )
        axs[0, i + train_inputs].axis("off")
        axs[0, i + train_inputs].set_title(f"Test Input {i}")
        axs[0, i + train_inputs].set_aspect("equal")

        x = np.array(test_input["output"])
        x[x > 0] = 10
        axs[1, i + train_inputs].pcolor(
            x, edgecolors="darkgrey", linewidth=2, cmap=cmap, norm=norm
        )
        axs[1, i + train_inputs].axis("off")
        axs[1, i + train_inputs].set_title(f"Test Output {i}")
        axs[1, i + train_inputs].set_aspect("equal")

    # add title of filename
    fig.suptitle(f"{filepath.name} - {iter+skip}")

    plt.gcf().canvas.mpl_connect("key_press_event", press)
    mng = plt.get_current_fig_manager()
    mng.resize(3000, 1664)
    # plt.show()
    while not plt.waitforbuttonpress():
        pass

    plt.close()

    save_file(filepath, data, the_key)


def save_file(filepath, data, the_key):
    if the_key == "y":
        new_filepath = filepath.parent.parent.parent / "b&w" / filepath.parent.stem / "yes" / filepath.name
        with open(new_filepath, "w") as f:
            json.dump(data, f)

    elif the_key == "n":
        new_filepath = filepath.parent.parent.parent / "b&w" / filepath.parent.stem / "no" / filepath.name
        with open(new_filepath, "w") as f:
            json.dump(data, f)


def get_files(data_path):
    files = sorted(data_path.glob("*.json"))
    max_len = len(files)

    # make the directories needed
    yes_dir = data_path.parent.parent / "b&w" / data_path.stem / "yes"
    no_dir = data_path.parent.parent / "b&w" / data_path.stem / "no"
    yes_dir.mkdir(parents=True, exist_ok=True)
    no_dir.mkdir(parents=True, exist_ok=True)

    already_parsed_files = list(yes_dir.glob("*.json"))
    already_parsed_files += list(no_dir.glob("*.json"))
    already_parsed_files = sorted(file.name for file in already_parsed_files)

    last_done_file = already_parsed_files[-1] if already_parsed_files else None

    if last_done_file:
        last_done_file_idx = [file.name for file in files].index(last_done_file)
        files = files[last_done_file_idx + 1 :]

    return files, max_len


def main(args):
    data_path = Path("../data") / args.type

    assert data_path.exists(), f"{data_path} does not exist"

    files, max_files = get_files(data_path)
    skip = max_files - len(files)
    print(f"Starting at file {files[0]}")

    for i, file in enumerate(files):
        plot_single_image(file, CMAP, NORM, i, skip)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
