"""
Creates dataset.yaml inside /kaggle/working/.

Why this is a separate script: /kaggle/working/ gets wiped every time the
Kaggle session restarts, so this file needs to be re-created each session -
it can't just live as a static file the way it does in this repo. Run this
AFTER train_val_split.py and BEFORE train.py.
"""

OUTPUT_PATH = "/kaggle/working/dataset.yaml"

yaml_content = """\
path: /kaggle/working/dataset
train: images/train
val: images/val

nc: 1
names: ["pneumonia"]
"""


def main():
    with open(OUTPUT_PATH, "w") as f:
        f.write(yaml_content)
    print("created:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
