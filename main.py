import kagglehub
import shutil
import os


def main():
    print("Downloading GiveMeSomeCredit dataset from Kaggle...")
    path = kagglehub.competition_download('GiveMeSomeCredit')
    print(f"Downloaded to: {path}")

    dest = os.path.join(os.path.dirname(__file__), "Data")
    os.makedirs(dest, exist_ok=True)

    for item in os.listdir(path):
        src = os.path.join(path, item)
        shutil.copy2(src, os.path.join(dest, item))
        print(f"Copied: {item}")

    print(f"Dataset ready in: {dest}")


if __name__ == "__main__":
    main()
