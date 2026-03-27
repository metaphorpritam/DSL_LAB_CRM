import kagglehub
import shutil
import os

DEST = os.path.join(os.path.dirname(__file__), "..", "Data")


def download():
    print("Downloading GiveMeSomeCredit dataset...")
    path = kagglehub.competition_download("GiveMeSomeCredit")
    print(f"Cached at: {path}")

    os.makedirs(DEST, exist_ok=True)
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(DEST, item)
        shutil.copy2(src, dst)
        print(f"  -> {item}")

    print(f"Done. Files saved to: {os.path.abspath(DEST)}")


if __name__ == "__main__":
    download()
