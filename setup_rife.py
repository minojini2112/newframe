"""Download Practical-RIFE repo + v4.25 inference weights (run once)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRACTICAL_RIFE = ROOT / "Practical-RIFE"
TRAIN_LOG = PRACTICAL_RIFE / "train_log"
REPO_URL = "https://github.com/hzwer/Practical-RIFE.git"
# v4.25 inference model (README line 21) — not the training-repo zip.
WEIGHTS_ID = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"


def rife_ready() -> bool:
    return (TRAIN_LOG / "flownet.pkl").exists() and (PRACTICAL_RIFE / "model").is_dir()


def clone_repo() -> None:
    if (PRACTICAL_RIFE / "model").is_dir():
        print(f"Repo present: {PRACTICAL_RIFE}")
        return
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(PRACTICAL_RIFE)],
        check=True,
    )


def download_weights() -> None:
    if (TRAIN_LOG / "flownet.pkl").exists() and (TRAIN_LOG / "RIFE_HDv3.py").exists():
        print(f"Weights present: {TRAIN_LOG / 'flownet.pkl'}")
        return

    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown

    TRAIN_LOG.mkdir(parents=True, exist_ok=True)
    zip_path = TRAIN_LOG / "v4.25_infer.zip"
    print("Downloading RIFE v4.25 weights (~23 MB) ...")
    gdown.download(
        f"https://drive.google.com/uc?id={WEIGHTS_ID}",
        str(zip_path),
        quiet=False,
    )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TRAIN_LOG)

    nested = TRAIN_LOG / "train_log"
    if nested.is_dir():
        for item in nested.iterdir():
            dest = TRAIN_LOG / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        shutil.rmtree(nested)

    zip_path.unlink(missing_ok=True)
    shutil.rmtree(TRAIN_LOG / "__MACOSX", ignore_errors=True)
    print(f"Installed: {TRAIN_LOG / 'flownet.pkl'}")


def verify() -> None:
    import torch

    sys.path.insert(0, str(PRACTICAL_RIFE))
    from train_log.RIFE_HDv3 import Model

    model = Model()
    sd = torch.load(TRAIN_LOG / "flownet.pkl", map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    wanted = model.flownet.state_dict()
    sd = {k: v for k, v in sd.items() if k in wanted}
    model.flownet.load_state_dict(sd, strict=True)
    print(f"Verify OK — RIFE v{getattr(model, 'version', '?')}")


def main() -> None:
    clone_repo()
    download_weights()
    if not rife_ready():
        raise SystemExit("RIFE setup incomplete — missing flownet.pkl or model/")
    verify()
    print("\nDone. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
