"""Download the processed Cleveland Heart Disease dataset (public UCI-derived copy)."""
from pathlib import Path
import pandas as pd
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
DATA_DIR.mkdir(exist_ok=True)
URL = "https://raw.githubusercontent.com/saratambal/UCI-Heart-Disease-Analysis/master/data/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def main():
    destination = DATA_DIR / "heart.csv"
    print("Downloading Cleveland Heart Disease data...")
    request = urllib.request.Request(URL, headers={"User-Agent": "AIA-Heart-Disease-Project/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")
    destination.write_text(",".join(COLUMNS) + "\n" + content.replace("?", ""), encoding="utf-8")
    df = pd.read_csv(destination)
    print(f"Saved {len(df)} rows to {destination}")
    print("Note: target is original diagnosis 0–4; training code converts 0 to 0 and 1–4 to 1.")

if __name__ == "__main__":
    main()
