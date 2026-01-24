import importlib

PKGS = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "yaml",
    "networkx",
    "xgboost",
    "lightgbm",
]

def main():
    missing = []
    for p in PKGS:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)

    if missing:
        raise SystemExit(f"Missing packages: {missing}")

    print("Environment looks good. All core packages import successfully.")

if __name__ == "__main__":
    main()