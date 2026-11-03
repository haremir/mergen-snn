import sys

def test_source(name, fn):
    print(f"\n--- Testing {name} ---")
    try:
        fn()
        print(f"SUCCESS: {name}")
    except Exception as e:
        print(f"FAILED {name}: {type(e).__name__}: {e}")

from datasets import load_dataset

# 1. The Stack
def test_stack():
    try:
        ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
        sample = next(iter(ds))
        print("The Stack v2 sample keys:", list(sample.keys()))
    except Exception as e:
        print("the-stack-v2 failed, testing the-stack-smol or alternative...")
        # fallback alternative for code
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", streaming=True, split="train")
        sample = next(iter(ds))
        print("the-stack-smol sample keys:", list(sample.keys()))

# 2. CulturaX tr
def test_culturax():
    ds = load_dataset("uonlp/CulturaX", "tr", streaming=True, split="train")
    sample = next(iter(ds))
    print("CulturaX sample keys:", list(sample.keys()))

# 3. S2ORC / ML-ArXiv-Papers
def test_s2orc():
    try:
        ds = load_dataset("allenai/s2orc", streaming=True, split="train")
        sample = next(iter(ds))
        print("s2orc sample keys:", list(sample.keys()))
    except Exception as e:
        print("s2orc failed, trying ML-ArXiv-Papers...")
        ds = load_dataset("CShorten/ML-ArXiv-Papers", streaming=True, split="train")
        sample = next(iter(ds))
        print("ML-ArXiv-Papers sample keys:", list(sample.keys()))

# 4. FineWeb
def test_fineweb():
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", streaming=True, split="train")
    sample = next(iter(ds))
    print("FineWeb sample keys:", list(sample.keys()))

# 5. Local txt
def test_local():
    from pathlib import Path
    raw_dir = Path("data/raw/cs_corpus")
    txt_files = list(raw_dir.rglob("*.txt"))
    total_sz = sum(f.stat().st_size for f in txt_files)
    print(f"Local txt files count: {len(txt_files)}, total size: {total_sz / (1024*1024):.2f} MB")

test_source("Local TXT", test_local)
test_source("FineWeb", test_fineweb)
test_source("CulturaX", test_culturax)
test_source("S2ORC", test_s2orc)
test_source("The Stack", test_stack)
