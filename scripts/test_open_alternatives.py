from datasets import load_dataset

def test(name, fn):
    try:
        fn()
        print(f"SUCCESS: {name}")
        return True
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False

# Turkish open datasets
print("--- Testing Turkish Open Datasets ---")
test("wikipedia-tr", lambda: print("wiki:", next(iter(load_dataset("wikimedia/wikipedia", "20231101.tr", streaming=True, split="train")))["text"][:60]))
test("turkish_news", lambda: print("news:", next(iter(load_dataset("turkish-nlp-suite/turkish-news-corpus", streaming=True, split="train")))["text"][:60]))

# Code open datasets
print("\n--- Testing Code Open Datasets ---")
test("codeparrot-clean", lambda: print("codeparrot:", next(iter(load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")))["content"][:60]))
test("python-code", lambda: print("code_search_net:", next(iter(load_dataset("code_search_net", "python", streaming=True, split="train")))["func_code_string"][:60]))
test("c-cpp-code", lambda: print("code_search_net c:", next(iter(load_dataset("code_search_net", "c", streaming=True, split="train")))["func_code_string"][:60]))
