"""Fetch and curate CS corpus for Mergen RAG.

Sources:
1. Software Engineering Handbook (github.com/maniebra/software-engineering-handbook) - markdown
2. OpenLib.cs (github.com/benjaminshawki/openlib.cs) - CS lecture notes
3. Papers We Love (github.com/papers-we-love/papers-we-love) - programming_languages & distributed_systems
4. Local CS corpus (data/raw/cs_corpus) - Turkish and fundamental CS texts
Target: ~50 MB clean CS text in data/rag_corpus/
"""

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


def remove_readonly(func, path, exc_info):
    """Handle read-only git pack files on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def clone_and_extract(repo_url: str, dest_dir: Path, subpath: str = "", exts=(".md", ".txt")):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "repo"
        print(f"Cloning {repo_url} (depth=1)...")
        res = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(tmp_path)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"Warning: git clone failed for {repo_url}: {res.stderr}")
            return 0, 0

        search_root = tmp_path / subpath if subpath else tmp_path
        copied = 0
        total_bytes = 0

        for ext in exts:
            for f in search_root.rglob(f"*{ext}"):
                if f.is_file():
                    # Preserve relative path structure
                    try:
                        rel = f.relative_to(search_root)
                    except ValueError:
                        rel = f.name
                    target = dest_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(f, target)
                        copied += 1
                        total_bytes += target.stat().st_size
                    except Exception as e:
                        print(f"Failed to copy {f}: {e}")

        # Clean git read-only files before context exit
        for root, dirs, files in os.walk(tmp_path):
            for file in files:
                try:
                    os.chmod(os.path.join(root, file), stat.S_IWRITE)
                except Exception:
                    pass

        return copied, total_bytes


def main():
    rag_dir = Path("data/rag_corpus")
    rag_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("Mergen-SNN — RAG Corpus Fetcher")
    print(f"Target Directory: {rag_dir.resolve()}")
    print("==========================================")

    stats = {}

    # 1. Software Engineering Handbook
    seh_dir = rag_dir / "se_handbook"
    if not seh_dir.exists() or not any(seh_dir.iterdir()):
        cnt, b = clone_and_extract(
            "https://github.com/maniebra/software-engineering-handbook.git",
            seh_dir,
            exts=(".md", ".txt"),
        )
        stats["Software Engineering Handbook"] = (cnt, b)
    else:
        files = [f for f in seh_dir.rglob("*") if f.is_file()]
        stats["Software Engineering Handbook"] = (len(files), sum(f.stat().st_size for f in files))

    # 2. OpenLib.cs
    openlib_dir = rag_dir / "openlib_cs"
    if not openlib_dir.exists() or not any(openlib_dir.iterdir()):
        cnt, b = clone_and_extract(
            "https://github.com/benjaminshawki/openlib.cs.git",
            openlib_dir,
            exts=(".md", ".txt"),
        )
        stats["OpenLib.cs"] = (cnt, b)
    else:
        files = [f for f in openlib_dir.rglob("*") if f.is_file()]
        stats["OpenLib.cs"] = (len(files), sum(f.stat().st_size for f in files))

    # 3. Papers We Love (programming_languages & distributed_systems)
    pwl_dir = rag_dir / "papers_we_love"
    if not pwl_dir.exists() or not any(pwl_dir.iterdir()):
        total_cnt = 0
        total_b = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "pwl"
            print("Cloning Papers We Love (depth=1)...")
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/papers-we-love/papers-we-love.git", str(tmp_path)],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                for cat in ["programming_languages", "distributed_systems"]:
                    cat_dir = tmp_path / cat
                    cat_dest = pwl_dir / cat
                    cat_dest.mkdir(parents=True, exist_ok=True)
                    if cat_dir.exists():
                        for f in cat_dir.rglob("*"):
                            if f.is_file() and f.suffix.lower() in [".md", ".txt"]:
                                rel = f.relative_to(cat_dir)
                                target = cat_dest / rel
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(f, target)
                                total_cnt += 1
                                total_b += target.stat().st_size

            for root, dirs, files in os.walk(tmp_path):
                for file in files:
                    try:
                        os.chmod(os.path.join(root, file), stat.S_IWRITE)
                    except Exception:
                        pass

        stats["Papers We Love (PL & DS)"] = (total_cnt, total_b)
    else:
        files = [f for f in pwl_dir.rglob("*") if f.is_file()]
        stats["Papers We Love (PL & DS)"] = (len(files), sum(f.stat().st_size for f in files))

    # Calculate current size
    curr_bytes = sum(b for _, b in stats.values())
    target_bytes = 50 * 1024 * 1024  # 50 MB
    needed_bytes = max(0, target_bytes - curr_bytes)

    # 4. Local CS corpus (data/raw/cs_corpus)
    local_cs_src = Path("data/raw/cs_corpus")
    local_rag_dir = rag_dir / "local_cs"
    local_rag_dir.mkdir(parents=True, exist_ok=True)

    copied_local = 0
    local_bytes = 0

    if local_cs_src.exists():
        # Priority subdirectories
        priority_dirs = [
            "tr_docs",
            "algorithms",
            "theory",
            "systems",
            "lecture_notes",
            "ostep",
            "databases",
            "compilers",
            "architecture",
            "code_docs",
            "python_docs",
            "cpp_docs",
            "c_docs",
            "networking",
            "unix_os",
            "ai_ml",
        ]
        
        # Collect candidate files
        candidate_files = []
        for pdir in priority_dirs:
            d = local_cs_src / pdir
            if d.exists():
                candidate_files.extend(list(d.rglob("*.txt")))

        # Add remaining if any
        all_txt = list(local_cs_src.rglob("*.txt"))
        for f in all_txt:
            if f not in candidate_files:
                candidate_files.append(f)

        print(f"Adding local CS documents up to ~50 MB target (need ~{needed_bytes / (1024*1024):.1f} MB)...")
        for f in candidate_files:
            if local_bytes >= needed_bytes and needed_bytes > 0:
                break
            try:
                rel = f.relative_to(local_cs_src)
                target = local_rag_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied_local += 1
                local_bytes += target.stat().st_size
            except Exception as e:
                pass

    stats["Local CS Corpus"] = (copied_local, local_bytes)

    # Summary
    print("\n=== RAG Corpus Toplama Özeti ===")
    total_files = 0
    total_size = 0
    for name, (cnt, sz) in stats.items():
        print(f"- {name}: {cnt} dosya ({sz / (1024*1024):.2f} MB)")
        total_files += cnt
        total_size += sz

    print("------------------------------------------")
    print(f"TOPLAM: {total_files} dosya, {total_size / (1024*1024):.2f} MB")
    print("==========================================")


if __name__ == "__main__":
    main()
