from pathlib import Path
import sys
from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

BASE_DIR = Path(__file__).resolve().parent
pyx_path = str(BASE_DIR / "fast_lif.pyx")

extra_compile_args = []
extra_link_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/fp:fast", "/openmp"]
else:
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        name="mergen.cython_ext.fast_lif",
        sources=[pyx_path],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="fast_lif",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
    package_dir={"": "src"},
)
