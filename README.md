# EinHops: Einsum Notation for Expressive Homomorphic Operations on RNS-CKKS Tensors

## Installation
We tested `Einhops` on an Intel(R) Xeon(R) Platinum
8480+ processor (26 threads, 13 cores, 2 threads per core) that is
equipped with a single NVIDIA H100 GPU. We use `Python` version `3.11.13` and require roughly 30GB of RAM (or VRAM if you run `EinHops` on a GPU).
```bash
>>> python --version
Python 3.11.13
```

Clone the repo, change directories, and install the required python packages:
```bash
>>> pip install -r requirements.txt
```

If you have an NVIDIA GPU with at least 30GB of VRAM, you may run `EinHops` on your GPU. Check your CUDA version using `nvidia-smi` (upper right corner). Install the corresponding [`desilo` library](https://fhe.desilo.dev/latest/install/). In our case, we have CUDA version `12.8` so we run:
```bash
 pip install desilofhe-cu128
 ```
If you do not have an NVIDIA GPU with at least 30GB of RAM, please run:
```bash
pip install desilofhe
```
Additionally, change the thread count to your preference in `crypto/backend.py`.

At the root level of the this repository, run:
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Running `EinHops`
Run Listing1 from the submission:
```bash
python listing1.py
```

For our main results, run each benchmark in the `benchmarks` directory.
