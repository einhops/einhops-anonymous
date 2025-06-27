import torch
import einhops

import timeit

def torch_add():
    a = torch.randn(2,3)
    b = torch.randn(2,3)
    return a + b

def einhops_add():
    a = torch.randn(2,3)
    b = torch.randn(2,3)
    a_ckks = einhops.encrypt(a)
    b_ckks = einhops.encrypt(b)
    return einhops.decrypt(a_ckks + b_ckks)

if __name__ == "__main__":
    print("Timing PyTorch vs Einhops operations...")
    print("=" * 50)

    # Test addition
    torch_time = timeit.timeit(torch_add, number=100)
    print(f"PyTorch addition (100 runs): {torch_time:.4f}s ({torch_time*10:.2f}ms per run)")

    einhops_time = timeit.timeit(einhops_add, number=10)  # Fewer runs since FHE is slow
    print(f"Einhops addition (10 runs):  {einhops_time:.4f}s ({einhops_time*100:.0f}ms per run)")
    print(f"Slowdown factor: {(einhops_time/10) / (torch_time/100):.0f}x")
    print()
