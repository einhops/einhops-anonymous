import math
import torch
from tqdm import tqdm

from .backend import SLOT_COUNT, fhe_encrypt, fhe_mul_plain, fhe_hoisted_rotate, fhe_add, fhe_rotate, fhe_rotate_fixed

def select_bsgs_factors(n: int) -> tuple[int, int]:
    """
    """
    power = int(math.log2(n))
    if power % 2 == 0:
        N1 = int(math.sqrt(n))
    else:
        N1 = int(math.sqrt(2 * n))
    N2 = n // N1
    assert N1 * N2 == n
    return (N1, N2)

def bsgs_matrix_mult(matrix, ciphertext, debug=False):
    N1, N2 = select_bsgs_factors(SLOT_COUNT)
    idxs = torch.arange(N1).unsqueeze(1) + torch.arange(SLOT_COUNT).unsqueeze(0)  # (N1, slots)
    idxs = idxs % SLOT_COUNT

    set_rots_needed = set()
    for j in range(N2):
        rows = torch.arange(SLOT_COUNT).unsqueeze(0).expand(N1, SLOT_COUNT)  # (N1, slots)
        diags = matrix[rows, idxs]                            # (N1, slots)
        idxs = (idxs + N1) % SLOT_COUNT

        # find which rotations actually contribute
        nz_rows = (diags.sum(dim=1) != 0).nonzero(as_tuple=False).squeeze(1)
        set_rots_needed.update(nz_rows.tolist())
    
    # hoisted rotations
    set_rots = list(sorted([i for i in set_rots_needed]))
    rotated_inputs = fhe_hoisted_rotate(ciphertext, set_rots)
    rotated_inputs_map = {rot_amount: rotated_inputs[i] for i, rot_amount in enumerate(set_rots)}

    result = fhe_encrypt(torch.zeros(SLOT_COUNT))
    for j in tqdm(range(N2), disable=not debug):
        diags = matrix[range(SLOT_COUNT), idxs]

        non_zero_diags = (diags.sum(dim=1) != 0).nonzero(as_tuple=False).squeeze(1)
        if non_zero_diags.numel() > 0:
            curr_block = fhe_encrypt(torch.zeros(SLOT_COUNT))
            for i in non_zero_diags.tolist():
                diag_i = torch.roll(diags[i], shifts=N1 * j)
                prod = fhe_mul_plain(rotated_inputs_map[i], diag_i)
                curr_block = fhe_add(curr_block, prod)
            rotated_block = fhe_rotate_fixed(curr_block, -N1*j)
            result = fhe_add(result, rotated_block)

        idxs += N1
        idxs %= SLOT_COUNT

    return result
