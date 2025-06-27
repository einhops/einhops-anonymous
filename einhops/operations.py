import math
import torch
from tqdm import tqdm

from .crypto.backend import SLOT_COUNT, fhe_mul_plain, fhe_encode
from .crypto.bsgs import bsgs_matrix_mult
from .crypto.ckkstensor import CKKSTensor, tensor_to_packed_vector, tensor_to_padded_tensor, tensor_to_ptxt

def BSGS(matrix, cipher):
    # get properties of ciphertext
    shape = cipher.shape
    fhe_shape = cipher.fhe_shape
    ndim = cipher.ndim

    out = bsgs_matrix_mult(matrix, cipher.slots)
    return CKKSTensor(out, shape, fhe_shape, ndim)

def expand_tensor(operand, src_dim, dst_dim, dim_sizes):
    if src_dim == dst_dim:
        return operand
    
    original_order_label = list(src_dim)
    original_order_index = list(range(len(src_dim)))
    new_order_label = list(filter(lambda dim: dim in src_dim, dst_dim))
    order_map = {elem: idx for idx, elem in enumerate(new_order_label)}
    new_order_index = sorted(original_order_index, key=lambda i: order_map[original_order_label[i]])

    operand = operand.permute(*new_order_index)
    new_shape = [1] * len(dst_dim)
    for i, dim in enumerate(src_dim):
        new_shape[dst_dim.index(dim)] = dim_sizes[dim]

    operand = operand.view(*new_shape)
    operand = torch.broadcast_to(operand, [dim_sizes[dim] for dim in dst_dim])
    return tensor_to_ptxt(operand)


def expand_dimensions(operand, src_dim, dst_dim, dim_sizes, fhe_dim_sizes):
    if isinstance(operand, torch.Tensor):
        return expand_tensor(operand, src_dim, dst_dim, dim_sizes)

     # no expansion needed
    if src_dim == dst_dim:
        return operand

    # get the stride of each dim in the src tensor
    src_shape = [fhe_dim_sizes[dim] for dim in src_dim]
    src_stride = torch.tensor(torch.empty(*src_shape).stride())

    # get the stride of each dim in the dst tensor
    dst_shape = [fhe_dim_sizes[dim] for dim in dst_dim]
    dst_stride = torch.tensor(torch.empty(*dst_shape).stride())

    # map src dimensions to their new stride in dst
    src_to_dst_indices = [dst_dim.index(d) for d in src_dim]
    new_stride = dst_stride[src_to_dst_indices]

    # get flattened indices of src tensor
    src_tensor_idxs = torch.cartesian_prod(*[torch.arange(fhe_dim_sizes[dim]) for dim in src_dim])
    if src_tensor_idxs.ndim == 1: # handling 1-d arrays
        src_tensor_idxs = src_tensor_idxs.unsqueeze(1)
    src_idxs = torch.sum(src_tensor_idxs * src_stride, dim=1)
    new_idxs = torch.sum(src_tensor_idxs * new_stride, dim=1)

    # create linear transformation to re-arrange the src to the expanded dst tensor
    if not torch.equal(new_idxs, src_idxs):
        T = torch.zeros((SLOT_COUNT, SLOT_COUNT), dtype=torch.float32)
        T[new_idxs, src_idxs] = 1

        # perform linear transform
        out = BSGS(T, operand)
    else:
        out = operand

    # replicate the missing dimensions
    for dim in reversed(dst_dim):
        if dim not in src_dim:
            stride = dst_stride[dst_dim.index(dim)].item()
            num_rots = int(math.log2(fhe_dim_sizes[dim])) 
            for rep in range(num_rots):
                rotated = out.rotate(stride)
                out = out + rotated
                stride *= 2

    # set new shapes 
    out_shape = torch.Size([dim_sizes[dim] for dim in dst_dim])
    out_fhe_shape = torch.Size([fhe_dim_sizes[dim] for dim in dst_dim])
    out_ndim = len(out_fhe_shape)

    out.shape = out_shape
    out.fhe_shape = out_fhe_shape
    out.ndim = out_ndim
    return out

def gather_slots(cipher, o_dims, dim_sizes, fhe_dim_sizes):
    mask = tensor_to_packed_vector(torch.ones(cipher.shape, dtype=torch.float32))
    out = fhe_mul_plain(cipher.slots, mask)

    out_shape = torch.Size([dim_sizes[dim] for dim in o_dims])
    fhe_out_shape = torch.Size([fhe_dim_sizes[dim] for dim in o_dims])
    out_ndim = len(fhe_out_shape)

    return CKKSTensor(out, out_shape, fhe_out_shape, out_ndim)
