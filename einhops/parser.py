import math
import torch
import opt_einsum

def parse_dims(equation, *args):
    """
    Parses the dimensions of the inputs using opt_einsum
    Returns the input dimensions, output dimensions, and reduction dimensions.

    >>> import torch
    >>> i_dims, o_dims, r_dims = parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4))
    >>> i_dims
    ['ij', 'jk']
    >>> ''.join(sorted(o_dims))
    'ik'
    >>> ''.join(sorted(r_dims))
    'j'
    """
    input_subs, o_dims, _ = opt_einsum.parser.parse_einsum_input((equation, *args))
    i_dims = input_subs.strip().split(",")
    r_dims = "".join([d for d in set("".join(i_dims)) if d not in o_dims])
    return i_dims, o_dims, r_dims

def get_dim_sizes(input_dims, *args):
    """
    Extracts the real tensor dimension from all inputs.
    For parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4)), 
    dim_sizes = {
        'i': 2,
        'j': 3,
        'k': 4
    }
    """
    dim_sizes = {}
    for arg, input_dim in zip(args, input_dims):
        for i, dim in enumerate(input_dim):
            dim_sizes[dim] = arg.shape[i]
    return dim_sizes

def next_power_of_two(n):
    """
    Return the smallest power of two >= n.
    >>> next_power_of_two(9)
    16
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << (math.ceil(math.log2(n)))

def get_fhe_dim_sizes(input_dims, *args):
    """
    Extracts the FHE tensor dimension from all inputs.
    For Einhops, all FHE tensors have power-of-two dimensions.
    For parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4)), 
    dim_sizes = {
        'i': 2,
        'j': 4,
        'k': 4
    }
    """
    fhe_dim_sizes = {}
    for arg, input_dim in zip(args, input_dims):
        # if not hasattr(arg, 'fhe_shape'):
        #     raise ValueError(f"Argument must be CKKSTensor with fhe_shape attribute")
        for i, dim in enumerate(input_dim):
            fhe_dim_sizes[dim] = next_power_of_two(arg.shape[i])
    return fhe_dim_sizes
