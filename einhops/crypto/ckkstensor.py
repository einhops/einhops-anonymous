import math
import torch
from dataclasses import dataclass
from .backend import (
    fhe_encode, fhe_encrypt, fhe_decrypt, fhe_decode,
    fhe_add, fhe_mul, fhe_rotate, fhe_mul_plain,
    SLOT_COUNT
)
import desilofhe

@dataclass
class CKKSTensor:
    slots: torch.Tensor        # For now: dummy torch tensor
    shape: torch.Size          # Original logical shape
    fhe_shape: torch.Size      # Padded power-of-2 shape  
    ndim: int

    def __str__(self):
        return f"CKKSTensor(shape={self.shape}, fhe_shape={self.fhe_shape})"
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError("CKKSTensors must have the same shape")

        res_add = fhe_add(self.slots, other.slots)

        return CKKSTensor(
            res_add,
            self.shape, 
            self.fhe_shape, 
            self.ndim
        )
    
    def __mul__(self, other):
        if self.shape != other.shape:
            raise ValueError("CKKSTensors must have the same shape")

        res_mul = fhe_mul(self.slots, other.slots
                          )
        return CKKSTensor(
            res_mul,
            self.shape,
            self.fhe_shape, 
            self.ndim
        )
    
    def rotate(self, steps):
        """Rotate slots by given number of steps"""
        res_rot = fhe_rotate(self.slots, steps)
        return CKKSTensor(res_rot, self.shape, self.fhe_shape, self.ndim)

# functions for translating arbitrary tensors -> FHE tensors -> CKKS vectors
def next_power_of_two(n):
    """
    Return the smallest power of two >= n.
    >>> next_power_of_two(9)
    16
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << (math.ceil(math.log2(n)))

def power_of_two_dims(sizes):
    """
    Pads each dimension to the closest power of two
    >>> power_of_two_dims(torch.Size([2, 3]))
    [2, 4]
    >>> power_of_two_dims(torch.Size([3, 17]))
    [4, 32]
    """
    return torch.Size(list(map(next_power_of_two, sizes)))

def tensor_to_padded_tensor(clear_tensor):
    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor
    return fhe_tensor

def tensor_to_packed_vector(clear_tensor):
    ## hot fix for now
    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor

    slots = torch.zeros(SLOT_COUNT)
    slots[:fhe_tensor.numel()] = fhe_tensor.flatten()

    return slots

def tensor_to_ptxt(clear_tensor):
    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor

    slots = torch.zeros(SLOT_COUNT)
    slots[:fhe_tensor.numel()] = fhe_tensor.flatten()

    encoded_tensor = fhe_encode(slots)

    return CKKSTensor(
        slots=encoded_tensor,
        shape=clear_shape,
        fhe_shape=fhe_shape,
        ndim=len(clear_shape)
    )


def tensor_to_fhe(clear_tensor):
    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor

    slots = torch.zeros(SLOT_COUNT)
    slots[:fhe_tensor.numel()] = fhe_tensor.flatten()

    encoded_tensor = fhe_encode(slots)
    encrypted_tensor = fhe_encrypt(encoded_tensor)

    return CKKSTensor(
        slots=encrypted_tensor,
        shape=clear_shape,
        fhe_shape=fhe_shape,
        ndim=len(clear_shape)
    )

def fhe_to_tensor(fhe_tensor):

    ptxt_tensor = fhe_decrypt(fhe_tensor.slots)
    clear_tensor = torch.tensor(fhe_decode(ptxt_tensor), dtype=torch.float32)

    # edge-case: zero (unit tensors) or one dimensional (1-d arrays)
    if fhe_tensor.ndim == 0:
        return clear_tensor[0]
    elif fhe_tensor.ndim == 1:
        return clear_tensor[:fhe_tensor.shape.numel()]

    clear_shape = fhe_tensor.shape
    clear_strides = torch.tensor(torch.empty(fhe_tensor.fhe_shape).stride())
    clear_tensor_indices = torch.cartesian_prod(*[torch.arange(shape) for shape in clear_shape])
    clear_indices = torch.sum(clear_tensor_indices * clear_strides, dim=1)
    clear_tensor = clear_tensor.flatten()[clear_indices]
    clear_tensor = clear_tensor.reshape(clear_shape)
    return clear_tensor
