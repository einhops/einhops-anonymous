from .crypto.ckkstensor import CKKSTensor, tensor_to_fhe, fhe_to_tensor
from .engine import EinsumEngine

_engine = EinsumEngine()

def encrypt(tensor):
    # just dummy for now
    return tensor_to_fhe(tensor)

def decrypt(ckkstensor):
    # again, dummy for now
    return fhe_to_tensor(ckkstensor)

def einsum(equation, *args, **kwargs):
    # run the main einsum engine
    return _engine.einsum(equation, *args, **kwargs)

# for calling from einhops import *
__all__ = [
    'encrypt',
    'decrypt',
    'einsum',
    'CKKSTensor'
]