import os
import torch
import desilofhe
from desilofhe import Engine

print("Initializing the CKKS Context.")
if torch.cuda.is_available():
    engine = Engine(max_level=17, mode='gpu')
else:
    # thread_count = min(os.cpu_count(), 16)
    thread_count = 26
    engine = Engine(max_level=17, mode="parallel", thread_count=thread_count)

# generate secret key
secret_key = engine.create_secret_key()
# generate public and evaluation keys
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
rotation_key = engine.create_rotation_key(secret_key)

SLOT_COUNT = engine.slot_count
assert SLOT_COUNT == 16384, "CKKS slot count must be 16384"

# generate maximal set of keys needed for BSGS
print("Generating BSGS keys...")
N1 = 128
N2 = 128
maximal_set = [-i for i in range(N1)] + [-j*N1 for j in range(N2)]
fixed_rotation_keys = {i : engine.create_fixed_rotation_key(secret_key, i) for i in maximal_set}
print("Complete.")

def fhe_encode(values):
    assert len(values) <= SLOT_COUNT, f"You must encode 1-d vectors of size {SLOT_COUNT} or smaller"
    return engine.encode(values)

def fhe_decode(values):
    return engine.decode(values)

def fhe_encrypt(plaintext):
    return engine.encrypt(plaintext, public_key)

def fhe_decrypt(ciphertext):
    return engine.decrypt_to_plaintext(ciphertext, secret_key)

def fhe_add(op1, op2):
    return engine.add(op1, op2)

def fhe_mul(op1, op2):
    ## rescale first -> check here: https://fhe.desilo.dev/latest/quickstart/#seal-style-api
    ## right now, this is CT-CT mul
    rescaled1 = engine.rescale(op1)
    rescaled2 = engine.rescale(op2)
    if isinstance(rescaled1, desilofhe.Plaintext):
        return engine.multiply(rescaled1, rescaled2)
    elif isinstance(rescaled2, desilofhe.Plaintext):
        return engine.multiply(rescaled1, rescaled2)
    else:
        return engine.multiply(rescaled1, rescaled2, relinearization_key)

def fhe_mul_plain(op1, op2):
    op2_ptxt = fhe_encode(op2)
    rescaled1 = engine.rescale(op1)
    rescaled2 = engine.rescale(op2_ptxt)
    return engine.multiply(rescaled1, rescaled2)

def fhe_rotate(op1, delta):
    ## their api: rotate(1) == np.roll(1), opposite of most libraries
    ## warning: multiple key-switches -> only uses power of 2 keys.
    return engine.rotate(op1, rotation_key, delta)

def fhe_rotate_fixed(op1, delta):
    return engine.rotate(op1, fixed_rotation_keys[delta])

def fhe_hoisted_rotate(op1, deltas):
    keys = []
    for delta in deltas:
        keys.append(fixed_rotation_keys[-delta])
    return engine.rotate_batch(op1, keys)
