import math
import torch
import opt_einsum
from .parser import parse_dims, get_dim_sizes, get_fhe_dim_sizes
from .operations import expand_dimensions, gather_slots

class EinsumEngine:
    """Einhops: Einsum Notation for Homomorphic Operations on CKKS Tensors.

    This class is the main driver for Einhops. It is responsible for:
    - Validating and parsing an einsum expression
    - Pre-processing the inputs
    - Performing the einsum contraction
    - Post-processing the output to re-align the data
    """

    def einsum(self, equation, *args, debug=False):

        # Validate the einsum call using a pre-existing library.
        if debug:
            print("Stage 1 (Validation)")
        out_clear_shape = self._validate_expression(equation, *args)

        # Parse the einsum string and get the input, output, and reduction dims.
        if debug:
            print("Stage 2 (Dimension Semantics)")
        i_dims, o_dims, r_dims = parse_dims(equation, *args)
        dim_sizes = get_dim_sizes(i_dims, *args)
        fhe_dim_sizes = get_fhe_dim_sizes(i_dims, *args)
        dst_dims = r_dims + o_dims

        # Expand the dimensions of each input to match the output dimensions.
        if debug:
            print("Stage 3 (Dimension Expansion and Slot Alignment)")
        expanded_inputs = []
        for (src_dim, arg) in zip(i_dims, args):
            expanded = expand_dimensions(arg, src_dim, dst_dims, dim_sizes, fhe_dim_sizes)
            expanded_inputs.append(expanded)

        # Perform the multiplication; now, all inputs have the same dimensions.
        if debug:
            print("Stage 4 (Multiplication)")
        out_mult = self._multiply(expanded_inputs)

        # Letters omitted from the output are summed over. 
        if debug:
            print("Stage 5 (Reduction)")
        out_sum = self._reduce(out_mult, r_dims, o_dims, dim_sizes, fhe_dim_sizes)

        # We need to re-align our data to the top slots.
        if debug:
            print("Stage 6 (Gathering the output)")
        out = gather_slots(out_sum, o_dims, dim_sizes, fhe_dim_sizes)

        # Final check with the clear-text calculation. 
        assert out.shape == out_clear_shape, f"Error! Output shape mismatch: {out.shape} != {out_clear_shape}"
        return out

    def _validate_expression(self, equation, *args):
        if '->' not in equation:
            raise ValueError("Equation must contain ->")
        
        shapes = [arg.shape for arg in args]
        try:
            res_ = opt_einsum.contract(equation, *[torch.empty(shape) for shape in shapes])
            return res_.shape
        except Exception as e:
            raise ValueError(f"Invalid einsum expression: {e}")
        
    def _multiply(self, expanded_inputs):
        if len(expanded_inputs) == 1: # no multiplication needed
            return expanded_inputs[0]
        
        out = expanded_inputs[0]
        for operand in expanded_inputs[1:]:
            out = out * operand
        return out
    
    def _reduce(self, partial_mult_tensor, r_dims, o_dims, dim_sizes, fhe_dim_sizes):
        if not r_dims:  # No reduction needed
            return partial_mult_tensor

        
         
        rot_amount = math.prod([fhe_dim_sizes[dim] for dim in o_dims])
        num_repeated_o_dims = math.prod([fhe_dim_sizes[dim] for dim in r_dims])
        num_rots = int(math.log2(num_repeated_o_dims))
        
        result = partial_mult_tensor
        for i in range(num_rots):
            rotated = result.rotate(-rot_amount)
            result = result + rotated
            rot_amount *= 2

        out_shape = torch.Size([dim_sizes[dim] for dim in o_dims])
        out_fhe_shape = torch.Size([fhe_dim_sizes[dim] for dim in o_dims])
        out_ndim = len(out_fhe_shape)

        result.shape = out_shape
        result.fhe_shape = out_fhe_shape
        result.ndim = out_ndim
        return result
