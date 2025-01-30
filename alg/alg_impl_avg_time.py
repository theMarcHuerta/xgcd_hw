#!/usr/bin/env python3
import math
##########################################################################################
##########################################################################################
# TO DOs: 
# 1. Add support for bits beyond standard python sizes (eg 64 bits)
#       maybe look into making arrays or something 
# 2. Update some of the fixed point logic
# 3. Add checks for factoring out factors of 2 and adding it back in at the end
# 4. Update bit_length to work with the longer bit lengths as well
# 5. Update allign_b to work with larger numbers
# 6. Add counter to count how many iterations it's been to reach a gcd
##########################################################################################
##########################################################################################

def xgcd_bitwise(a_in, b_in, total_bits=8, approx_bits=4, rounding_mode='truncate'):
    """
    Compute the GCD of a_in and b_in using the custom XGCD bitwise approach from Kavya's Thesis
    
    :param a_in: First integer (up to 1024 bits) (can be as low as 8)
    :param b_in: Second integer (up to 1024 bits) (can be as low as 8)
    :param total_bits: The fixed total bit-width we assume for a and b.
    :param approx_bits: Number of bits for the approximate division step.
                        The first bit is treated as integer '1', and the next (approx_bits-1) bits are fractional.
    :param rounding_mode: How to handle the fractional part. Can be 'truncate', 'floor', or 'round' (to nearest),
                          or 'ceil' as you see fit.
    :return: The GCD of a_in and b_in according to the custom iteration.
    """

    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    # STEP 1) Normalize inputs (ensure within requested bit size, pick a >= b)
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------

    # Just to be safe, mask off anything above total_bits
    mask = (1 << total_bits) - 1
    a = a_in & mask
    b = b_in & mask

    # Ensure a >= b initially
    if b > a:
        a, b = b, a

    # Quick check for trivial cases
    if b == 0:
        return a
    if a == 0:
        return b
    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ################################ HELPER FUNCTIONS ########################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    def bit_length(x):
        """Return the bit length of x not counting leading 0s (in order to allign a and b)"""
        return x.bit_length()

    def align_b(a_val, b_val):
        """
        Shift b_val left so that its leading '1' lines up with the leading '1' of a_val.
        Returns (b_shifted, shift_amount).
        """
        len_a = bit_length(a_val)
        len_b = bit_length(b_val)
        shift_amount = len_a - len_b
        if shift_amount > 0:
            b_shifted = b_val << shift_amount
        else:
            # If b is already as large or larger in bit-length, we might not shift or might shift the other way
            b_shifted = b_val
            shift_amount = 0
        return b_shifted, shift_amount
    
    def get_fixed_point_approx(x_val, approx_bits):
        """
        Extract the top `approx_bits` from x_val's leading bits and interpret as:
            1 . (approx_bits-1 fractional bits) in fixed point
                i.e. approx_bits = 4, x_val = 110001110, turn it into 1.100

        If x_val has fewer than approx_bits bits, we pad the fractional bits with zeros.

        Returns the *floating* or *fractional* representation (depending on rounding_mode).
        """
        # If x_val == 0, return 0.0 directly:
        if x_val == 0:
            return 0.0

        length = bit_length(x_val)
        if length <= approx_bits:
            # The topmost bit is '1' (assuming x_val>0), remainder are fraction
            # i.e. x_val = 0b101 ===> length=3 ===> approx_bits=4 ===> top=101 ===> treat as 1.01 in binary
            bits_to_pad = approx_bits - length
            x_val = x_val << bits_to_pad
            length = approx_bits

        # length > approx_bits => we need to cut the top 'approx_bits' bits out
        shift_down = length - approx_bits
        top_bits = x_val >> shift_down  # This extracts the top 'approx_bits' bits

        # The top bit is the integer '1' in the fixed-point sense, rest are fractional
        fractional_length = approx_bits - 1  # # of fraction bits
        frac_part = top_bits & ((1 << fractional_length) - 1)
        
        # Now interpret top_bits as: 1.<frac_part> in binary
        # fraction = frac_part / 2^(fractional_length)
        # Now build the fractional value:
        # binary fraction = frac_part / 2^(fractional_length)
        # but frac_part includes the leftover bits after the topmost '1'.
        # For instance, if top_bits=5(=0b101), fractional_length=3 => leadisng bit=1, fraction=0b01 => 1 + 1/4
        fraction_value = frac_part / float(1 << fractional_length)
        approx_value = 1.0 + fraction_value

        return approx_value
    

    ## truncating would mean 1/2^fractional bits times, it will round up?
    def lut_result(q_val, approx_bits):
        """
        This would approximate what we would have in a hardware LUT. 
        It will return a float with approx_bits of precision in fixed point representation.
        For ex. if q_val is 1.95 and we have 3 fractional bits
            With rounding, we should get 2.0
            With truncating, we should get 1.875
        Another ex. again with 3 fractional bits, if q_val is 1.82
            With rounding, we should get 1.875
            With truncating we should get 1.75.
        Returns a floating point number representing the fixed point approximation
        for our selected preision
        """
        # Scale the value by 2^(approx_bits)
        scale = 1 << approx_bits
        scaled_val = q_val * scale

        if rounding_mode == "round":
            # Standard Python round (ties to even).  
            # If you want "round half away from zero" instead, you can implement it yourself:
            # scaled_int = int(math.floor(scaled_val + 0.5)) if scaled_val >= 0 else int(math.ceil(scaled_val - 0.5))
            scaled_int = round(scaled_val)
        else:
            # Default is truncate => floor for positive numbers
            scaled_int = math.floor(scaled_val)

        # Convert back by dividing by 2^(approx_bits)
        return scaled_int / scale

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ######################## ALGORITHM IMPLEMENTATION ########################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    while b != 0:
        # STEP 2) Align b so that the leading 1 matches a's leading 1 
        b_aligned, shift_amount = align_b(a, b)

        # If aligning yields b_aligned = 0 (only if b=0 originally), break
        if b_aligned == 0:
            break

        # STEP 3) Approximate division with approx_bits
        # Extract top approx_bits from a => a_approx
        a_approx = get_fixed_point_approx(a, approx_bits)
        # Extract top approx_bits from b_aligned => b_approx
        # (We use b_aligned rather than b so that the leading '1' of b_aligned lines up with a)
        b_approx = get_fixed_point_approx(b_aligned, approx_bits)

        if b_approx == 0.0: 
            # avoid div by 0
            quotient = 0.0
        else:
            float_q = a_approx / b_approx  # floating approx
            quotient = lut_result(float_q, approx_bits) # convert to what would be in our LUT
        
        # STEP 4) Shift the quotient by shift_amount => multiply by 2^(shift_amount)
        shifted_q = quotient * (1 << shift_amount)
        
        # STEP 5) Take the integer part of shifted_q => Q
        Q = int(shifted_q)  # floor/truncate to get the integer portion

        # STEP 6) b_adjusted = Q * b (the original b, not b_aligned)
        b_adjusted = b * Q

        # a_new = a - b_adjusted
        residual = a - b_adjusted
        # if a_new < 0:
        #     a_new = abs(a_new)

        # STEP 7) Prepare next iteration: 
        #    the old b becomes new a, the result becomes new b
        #    if the result > old b, swap them.
        #    old b is 'b', new residual is 'a_new'
        if residual > b:
            # then a_new becomes 'a', and b stays b
            a = residual
        else:
            # a_new is smaller => a = b, b = a_new
            a, b = b, residual

    # When b=0, a is the GCD
    return a

# -------------------------------------------------------------------------
# A small demo/test
if __name__ == "__main__":
    import sys

    # If you want interactive input:
    # a_in = int(input("Enter first number: "))
    # b_in = int(input("Enter second number: "))

    # Or hard-code a small example
    a_in = 128
    b_in = 56

    gcd_val = xgcd_bitwise(a_in, b_in,
                           total_bits=8,
                           approx_bits=4,
                           rounding_mode='truncate')
    
    print(f"GCD of {a_in} and {b_in} (via custom XGCD) is {gcd_val}")
