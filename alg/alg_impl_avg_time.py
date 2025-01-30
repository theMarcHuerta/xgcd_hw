#!/usr/bin/env python3
import math
##########################################################################################
##########################################################################################
# TO DOs: 
# 1. What to do if residual is negative
# 2. Add checks for factoring out factors of 2 and adding it back in at the end
# 3. Add bit clears
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
    :param rounding_mode: How to handle the fractional part. Can be 'truncate', or 'round' (to nearest)
    :return: The GCD of a_in and b_in according to the custom iteration and iteration counts
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
    
    def get_fixed_top_bits(x_val, approx_bits):
        """
        Extract the top `approx_bits` bits of x_val as an integer.
        - If x_val = 0 => return 0.
        - If x_val has fewer than approx_bits bits, we left-shift it so it becomes exactly approx_bits bits.
        - Otherwise we shift down so we only keep the top approx_bits bits.
        
        Example: approx_bits=4
          - x_val in binary: 1100 1010 0111 ...
          - we keep top 4 bits: 1100 (which is 12 decimal)
        """
        if x_val == 0:
            return 0

        length = bit_length(x_val)
        if length <= approx_bits:
            # shift up so it becomes exactly approx_bits bits
            return x_val << (approx_bits - length)
        else:
            # shift down so we only keep top approx_bits
            shift_down = length - approx_bits
            return x_val >> shift_down
    

    def lut_result(a_top, b_top, approx_bits, rounding_mode):
        """
        Compute the ratio (a_top / b_top) in fixed-point with 'approx_bits' fractional bits
        """
        if b_top == 0:
            return 0  # avoid div by zero (should not happen if b!=0, but just in case)

        numerator = a_top << approx_bits  # up-shift a_top by approx_bits
        if rounding_mode == "round":
            # do integer division with rounding:
            # equivalent to floor( (numerator + b_top/2 ) / b_top )
            return (numerator + (b_top >> 1)) // b_top
        else:
            # default: truncate / floor
            return numerator // b_top

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ######################## ALGORITHM IMPLEMENTATION ########################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    iteration_count = 0  # to track how many loop iterations

    while b != 0:

        iteration_count += 1

        # STEP 2) Align b so that the leading 1 matches a's leading 1 
        b_aligned, shift_amount = align_b(a, b)

        # If aligning yields b_aligned = 0 (only if b=0 originally), break
        if b_aligned == 0:
            break

        # STEP 3) Approximate division with approx_bits
        # Extract top approx_bits
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b_aligned, approx_bits)

        quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
        
        # STEP 4) Shift the quotient by shift_amount => multiply by 2^(shift_amount)
        # STEP 5) Take the integer part of shifted_q => Q
        Q = (quotient << shift_amount) >> approx_bits

        # STEP 6) b_adjusted = Q * b (the original b, not b_aligned)
        b_adjusted = b * Q

        # a_new = a - b_adjusted
        residual = a - b_adjusted
        if residual < 0:
            residual = -residual

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
    return (a, iteration_count)

# -------------------------------------------------------------------------
# A small demo/test
if __name__ == "__main__":
    import sys

    # If you want interactive input:
    # a_in = int(input("Enter first number: "))
    # b_in = int(input("Enter second number: "))

    # 1) A small example
    a_in = 128
    b_in = 56
    gcd_val, count = xgcd_bitwise(a_in, b_in,
                                  total_bits=8,
                                  approx_bits=4,
                                  rounding_mode='truncate')
    print(f"(Small) GCD of {a_in} and {b_in} is {gcd_val}, reached in {count} iterations.")

    # 2) A bigger example (still within 16 bits)
    a_in = 34470   # 1100111110101011 in binary (16 bits)
    b_in = 45960   # 1011001111011000 in binary (16 bits)
    gcd_val, count = xgcd_bitwise(a_in, b_in,
                                  total_bits=16,
                                  approx_bits=4,
                                  rounding_mode='truncate')
    print(f"(Medium) GCD of {a_in} and {b_in} is {gcd_val}, reached in {count} iterations.")

    # 3) A large example: let’s do 256-bit numbers
    #    We can define them in hex and then parse via int(...)
    A_HEX = "F123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"
    B_HEX = "EFFF88889999AAAA77775555CCCFA123ABCDEF9876543210FABCD123456789AA"
    a_in = int(A_HEX, 16)
    b_in = int(B_HEX, 16)

    # We'll set total_bits=256 to clamp them at 256 bits, approx_bits=8 for a bigger approximation
    gcd_val, count = xgcd_bitwise(a_in, b_in,
                                  total_bits=256,
                                  approx_bits=8,
                                  rounding_mode='round')

    print(f"(Large 256-bit) GCD is {gcd_val}")
    print(f"  Found in {count} iterations.")
