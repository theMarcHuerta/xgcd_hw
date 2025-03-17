#!/usr/bin/env python3
import math
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

def xgcd_bitwise(a_in, b_in, total_bits=8, approx_bits=4):
    """
    Compute the GCD (and extended GCD) of a_in and b_in using the custom XGCD bitwise approach 
    from Kavya's Thesis.
    
    In addition to the gcd, this extended version also returns x and y such that:
         a_in * x + b_in * y = gcd(a_in, b_in)
    
    :param a_in: First integer (up to 1024 bits) (can be as low as 8)
    :param b_in: Second integer (up to 1024 bits) (can be as low as 8)
    :param total_bits: The fixed total bit-width we assume for a and b.
    :param approx_bits: Number of bits for the approximate division step.
                        The first bit is treated as integer '1', and the next (approx_bits-1) bits are fractional.
    :param rounding_mode: How to handle the fractional part. Can be 'truncate', or 'round' (to nearest)
    :param enable_plotting: If True, will plot bit-clears per iteration at the end.
    
    :return: (gcd_value, x, y, iteration_count, avg_bit_clears)
            gcd_value       -> final GCD
            x, y            -> coefficients satisfying a_in*x + b_in*y = gcd_value
            iteration_count -> how many loops ran
            avg_bit_clears  -> average # of cleared bits per iteration
    """

    # ----------------------------------------------------------------------------------------
    # STEP 1) Normalize inputs (ensure within requested bit size, pick a >= b)
    # ----------------------------------------------------------------------------------------
    mask = (1 << total_bits) - 1
    a = a_in & mask
    b = b_in & mask

    # Ensure a >= b initially
    if b > a:
        a, b = b, a
        # Also swap coefficients if you want to maintain a_in and b_in association.
    
    # Quick check for trivial cases
    if b == 0:
        # For xgcd: a_in = a*1 + b*0
        return (a, 1, 0, 0, 0.0)
    if a == 0:
        return (b, 0, 1, 0, 0.0)

    # ----------------------------------------------------------------------------------------
    # [xgcd] Initialize coefficient pairs.
    # For 'a': (x, y) so that a = a_in*x + b_in*y
    # For 'b': (u, v) so that b = a_in*u + b_in*v
    x, y = 1, 0
    u, v = 0, 1

    ##########################################################################################
    ################################ HELPER FUNCTIONS ########################################
    ##########################################################################################

    def bit_length(x):
        """Return the bit length of x (number of bits required for x)."""
        return x.bit_length()

    def align_b(a_val, b_val):
        """
        Shift b_val left so that its leading '1' lines up with the leading '1' of a_val.
        Returns (b_shifted, shift_amount).
        """
        len_a = bit_length(a_val)
        len_b = bit_length(b_val)
        shift_amount = len_a - len_b
        return shift_amount
    
    def get_fixed_top_bits(x_val, approx_bits):
        """
        Extract the top `approx_bits` bits of x_val as an integer.
        """
        length = bit_length(x_val)
        if length <= approx_bits:
            return x_val << (approx_bits - length)
        else:
            shift_down = length - approx_bits
            return x_val >> shift_down
    
    def lut_result(a_top, b_top, approx_bits):
        """
        Compute the ratio (a_top / b_top) in fixed-point with 'approx_bits' fractional bits.
        """
        numerator = a_top << approx_bits  # up-shift a_top by approx_bits
        return numerator // b_top

    ##########################################################################################
    ######################## ALGORITHM IMPLEMENTATION ########################################
    ##########################################################################################

    while b != 0:
        # STEP 2) Align b with a
        shift_amount = align_b(a, b)

        # STEP 3) Approximate division with approx_bits: extract top bits and compute quotient
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b, approx_bits)
        quotient = lut_result(a_top, b_top, approx_bits)
        
        b_shifted = b
        u_shifted = u
        v_shifted = v
        # STEP 4 & 5) Adjust quotient by shifting
        if (shift_amount > (approx_bits - 1)):
            shift_vars_amount = shift_amount - (approx_bits - 1)
            b_shifted = b << shift_vars_amount
            u_shifted = u << shift_vars_amount
            v_shifted = v << shift_vars_amount
            shift_amount = (approx_bits - 1)

        Q = (quotient << shift_amount) >> (approx_bits)

        # STEP 6) Compute the residual: r = a - Q*b
        b_mul = b_shifted * Q
        u_mul = u_shifted * Q
        v_mul = v_shifted * Q
        residual = a - b_mul
        # Alternative residual computation Q over estimate if it gives a smaller residual:
        residual_two = residual - b

        was_negative = False
        if residual < 0:
            was_negative = True
            residual = -residual

        was_negative_two = False
        if residual_two < 0:
            was_negative_two = True
            residual_two = -residual_two

        # see whcih residual is smaller
        if residual_two < residual:
            was_negative = was_negative_two
            residual = residual_two
            u_mul += u
            v_mul += v

        # [xgcd] Update coefficient pair for the new remainder.
        # Let new coefficients be:
        new_x = x - u_mul
        new_y = y - v_mul

        if (was_negative):
            new_x = -x + u_mul
            new_y = -y + v_mul

        # STEP 7) Update a and b (and their coefficients) for the next iteration.
        if residual > b:
            # Update only a (and its coefficients); b remains unchanged.
            a = residual
            x, y = new_x, new_y
            # u, v remain the same.
        else:
            # Swap: new a becomes old b, new b becomes the residual.
            a, b = b, residual
            x, y, u, v = u, v, new_x, new_y

    gcd_val = a
    # At termination, b is zero. The coefficients (x, y) associated with a satisfy:
    #    a_in*x + b_in*y = gcd_val
    # After the main loop, before returning:
    assert a_in * x + b_in * y == gcd_val, "Bézout identity is not satisfied!"

    return (gcd_val, x, y)