#!/usr/bin/env python3
import math
##########################################################################################
##########################################################################################
# TO DOs: 
# 1. Add checks for factoring out factors of 2 and adding it back in at the end
##########################################################################################
##########################################################################################

def xgcd_bitwise(a_in, b_in, total_bits=8, approx_bits=4, rounding_mode='truncate', 
                 integer_rounding=True, plus_minus=False, enable_plotting=False):
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
        if shift_amount > 0:
            b_shifted = b_val << shift_amount
        else:
            b_shifted = b_val
            shift_amount = 0
        return b_shifted, shift_amount
    
    def get_fixed_top_bits(x_val, approx_bits):
        """
        Extract the top `approx_bits` bits of x_val as an integer.
        """
        if x_val == 0:
            return 0

        length = bit_length(x_val)
        if length <= approx_bits:
            return x_val << (approx_bits - length)
        else:
            shift_down = length - approx_bits
            return x_val >> shift_down
    
    def lut_result(a_top, b_top, approx_bits, rounding_mode):
        """
        Compute the ratio (a_top / b_top) in fixed-point with 'approx_bits' fractional bits.
        """
        if b_top == 0:
            return 0  # avoid division by zero

        numerator = a_top << approx_bits  # up-shift a_top by approx_bits
        if rounding_mode == "round":
            return (numerator + (b_top >> 1)) // b_top
        else:
            return numerator // b_top

    ##########################################################################################
    ######################## ALGORITHM IMPLEMENTATION ########################################
    ##########################################################################################

    avg_q = 0
    iteration_count = 0 
    bit_clears_list = []  # store number of bits cleared each iteration

    while b != 0:
        iteration_count += 1

        # STEP 2) Align b with a
        b_aligned, shift_amount = align_b(a, b)
        if b_aligned == 0:
            break

        # STEP 3) Approximate division with approx_bits: extract top bits and compute quotient
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b_aligned, approx_bits)
        quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
        
        # STEP 4 & 5) Adjust quotient by shifting
        Q_pre_round = (quotient << shift_amount) >> (approx_bits-1)

        Q = Q_pre_round >> 1
        # Q = a // b

        if (Q_pre_round&1 and integer_rounding):
            Q += 1
            
        avg_q += Q

        # STEP 6) Compute the residual: r = a - Q*b
        b_adjusted = b * Q
        residual = a - b_adjusted

        was_negative = False
        if residual < 0:
            was_negative = True
            residual = -residual
        
        # Alternative residual computation (using Q_pre_round) if it gives a smaller residual:
        # b_adjusted_two = b * (Q_pre_round >> 1)
        # residual_two = a - b_adjusted_two

        # if residual_two < 0:
        #     residual_two = -residual_two
        # if residual_two < residual:
        #     residual = residual_two
        #     Q = (Q_pre_round >> 1)

        msb_a = bit_length(a)
        msb_res = bit_length(residual)
        clears_this_iter = msb_a - msb_res
        if clears_this_iter < 0:
            clears_this_iter = 0
        bit_clears_list.append(clears_this_iter)

        # [xgcd] Update coefficient pair for the new remainder.
        # Let new coefficients be:
        new_x = x - Q * u
        new_y = y - Q * v

        if (was_negative):
            new_x = -x + Q * u
            new_y = -y + Q * v


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

        # (Optional debugging output)
        # print(f"A: {a}")        
        # print(f"B: {b}")
        # print(f"R: {residual}")
        # print(f"Q: {Q}")
        # print(f"C: {clears_this_iter} \n")

    # Adjust the last bit clear count
    bit_clears_list[-1] += bit_length(a)
    gcd_val = a
    if iteration_count > 0:
        avg_bit_clears = sum(bit_clears_list) / iteration_count
    else:
        avg_bit_clears = 0.0

    # Optional Plotting (if enabled)
    if enable_plotting and iteration_count > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(iteration_count), bit_clears_list, marker='o')
        plt.title("Bit Clears per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Bit Clears")
        cumulative = []
        running = 0
        for c in bit_clears_list:
            running += c
            cumulative.append(running)
        plt.subplot(1, 2, 2)
        plt.plot(range(iteration_count), cumulative, marker='o', color='orange')
        plt.title("Cumulative Sum of Bit Clears")
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Clears")
        plt.tight_layout()
        plt.show()

    # At termination, b is zero. The coefficients (x, y) associated with a satisfy:
    #    a_in*x + b_in*y = gcd_val
    # After the main loop, before returning:
    assert a_in * x + b_in * y == gcd_val, "Bézout identity is not satisfied!"

    return (gcd_val, x, y, iteration_count, avg_bit_clears)

# -------------------------------------------------------------------------
# A small demo/test
if __name__ == "__main__":
    # Example values
    a_in = 46368
    b_in = 28657
    result = xgcd_bitwise(a_in, b_in,
                          total_bits=16,
                          approx_bits=4,
                          rounding_mode='truncate',
                          integer_rounding=True,
                          plus_minus=False,
                          enable_plotting=False)
    gcd_val, x, y, count, avg_clears = result
    print(f"(Extended) GCD of {a_in} and {b_in} is {gcd_val}, reached in {count} iterations.")
    print(f"  Coefficients: x = {x}, y = {y}")
    print(f"  Average bit clears: {avg_clears:.3f}")
