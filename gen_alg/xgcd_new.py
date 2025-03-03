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
    Compute the GCD of a_in and b_in using the custom XGCD bitwise approach from Kavya's Thesis
    
    :param a_in: First integer (up to 1024 bits) (can be as low as 8)
    :param b_in: Second integer (up to 1024 bits) (can be as low as 8)
    :param total_bits: The fixed total bit-width we assume for a and b.
    :param approx_bits: Number of bits for the approximate division step.
                        The first bit is treated as integer '1', and the next (approx_bits-1) bits are fractional.
    :param rounding_mode: How to handle the fractional part. Can be 'truncate', or 'round' (to nearest)
    :param enable_plotting: If True, will plot bit-clears per iteration at the end.
    
    :return: (gcd_value, iteration_count, avg_bit_clears, bit_clears_list)
            gcd_value       -> final GCD
            iteration_count -> how many loops ran
            avg_bit_clears  -> average # of cleared bits per iteration
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
        return (a, 0, 0.0)  # gcd = a, 0 iterations, 0 bit clears
    if a == 0:
        return (b, 0, 0.0)  # gcd = b, 0 iterations, 0 bit clears

    
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

    avg_q = 0
    iteration_count = 0 

    bit_clears_list = []  # store how many bits we clear each iteration

    while b != 0:
        
        # print(f"A: {a:b}")        
        # print(f"B: {b:b}")
        # print("A is: ", a)
        # print("B is: ", b, "\n")

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
        # Q = (quotient << shift_amount) >> approx_bits
        Q_pre_round = (quotient << shift_amount) >> (approx_bits-1)
        Q = Q_pre_round >> 1

        # if (Q_pre_round&1 and integer_rounding):
        Q += 1

        avg_q += Q
        # STEP 6) b_adjusted = Q * b (the original b, not b_aligned)
        b_adjusted = b * Q
        b_adjusted_two = b * (Q_pre_round >> 1)

        # if (Q == 0):
        #     print("Q = 0")

        # print(Q)
        # a_new = a - b_adjusted
        residual = a - b_adjusted
        residual_two = a - b_adjusted_two

        # print("Q is: ", Q)

        if residual < 0:
            # print("Residual was negative")
            residual = -residual
        
        if residual_two < 0:
            # print("Residual was negative")
            residual_two = -residual_two

        if (residual_two < residual):
            residual = residual_two
            Q = (Q_pre_round >> 1)

        # print("\n")
        msb_a = bit_length(a)
        msb_res = bit_length(residual)
        clears_this_iter = msb_a - msb_res
        if clears_this_iter < 0:
            clears_this_iter = 0  # in case we "went backwards" in bit length
        bit_clears_list.append(clears_this_iter)

        # if (iteration_count == 1 and clears_this_iter == 1):
        # print(f"A: {a}")        
        # print(f"B: {b}")
        # print(f"R: {residual}")
        # print(f"Q: {Q}")
        # print(f"C: {clears_this_iter} \n")

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

        # print(a)
        # print(b)

    bit_clears_list[-1] += bit_length(a)# last element is off by A bits clear so we just adjust it

    gcd_val = a
    if iteration_count > 0:
        # avg_bit_clears = total_bits*2 / iteration_count
        avg_bit_clears = sum(bit_clears_list) / iteration_count
    else:
        avg_bit_clears = 0.0

    # print("NEW\n")
    # print(f"  Average q val: {avg_q/iteration_count:.3f}")
    # ----------------------------------------------------------------------------------------
    # Optional Plotting
    # ----------------------------------------------------------------------------------------
    if enable_plotting and iteration_count > 0:
        import matplotlib.pyplot as plt

        # Plot 1: bit clears per iteration
        plt.figure(figsize=(10, 4))

        # Subplot A: Clears per iteration
        plt.subplot(1, 2, 1)
        plt.plot(range(iteration_count), bit_clears_list, marker='o')
        plt.title("Bit Clears per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Bit Clears")

        # Subplot B: Cumulative sum of clears
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

    # print(bit_clears_list)
    return (gcd_val, iteration_count, avg_bit_clears)

# -------------------------------------------------------------------------
# A small demo/test
if __name__ == "__main__":
    import sys

    # # If you want interactive input:
    # # a_in = int(input("Enter first number: "))
    # # b_in = int(input("Enter second number: "))

    # 1) Small example
    a_in = 178787
    b_in = 125725
    gcd_val, count, avg_clears = xgcd_bitwise(a_in, b_in,
                                                           total_bits=20,
                                                           approx_bits=4,
                                                           rounding_mode='truncate',
                                                           integer_rounding=True,
                                                           plus_minus=False,
                                                           enable_plotting=False)
    print(f"(Small) GCD of {a_in} and {b_in} is {gcd_val}, reached in {count} iterations.")
    print(f"  Average bit clears: {avg_clears:.3f}")

    # # 2) Another example with 16 bits
    # a_in = 118184
    # b_in = 82273
    # gcd_val, count, avg_clears = xgcd_bitwise(a_in, b_in,
    #                                                        total_bits=17,
    #                                                        approx_bits=4,
    #                                                        rounding_mode='truncate',
    #                                                        integer_rounding=True,
    #                                                        plus_minus=False,
    #                                                        enable_plotting=False)
    # print(f"(Medium) GCD of {a_in} and {b_in} is {gcd_val}, reached in {count} iterations.")
    # print(f"  Average bit clears: {avg_clears:.3f}")

    # # 3) A large 256-bit example (enable_plotting=True to see the charts)
    # A_HEX = "ec30e4c53f6724857556f50afc80013b0995173248cef4d38bc099887fa83367d5dbd26953b22fe57ecd1921fb7b6309ca7cac791eb08301891182cd299e32b5"
    # B_HEX = "91f95c012469cf51b3f2301676d57142af0216b58586ffbfce71423398033050ff6e792fe74dd0378e5ae71b340102babf12e11a58d527f177fe2fa327e011d8"
    # a_in = int(A_HEX, 16)
    # b_in = int(B_HEX, 16)

    # gcd_val, count, avg_clears = xgcd_bitwise(a_in, b_in,
    #                                                        total_bits=512,
    #                                                        approx_bits=4,
    #                                                        rounding_mode='truncate',
    #                                                        integer_rounding=False,
    #                                                        plus_minus=False,
    #                                                        enable_plotting=False)

    # print(f"(Large 256-bit) GCD is {gcd_val}")
    # print(f"  Found in {count} iterations.")
    # print(f"  Average bit clears: {avg_clears:.3f}")