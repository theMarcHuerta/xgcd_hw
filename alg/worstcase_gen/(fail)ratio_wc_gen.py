#!/usr/bin/env python3

import math

##############################################################################
# 1) Approximate XGCD Implementation
##############################################################################

def xgcd_bitwise(a_in, b_in, total_bits=8, approx_bits=4, 
                 rounding_mode='truncate', integer_rounding=True, 
                 enable_plotting=False):
    """
    Compute the GCD of a_in and b_in using the custom XGCD bitwise approach.
    """

    def bit_length(x):
        return x.bit_length()

    def align_b(a_val, b_val):
        len_a = bit_length(a_val)
        len_b = bit_length(b_val)
        shift_amount = len_a - len_b
        if shift_amount > 0:
            return (b_val << shift_amount), shift_amount
        else:
            return b_val, 0

    def get_fixed_top_bits(x_val, bits):
        if x_val == 0:
            return 0
        length = bit_length(x_val)
        if length <= bits:
            return x_val << (bits - length)
        else:
            return x_val >> (length - bits)

    def lut_result(a_top, b_top, abits, rmode):
        if b_top == 0:
            return 0
        numerator = a_top << abits
        if rmode == 'round':
            return (numerator + (b_top >> 1)) // b_top
        else:
            return numerator // b_top

    # Mask inputs to total_bits
    mask = (1 << total_bits) - 1
    a = a_in & mask
    b = b_in & mask

    # Ensure a >= b
    if b > a:
        a, b = b, a

    if b == 0:
        return (a, 0, 0.0)
    if a == 0:
        return (b, 0, 0.0)

    iteration_count = 0
    bit_clears_list = []

    while b != 0:
        iteration_count += 1

        # Align
        b_aligned, shift_amount = align_b(a, b)
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b_aligned, approx_bits)

        # LUT
        quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)

        # Shifting logic
        Q_pre = (quotient << shift_amount) >> (approx_bits - 1)
        Q = Q_pre >> 1
        if (Q_pre & 1) and integer_rounding:
            Q += 1

        # Subtraction
        b_adj = b * Q
        residual = a - b_adj
        if residual < 0:
            residual = -residual

        clears = bit_length(a) - bit_length(residual)
        if clears < 0:
            clears = 0
        bit_clears_list.append(clears)

        # Update
        if residual > b:
            a = residual
        else:
            a, b = b, residual

    # Last iteration might yield b=0, so we've effectively cleared all bits from that final step
    # but let's just store the final a as gcd:
    gcd_val = a

    if iteration_count > 0:
        avg_clears = sum(bit_clears_list) / iteration_count
    else:
        avg_clears = 0.0

    # Optional plot
    if enable_plotting and iteration_count > 0:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(bit_clears_list, marker='o')
        plt.title("Bit Clears per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Bit Clears")
        plt.show()

    return gcd_val, iteration_count, avg_clears


##############################################################################
# 2) A Two-Phase Worst-Case Generator
##############################################################################

def generate_worst_case_pair(total_bits=16, 
                             initial_bits=4, 
                             rounding_mode='truncate', 
                             desired_ratio=1.5):
    """
    Builds two integers A and B (up to total_bits).
    Phase A: 
        - Start from an initial seed of `initial_bits` that is near desired_ratio (e.g. 1.5).
        - For each new bit from (initial_bits+1) up to total_bits, 
          try all 4 possibilities for (a_bit, b_bit), 
          pick whichever yields a ratio closest to 'desired_ratio'. 
        - Do *not* apply any xgcd step or update.
    Phase B:
        - Return the final (a, b).
    """
    # 1) Pick a seed pair with initial_bits that yields a ratio near desired_ratio.
    #    For example, we can brute force all pairs in [2^(bits-1) .. 2^bits-1]
    #    and pick whichever ratio is closest to desired_ratio.
    #    Or just pick (12,8) if bits=4 for ratio = 1.5 exactly.
    if initial_bits < 2:
        raise ValueError("initial_bits must be >= 2")

    low = 1 << (initial_bits - 1)        # e.g. for 4 bits, low = 8 (1000 binary)
    high = (1 << initial_bits) - 1       # e.g. for 4 bits, high = 15 (1111 binary)

    best_seed = None
    best_seed_ratio_diff = None
    for sA in range(low, high+1):
        for sB in range(low, high+1):
            if sB == 0:
                continue
            rr = sA / sB
            diff = abs(rr - desired_ratio)
            if best_seed is None or diff < best_seed_ratio_diff:
                best_seed = (sA, sB)
                best_seed_ratio_diff = diff

    a, b = best_seed  # For 4 bits, this is typically (12, 8) if desired_ratio=1.5.

    # 2) Grow from initial_bits up to total_bits
    current_bits = initial_bits
    while current_bits < total_bits:
        next_width = current_bits + 1
        mask = (1 << next_width) - 1

        # Among the 4 ways to add bits, pick the ratio that stays closest to desired_ratio
        best_choice = None
        best_diff = None
        for abit in [0,1]:
            for bbit in [0,1]:
                A_candidate = ((a << 1) | abit) & mask
                B_candidate = ((b << 1) | bbit) & mask
                # Need to avoid B_candidate = 0 for ratio
                if B_candidate == 0:
                    continue
                rr = A_candidate / B_candidate
                diff = abs(rr - desired_ratio)
                if best_choice is None or diff < best_diff:
                    best_choice = (A_candidate, B_candidate)
                    best_diff = diff

        a, b = best_choice
        current_bits = next_width

    # Return the final pair
    return a, b


##############################################################################
# 3) Demo / Main
##############################################################################

if __name__ == "__main__":
    TOTAL_BITS = 19
    APPROX_BITS = 4
    ROUND_MODE  = 'truncate'
    DESIRED_Q   = 1.43626433
    
    print(f"Generating a worst-case pair for {TOTAL_BITS}-bit XGCD, "
          f"focusing on ratio ~ {DESIRED_Q} ...")

    # Phase A: build the pair
    a_candidate, b_candidate = generate_worst_case_pair(total_bits=TOTAL_BITS,
                                                        initial_bits=APPROX_BITS,
                                                        rounding_mode=ROUND_MODE,
                                                        desired_ratio=DESIRED_Q)
    print(f"  Final candidate A={a_candidate} (0b{a_candidate:016b})")
    print(f"  Final candidate B={b_candidate} (0b{b_candidate:016b})")
    if b_candidate != 0:
        print(f"  Final ratio A/B = {a_candidate/b_candidate:.4f}")
    else:
        print(f"  B=0!  (unlikely)")

    # Phase B: run approximate XGCD on the final pair
    gcd_val, iters, avg_clears = xgcd_bitwise(a_candidate, b_candidate, 
                                              total_bits=TOTAL_BITS,
                                              approx_bits=APPROX_BITS,
                                              rounding_mode=ROUND_MODE,
                                              integer_rounding=True,
                                              enable_plotting=False)

    print("\nXGCD results on that candidate:")
    print(f"  gcd(A,B) = {gcd_val}")
    print(f"  iteration_count = {iters}")
    print(f"  avg_bit_clears = {avg_clears:.3f}")
    print(f"(Expect worst-case near 15 iterations for 16-bit, if ratio ~ 1.5.)")
