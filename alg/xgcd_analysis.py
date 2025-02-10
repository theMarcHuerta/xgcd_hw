#!/usr/bin/env python3
import math
import random
import sys
from xgcd_impl import xgcd_bitwise

###########################################################################
# Helper Functions (extracted and slightly adjusted)
###########################################################################

def bit_length(x):
    """Return the bit length of x (ignoring leading zeros)."""
    return x.bit_length()

def align_b(a_val, b_val):
    """
    Shift b_val left so that its leading 1 lines up with the leading 1 of a_val.
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
    If x_val has fewer than approx_bits bits, left-shift it so it has exactly approx_bits.
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
        return 0  # safeguard
    numerator = a_top << approx_bits
    if rounding_mode == "round":
        return (numerator + (b_top >> 1)) // b_top
    else:
        return numerator // b_top


###########################################################################
# 1. LUT Analysis with Ratio Information
###########################################################################

def lut_analysis(approx_bits, rounding_mode='truncate', target_ratio=None):
    """
    For all possible a_top and b_top values (with exactly approx_bits bits),
    compute the LUT result and also record the ratio a_top/b_top.
    If target_ratio is given, also compute the error.
    Returns a list sorted by quotient or by error if target_ratio is specified.
    """
    results = []
    low = 1 << (approx_bits - 1)
    high = 1 << approx_bits
    for a_top in range(low, high):
        for b_top in range(low, high):
            q = lut_result(a_top, b_top, approx_bits, rounding_mode)
            ratio = a_top / b_top
            error = abs(ratio - target_ratio) if target_ratio is not None else None
            results.append((a_top, b_top, q, ratio, error))
    if target_ratio is not None:
        results.sort(key=lambda x: x[4])
    else:
        results.sort(key=lambda x: x[2])
    return results

###########################################################################
# 2. Reverse Algorithm Candidate (Targeted)
###########################################################################

def reverse_candidate_target(total_bits, target_ratio=1.43648):
    """
    Construct a candidate pair using a target ratio.
    One idea: choose b as round((2^total_bits - 1) / (target_ratio + 1))
    and let a = (2^total_bits - 1) - b.
    """
    max_val = (1 << total_bits) - 1
    b = round(max_val / (target_ratio + 1))
    a = max_val - b
    return a, b

###########################################################################
# 3. Hill Climbing Metaheuristic (Targeted)
###########################################################################

def hill_climb_xgcd_target(total_bits, approx_bits=4, rounding_mode='truncate',
                           target_ratio=1.43648, iterations=2000, weight=10, seed=None):
    """
    Hill climbing with a fitness that rewards high iteration count and penalizes deviation from target_ratio.
    Fitness = iteration_count - weight * |(a/b) - target_ratio|
    """
    if seed is not None:
        random.seed(seed)
    lower_bound = 1 << (total_bits - 1)
    upper_bound = (1 << total_bits) - 1

    a = random.randint(lower_bound, upper_bound)
    b = random.randint(lower_bound, upper_bound)
    if b > a:
        a, b = b, a

    _, iter_count, _ = xgcd_bitwise(a, b, total_bits=total_bits, approx_bits=approx_bits,
                                     rounding_mode=rounding_mode)
    curr_ratio = a / b if b != 0 else float('inf')
    best_fitness = iter_count - weight * abs(curr_ratio - target_ratio)
    best_candidate = (a, b)

    for i in range(iterations):
        new_a = a + random.randint(-10, 10)
        new_b = b + random.randint(-10, 10)
        new_a = max(lower_bound, min(upper_bound, new_a))
        new_b = max(lower_bound, min(upper_bound, new_b))
        if new_b > new_a:
            new_a, new_b = new_b, new_a
        _, new_iter, _ = xgcd_bitwise(new_a, new_b, total_bits=total_bits, approx_bits=approx_bits,
                                       rounding_mode=rounding_mode)
        new_ratio = new_a / new_b if new_b != 0 else float('inf')
        new_fitness = new_iter - weight * abs(new_ratio - target_ratio)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_candidate = (new_a, new_b)
            a, b = new_a, new_b
    return best_candidate, best_fitness

###########################################################################
# 4. Hybrid Bottom-Up Construction (Targeted)
###########################################################################

def hybrid_bottom_up_xgcd_target(initial_bits, target_bits, approx_bits=4, rounding_mode='truncate',
                                 target_ratio=1.43648, neighborhood=8, weight=10):
    """
    First, brute-force the worst-case pair for a small bit-width (initial_bits)
    using iteration count. Then "grow" the candidate one bit at a time,
    searching locally for candidates that maximize our fitness:
       fitness = iteration_count - weight * |(a/b)-target_ratio|
    """
    # Brute force on initial_bits:
    lower_bound = 1 << (initial_bits - 1)
    upper_bound = (1 << initial_bits) - 1
    best_candidate = None
    best_iter = -1
    for a in range(lower_bound, upper_bound + 1):
        for b in range(lower_bound, a + 1):
            _, iter_count, _ = xgcd_bitwise(a, b, total_bits=initial_bits, approx_bits=approx_bits,
                                           rounding_mode=rounding_mode)
            if iter_count > best_iter:
                best_iter = iter_count
                best_candidate = (a, b)
    candidate = best_candidate
    curr_bits = initial_bits

    while curr_bits < target_bits:
        new_bits = curr_bits + 1
        lower_bound = 1 << (new_bits - 1)
        upper_bound = (1 << new_bits) - 1
        a_old, b_old = candidate
        # Scale the candidate up roughly:
        scale = 1 << (new_bits - curr_bits)
        base_a = a_old * scale
        base_b = b_old * scale
        # Now search a small neighborhood:
        variants = []
        for da in range(-neighborhood, neighborhood + 1):
            for db in range(-neighborhood, neighborhood + 1):
                new_a = base_a + da
                new_b = base_b + db
                # Ensure bounds and a>=b:
                new_a = max(lower_bound, min(upper_bound, new_a))
                new_b = max(lower_bound, min(upper_bound, new_b))
                if new_b > new_a:
                    new_a, new_b = new_b, new_a
                _, iter_count, _ = xgcd_bitwise(new_a, new_b, total_bits=new_bits, approx_bits=approx_bits,
                                               rounding_mode=rounding_mode)
                ratio = new_a / new_b if new_b != 0 else float('inf')
                fitness = iter_count - weight * abs(ratio - target_ratio)
                variants.append((new_a, new_b, iter_count, fitness))
        best_variant = max(variants, key=lambda x: x[3])
        candidate = (best_variant[0], best_variant[1])
        curr_bits = new_bits

    _, final_iter, _ = xgcd_bitwise(candidate[0], candidate[1], total_bits=target_bits, approx_bits=approx_bits,
                                    rounding_mode=rounding_mode)
    return candidate, target_bits, final_iter

###########################################################################
# Main – Demonstrate the improved approaches.
###########################################################################

if __name__ == "__main__":
    # Set parameters:
    total_bits = 16  # You can try 14 or 16 etc.
    approx_bits = 4
    rounding_mode = 'truncate'  # or 'round'
    # For TRUNCATE mode, try target_ratio ~1.43666; for ROUND mode, you might try 1.56.
    target_ratio = 1.43648

    print("=== 0. Testing the xgcd_bitwise function ===")
    a_test = 12370
    b_test = 9053
    gcd_val, iterations, avg_clears = xgcd_bitwise(a_test, b_test, total_bits=total_bits,
                                                    approx_bits=approx_bits, rounding_mode=rounding_mode)
    print(f"GCD({a_test}, {b_test}) = {gcd_val}, iterations = {iterations}, avg bit clears = {avg_clears:.3f}\n")

    #####################################################################
    # 1. LUT Analysis with Ratio Info
    #####################################################################
    print("=== 1. LUT Analysis (with target_ratio={}) ===".format(target_ratio))
    lut_results = lut_analysis(approx_bits, rounding_mode, target_ratio=target_ratio)
    print("Top 10 entries sorted by closeness to target ratio:")
    for entry in lut_results[:10]:
        a_top, b_top, q, ratio, err = entry
        print(f" a_top = {a_top:>2}, b_top = {b_top:>2}, quotient = {q:>3}, ratio = {ratio:.3f}, error = {err:.3f}")
    print()

    #####################################################################
    # 2. Reverse Candidate (Targeted)
    #####################################################################
    print("=== 2. Reverse Candidate (target_ratio = {}) ===".format(target_ratio))
    candidate_rev = reverse_candidate_target(total_bits, target_ratio)
    gcd_val, iter_count, _ = xgcd_bitwise(candidate_rev[0], candidate_rev[1],
                                           total_bits=total_bits, approx_bits=approx_bits,
                                           rounding_mode=rounding_mode)
    ratio = candidate_rev[0] / candidate_rev[1] if candidate_rev[1] != 0 else float('inf')
    print(f"Reverse candidate for {total_bits}-bit: a = {candidate_rev[0]}, b = {candidate_rev[1]}")
    print(f"  Ratio = {ratio:.3f} (target {target_ratio}), iterations = {iter_count}\n")

    #####################################################################
    # 3. Hill Climbing Metaheuristic (Targeted)
    #####################################################################
    print("=== 3. Hill Climbing Metaheuristic (target_ratio = {}) ===".format(target_ratio))
    candidate_hill, hill_fitness = hill_climb_xgcd_target(total_bits, approx_bits, rounding_mode,
                                                           target_ratio, iterations=5000, weight=10, seed=42)
    gcd_val, iter_count, _ = xgcd_bitwise(candidate_hill[0], candidate_hill[1],
                                           total_bits=total_bits, approx_bits=approx_bits,
                                           rounding_mode=rounding_mode)
    ratio = candidate_hill[0] / candidate_hill[1] if candidate_hill[1] != 0 else float('inf')
    print(f"Hill Climbing candidate for {total_bits}-bit: a = {candidate_hill[0]}, b = {candidate_hill[1]}")
    print(f"  Ratio = {ratio:.3f} (target {target_ratio}), iterations = {iter_count} (fitness = {hill_fitness:.3f})\n")

    #####################################################################
    # 4. Hybrid Bottom-Up Construction (Targeted)
    #####################################################################
    init_bits = 6  # start with a small bit-width
    print("=== 4. Hybrid Bottom-Up Construction (target_ratio = {}) ===".format(target_ratio))
    candidate_hybrid, final_bits, hybrid_iter = hybrid_bottom_up_xgcd_target(init_bits, total_bits,
                                                                              approx_bits, rounding_mode,
                                                                              target_ratio, neighborhood=8, weight=10)
    gcd_val, iter_count, _ = xgcd_bitwise(candidate_hybrid[0], candidate_hybrid[1],
                                           total_bits=total_bits, approx_bits=approx_bits,
                                           rounding_mode=rounding_mode)
    ratio = candidate_hybrid[0] / candidate_hybrid[1] if candidate_hybrid[1] != 0 else float('inf')
    print(f"Hybrid candidate grown from {init_bits} to {final_bits} bits:")
    print(f"   a = {candidate_hybrid[0]}, b = {candidate_hybrid[1]}")
    print(f"  Ratio = {ratio:.3f} (target {target_ratio}), iterations = {iter_count} (fitness candidate iter = {hybrid_iter})\n")

    print("=== End of Demonstration ===")
