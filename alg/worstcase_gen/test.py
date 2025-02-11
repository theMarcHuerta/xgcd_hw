#!/usr/bin/env python3
import math
import random

###############################################################################
# Helper Functions (same as used in the xgcd_bitwise algorithm)
###############################################################################

def bit_length(x):
    """Return the bit length of x (if x == 0, return 1 for our purposes)."""
    if x == 0:
        return 1
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
    Extract the top `approx_bits` of x_val.
    If x_val has fewer than approx_bits bits, left-shift it to make it exactly approx_bits bits.
    """
    if x_val == 0:
        return 0
    length = bit_length(x_val)
    if length < approx_bits:
        return x_val << (approx_bits - length)
    else:
        shift_down = length - approx_bits
        return x_val >> shift_down

def lut_result(a_top, b_top, approx_bits, rounding_mode):
    """
    Compute the fixed-point ratio of a_top/b_top using approx_bits fractional bits.
    """
    if b_top == 0:
        return 0  # safeguard; should not happen if b ≠ 0.
    numerator = a_top << approx_bits
    if rounding_mode == "round":
        # rounding (round-half-up)
        return (numerator + (b_top >> 1)) // b_top
    else:
        # default is truncate/floor
        return numerator // b_top

###############################################################################
# Simulation of One XGCD-Style Iteration
###############################################################################

def simulate_xgcd_iteration(a, b, approx_bits, rounding_mode, integer_rounding):
    """
    Given numbers a and b, simulate one iteration of the xgcd-style update.
    This follows the steps in your xgcd_bitwise loop:
      - Align b with a.
      - Extract the top approx_bits from a and from the aligned b.
      - Compute the approximate quotient Q using the LUT.
      - Multiply b by Q, subtract from a, and compute the residual.
      - Compute the number of bits cleared (bit_length(a) - bit_length(residual)).
      - Update the pair according to the rule: if residual > b, then new_a = residual, else swap.
    
    Returns a dictionary with:
      new_a, new_b: the updated pair,
      quotient: the Q value,
      residual: the computed residual,
      bit_clears: number of bits cleared,
      trace: a dictionary of intermediate values for debugging.
    """
    trace = {}
    # Align b to a
    b_aligned, shift_amount = align_b(a, b)
    trace['shift_amount'] = shift_amount

    # Get top approx_bits of a and aligned b.
    a_top = get_fixed_top_bits(a, approx_bits)
    b_top = get_fixed_top_bits(b_aligned, approx_bits)
    trace['a_top'] = a_top
    trace['b_top'] = b_top

    # Compute the approximate quotient using the LUT.
    quotient_lut = lut_result(a_top, b_top, approx_bits, rounding_mode)
    trace['quotient_lut'] = quotient_lut

    # Compute Q with shifting adjustments.
    Q_pre_round = (quotient_lut << shift_amount) >> (approx_bits - 1)
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and integer_rounding:
        Q += 1
    trace['Q_pre_round'] = Q_pre_round
    trace['Q'] = Q

    # Multiply b by Q and subtract from a.
    b_adjusted = b * Q
    trace['b_adjusted'] = b_adjusted
    residual = a - b_adjusted
    if residual < 0:
        residual = -residual
    trace['residual'] = residual

    msb_a = bit_length(a)
    msb_res = bit_length(residual)
    bit_clears = msb_a - msb_res
    if bit_clears < 0:
        bit_clears = 0
    trace['msb_a'] = msb_a
    trace['msb_res'] = msb_res
    trace['bit_clears'] = bit_clears

    # Update rule: if residual > b then no swap, otherwise swap.
    if residual > b:
        new_a = residual
        new_b = b
        trace['update_rule'] = "non-swap"
    else:
        new_a = b
        new_b = residual
        trace['update_rule'] = "swap"
    trace['new_a'] = new_a
    trace['new_b'] = new_b

    return {
        'new_a': new_a,
        'new_b': new_b,
        'quotient': Q,
        'residual': residual,
        'bit_clears': bit_clears,
        'trace': trace
    }

###############################################################################
# Candidate Selection
###############################################################################

def choose_best(candidates):
    """
    Given a list of candidate dictionaries (each with a 'bit_clears' key),
    choose the candidate(s) with minimal bit clears.
    In the case of a tie, choose one at random.
    """
    if not candidates:
        return None
    min_clears = min(c['bit_clears'] for c in candidates)
    best_candidates = [c for c in candidates if c['bit_clears'] == min_clears]
    return random.choice(best_candidates)

###############################################################################
# Worst-Case Candidate Generation via Bit-by-Bit Extension
###############################################################################

def generate_worst_case_candidate(target_bits=16, approx_bits=4, bits_per_iter=2, 
                                  rounding_mode='truncate', integer_rounding=True):
    """
    Generate a worst-case candidate pair (a, b) for the xgcd algorithm by
    extending a seed pair bit-by-bit.
    
    Process:
      1. Start with all valid seed pairs of approx_bits-bit numbers (MSB = 1, a >= b).
      2. For each seed, simulate one xgcd iteration.
      3. Select the seed with the lowest “bit clears” (tie-break randomly).
      4. Then, while the current fixed bit-width is less than target_bits:
           a. Extend the current candidate by appending bits_per_iter new bits to a and b.
           b. Try all possible appended bit combinations (2^(2*bits_per_iter) possibilities).
           c. For each extension, simulate one xgcd iteration.
           d. Choose the extension that yields the lowest bit clears.
           e. Update the candidate pair (and record the extension in a trace).
      5. Return the final candidate pair and the full trace.
    """
    trace_history = []
    # --- Step 1: Seed Domain ---
    initial_candidates = []
    for a_val in range(2**(approx_bits-1), 2**approx_bits):
        for b_val in range(2**(approx_bits-1), 2**approx_bits):
            if a_val < b_val:
                continue  # enforce a >= b
            sim = simulate_xgcd_iteration(a_val, b_val, approx_bits, rounding_mode, integer_rounding)
            candidate = {
                'a': a_val,
                'b': b_val,
                'new_a': sim['new_a'],
                'new_b': sim['new_b'],
                'bit_clears': sim['bit_clears'],
                'quotient': sim['quotient'],
                'trace': [ { 'a': a_val, 'b': b_val, 'sim': sim['trace'] } ]
            }
            initial_candidates.append(candidate)
    best = choose_best(initial_candidates)
    # Use the candidate’s updated pair as the current candidate.
    current_a = best['new_a']
    current_b = best['new_b']
    current_width = approx_bits
    trace_history.extend(best['trace'])
    print(f"Initial seed chosen: a = {current_a} ({current_a:b}), b = {current_b} ({current_b:b}), "
          f"width = {current_width} bits, bit_clears = {best['bit_clears']}, quotient = {best['quotient']}\n")
    
    iteration = 1
    # --- Step 2: Bit-by-Bit Extension ---
    while current_width < target_bits:
        # Determine how many bits to append in this iteration.
        ext_bits = bits_per_iter if current_width + bits_per_iter <= target_bits else target_bits - current_width
        extension_candidates = []
        # Try all possible appended bit combinations for a and b.
        for a_ext in range(2**ext_bits):
            for b_ext in range(2**ext_bits):
                # Extend the current candidate by shifting left and OR-ing the new bits.
                candidate_a = (current_a << ext_bits) | a_ext
                candidate_b = (current_b << ext_bits) | b_ext
                # Enforce a >= b (if not, swap them)
                if candidate_b > candidate_a:
                    candidate_a, candidate_b = candidate_b, candidate_a
                sim = simulate_xgcd_iteration(candidate_a, candidate_b, approx_bits, rounding_mode, integer_rounding)
                extension_candidates.append({
                    'a': candidate_a,
                    'b': candidate_b,
                    'new_a': sim['new_a'],
                    'new_b': sim['new_b'],
                    'bit_clears': sim['bit_clears'],
                    'quotient': sim['quotient'],
                    'extension': {'a_ext': a_ext, 'b_ext': b_ext, 'ext_bits': ext_bits},
                    'trace': {'iteration': iteration,
                              'current_width': current_width,
                              'candidate_a': candidate_a,
                              'candidate_b': candidate_b,
                              'sim': sim['trace']}
                })
        best_ext = choose_best(extension_candidates)
        # Update the current candidate pair.
        current_a = best_ext['new_a']
        current_b = best_ext['new_b']
        current_width += ext_bits
        trace_history.append(best_ext['trace'])
        print(f"Iteration {iteration}: Extended by {ext_bits} bits to width {current_width}")
        print(f"  Chosen extension: a_ext = {best_ext['extension']['a_ext']:0{ext_bits}b}, "
              f"b_ext = {best_ext['extension']['b_ext']:0{ext_bits}b}")
        print(f"  Candidate a = {best_ext['a']} ({best_ext['a']:0{current_width}b})")
        print(f"  Candidate b = {best_ext['b']} ({best_ext['b']:0{current_width}b})")
        print(f"  After simulation: new_a = {best_ext['new_a']} ({best_ext['new_a']:0{current_width}b}), "
              f"new_b = {best_ext['new_b']} ({best_ext['new_b']:0{current_width}b})")
        print(f"  Bit clears = {best_ext['bit_clears']}, Quotient = {best_ext['quotient']}\n")
        iteration += 1

    return current_a, current_b, trace_history

###############################################################################
# The XGCD Bitwise Algorithm (as provided)
###############################################################################

def xgcd_bitwise(a_in, b_in, total_bits=16, approx_bits=4, rounding_mode='truncate', 
                 integer_rounding=True, plus_minus=False, enable_plotting=False):
    """
    Compute the GCD of a_in and b_in using the custom XGCD bitwise approach.
    
    Returns:
      (gcd_value, iteration_count, avg_bit_clears)
    """
    # Ensure inputs are masked to total_bits.
    mask = (1 << total_bits) - 1
    a = a_in & mask
    b = b_in & mask
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
        b_aligned, shift_amount = align_b(a, b)
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b_aligned, approx_bits)
        quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
        Q_pre_round = (quotient << shift_amount) >> (approx_bits-1)
        Q = Q_pre_round >> 1
        if (Q_pre_round & 1) and integer_rounding:
            Q += 1
        b_adjusted = b * Q
        residual = a - b_adjusted
        if residual < 0:
            residual = -residual
        msb_a = bit_length(a)
        msb_res = bit_length(residual)
        clears_this_iter = msb_a - msb_res
        if clears_this_iter < 0:
            clears_this_iter = 0
        bit_clears_list.append(clears_this_iter)
        if residual > b:
            a = residual
        else:
            a, b = b, residual
    bit_clears_list[-1] += bit_length(a)  # final adjustment
    avg_bit_clears = sum(bit_clears_list) / iteration_count if iteration_count > 0 else 0.0
    return (a, iteration_count, avg_bit_clears)

###############################################################################
# Main – Run Candidate Generation and Test XGCD
###############################################################################

if __name__ == "__main__":
    # Parameters for worst-case generation.
    TARGET_BITS = 16         # Final numbers will be 16-bit wide.
    APPROX_BITS = 4          # Use 4 bits in the fixed-point (top bits).
    BITS_PER_ITER = 2        # Append 2 bits per extension iteration.
    ROUNDING_MODE = 'truncate'  # Can also be 'round'
    INTEGER_ROUNDING = True

    print("Generating worst-case candidate pair for XGCD bitwise algorithm...\n")
    worst_a, worst_b, trace = generate_worst_case_candidate(
        target_bits=TARGET_BITS,
        approx_bits=APPROX_BITS,
        bits_per_iter=BITS_PER_ITER,
        rounding_mode=ROUNDING_MODE,
        integer_rounding=INTEGER_ROUNDING
    )
    print("Final worst-case candidate pair generated:")
    print(f"  a = {worst_a} ({worst_a:0{TARGET_BITS}b})")
    print(f"  b = {worst_b} ({worst_b:0{TARGET_BITS}b})\n")

    print("Running xgcd_bitwise on the candidate pair...\n")
    gcd_val, iter_count, avg_clears = xgcd_bitwise(
        worst_a, worst_b,
        total_bits=TARGET_BITS,
        approx_bits=APPROX_BITS,
        rounding_mode=ROUNDING_MODE,
        integer_rounding=INTEGER_ROUNDING,
        plus_minus=False,
        enable_plotting=False
    )
    print(f"Result: GCD = {gcd_val}, Iterations = {iter_count}, Average Bit Clears = {avg_clears:.3f}")
