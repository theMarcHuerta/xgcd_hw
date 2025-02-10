#!/usr/bin/env python3
"""
Generative Worst-Case Pair Generator for XGCD-style Algorithm

This script attempts to build an n-bit candidate pair (e.g. 16-bit numbers)
by extending a seed pair (normalized in a small domain, e.g. 4 bits in [8,15])
one bit at a time. At each extension step, it tests all 4 possibilities for the next bit
(for both a and b), simulates one xgcd-style step (using a LUT quotient method),
and scores the candidate extension by:
   - The number of bits cleared (we want as little progress as possible)
   - Tie-breaker: the absolute error of the candidate’s float quotient from 1.5
The candidate with the lowest score (in lexicographic order) is chosen.
After extending to full width, a simplified xgcd simulation is run to count iterations.

Parameters:
  - total_bits = 16  (target full precision)
  - approx_bits = 4  (seed domain; numbers in [8,15])
  - rounding_mode = "truncate", integer_rounding = True

Run with: python3 one_bit_wc_gen.py
"""

import math
import sys

# ---------------------------
# Helper Functions
# ---------------------------

def bit_length(x):
    """Return the bit length of x (ignoring leading zeros)."""
    return 0 if x == 0 else x.bit_length()

def align_b(a_val, b_val):
    """
    Shift b_val left so that its MSB aligns with that of a_val.
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
    Extract the top `approx_bits` bits of x_val.
    If x_val has fewer than approx_bits bits, left-shift it so that it has exactly approx_bits.
    """
    if x_val == 0:
        return 0
    L = bit_length(x_val)
    if L <= approx_bits:
        return x_val << (approx_bits - L)
    else:
        return x_val >> (L - approx_bits)

def lut_result(a_top, b_top, approx_bits, rounding_mode):
    """
    Compute the ratio (a_top / b_top) in fixed-point with approx_bits fractional bits.
    """
    if b_top == 0:
        return 0  # safeguard
    numerator = a_top << approx_bits
    if rounding_mode == "round":
        return (numerator + (b_top >> 1)) // b_top
    else:
        return numerator // b_top

def normalize(x, bits):
    """
    Force x into the normalized range for a given bit-width.
    For bits=4, ensure x is in [2^(4-1), 2^4 - 1] = [8,15].
    """
    mask = (1 << bits) - 1
    x = x & mask
    if x < (1 << (bits - 1)):
        x |= (1 << (bits - 1))
    return x

# ---------------------------
# Simulate One XGCD-Style Step
# ---------------------------
def simulate_xgcd_step(a, b, approx_bits, rounding_mode, integer_rounding=True):
    """
    Simulate one xgcd-style step on (a, b):
      1. Align b to a.
      2. Compute a_top and b_top using approx_bits.
      3. Compute the quotient using LUT.
      4. Compute Q via shifting and (if necessary) integer rounding.
      5. Compute b_adjusted = b * Q, then residual = |a - b_adjusted|.
      6. Compute clears = bit_length(a) - bit_length(residual) (clamped to 0).
    Returns (Q, residual, clears, shift_amount).
    """
    b_aligned, shift_amount = align_b(a, b)
    a_top = get_fixed_top_bits(a, approx_bits)
    b_top = get_fixed_top_bits(b_aligned, approx_bits)
    quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
    Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1)
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and integer_rounding:
        Q += 1
    b_adjusted = b * Q
    residual = a - b_adjusted
    if residual < 0:
        residual = -residual
    clears = bit_length(a) - bit_length(residual)
    if clears < 0:
        clears = 0
    return Q, residual, clears, shift_amount

# ---------------------------
# Generative Step: Extend Candidate by One Bit
# ---------------------------
def generative_step(a, b, current_bits, total_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Extend candidate a and b (which currently have current_bits bits) by one extra bit.
    For each combination:
       a_candidate = (a << 1) | a_bit
       b_candidate = (b << 1) | b_bit
    Then normalize each candidate to the new bit-width (current_bits+1).
    Simulate one xgcd step on (a_candidate_norm, b_candidate_norm) and obtain:
         - Integer quotient Q,
         - Residual,
         - Bit clears.
    Also compute a floating-point version of the quotient (float_q) as follows:
         Let a_top = get_fixed_top_bits(a_candidate_norm, approx_bits)
         Let b_aligned, _ = align_b(a_candidate_norm, b_candidate_norm)
         Let b_top = get_fixed_top_bits(b_aligned, approx_bits)
         Then quotient_raw = (a_top * (2^approx_bits)) / b_top  (as float),
         and then float_q = (quotient_raw * (2^(shift))) / (2^(approx_bits-1)) / 2.0.
    We score the candidate by the tuple (clears, |float_q - 1.5|); lower is better.
    Return the candidate (new_a, new_b, Q, residual, clears, shift) that minimizes the score,
    and also return the chosen bits (a_bit, b_bit).
    """
    best_candidate = None
    best_score = None
    best_bits = None
    # We assume we extend both a and b as long as current_bits < total_bits.
    for a_bit in [0, 1]:
        a_candidate = (a << 1) | a_bit
        for b_bit in [0, 1]:
            b_candidate = (b << 1) | b_bit
            new_width = current_bits + 1
            a_candidate_norm = normalize(a_candidate, new_width)
            b_candidate_norm = normalize(b_candidate, new_width)
            Q, residual, clears, shift = simulate_xgcd_step(a_candidate_norm, b_candidate_norm, approx_bits, rounding_mode, integer_rounding)
            # Compute a float quotient for tiebreaker:
            a_top = get_fixed_top_bits(a_candidate_norm, approx_bits)
            b_aligned, _ = align_b(a_candidate_norm, b_candidate_norm)
            b_top = get_fixed_top_bits(b_aligned, approx_bits)
            quotient_raw = (a_top * (1 << approx_bits)) / b_top  # float value from LUT perspective
            # Mimic the integer shifting (we use the same shift as computed above):
            float_q = (quotient_raw * (1 << shift)) / (1 << (approx_bits - 1)) / 2.0
            error = abs(float_q - 1.5)
            score = (clears, error)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (a_candidate_norm, b_candidate_norm, Q, residual, clears, shift)
                best_bits = (a_bit, b_bit)
    return best_candidate, best_bits

# ---------------------------
# Update Rule: Update and Normalize
# ---------------------------
def update_pair(a, b, Q, residual, new_width):
    """
    Update the pair (a, b) using the rule:
         if residual > b: new_a = residual, new_b = b   (non-swap)
         else:            new_a = b, new_b = residual   (swap)
    Then normalize both to new_width.
    Returns (new_a, new_b, update_type).
    """
    if residual > b:
        new_a = residual
        new_b = b
        update_type = "non-swap"
    else:
        new_a = b
        new_b = residual
        update_type = "swap"
    new_a = normalize(new_a, new_width)
    new_b = normalize(new_b, new_width)
    return new_a, new_b, update_type

# ---------------------------
# Full Generative Process: Build Worst-Case Pair Bit-by-Bit
# ---------------------------
def generate_candidate(total_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Starting from a seed pair in the approx_bits domain (e.g. 4 bits),
    extend a and b one bit at a time until they have total_bits.
    At each extension, try all 4 possibilities for the new bit pair,
    simulate one xgcd step, and choose the candidate with the minimal score,
    where score = (clears, |float_q - 1.5|).
    Then update the candidate pair using the update rule and normalize to the new width.
    Return the final candidate pair and the generative path (list of decisions).
    """
    # Seed pair: choose a = 9 (0b1001) and b = 8 (0b1000) in 4 bits.
    seed_bits = approx_bits
    seed_a = normalize(9, seed_bits)
    seed_b = normalize(8, seed_bits)
    a = seed_a
    b = seed_b
    current_bits = seed_bits
    generative_path = []
    
    while current_bits < total_bits:
        candidate_info, chosen_bits = generative_step(a, b, current_bits, total_bits, approx_bits, rounding_mode, integer_rounding)
        new_width = current_bits + 1
        new_a_candidate, new_b_candidate, Q, residual, clears, shift = candidate_info
        new_a, new_b, update_type = update_pair(new_a_candidate, new_b_candidate, Q, residual, new_width)
        generative_path.append({
            'current_bits': current_bits,
            'chosen_bits': chosen_bits,   # (a_bit, b_bit)
            'candidate_info': candidate_info,
            'update_type': update_type
        })
        a, b = new_a, new_b
        current_bits = new_width
    return a, b, generative_path

# ---------------------------
# Full XGCD Simulation (Simplified)
# ---------------------------
def full_xgcd(a, b, total_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Run a simplified xgcd-style algorithm (similar to your earlier version)
    on full-width numbers (masked to total_bits).
    Returns (gcd, iteration_count, avg_bit_clears).
    """
    mask = (1 << total_bits) - 1
    a = a & mask
    b = b & mask
    if b > a:
        a, b = b, a
    iteration_count = 0
    total_clears = 0
    while b != 0:
        iteration_count += 1
        b_aligned, shift = align_b(a, b)
        a_top = get_fixed_top_bits(a, approx_bits)
        b_top = get_fixed_top_bits(b_aligned, approx_bits)
        quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
        Q_pre_round = (quotient << shift) >> (approx_bits - 1)
        Q = Q_pre_round >> 1
        if (Q_pre_round & 1) and integer_rounding:
            Q += 1
        b_adjusted = b * Q
        residual = a - b_adjusted
        if residual < 0:
            residual = -residual
        clears = bit_length(a) - bit_length(residual)
        if clears < 0:
            clears = 0
        total_clears += clears
        if residual > b:
            a = residual
        else:
            a, b = b, residual
    avg_clears = total_clears / iteration_count if iteration_count else 0
    return a, iteration_count, avg_clears

# ---------------------------
# Main Routine
# ---------------------------
if __name__ == "__main__":
    total_bits = 16          # target full width (16-bit numbers)
    approx_bits = 4          # seed domain: 4 bits (numbers in [8,15])
    rounding_mode = "truncate"
    integer_rounding = True

    print("Generating candidate worst-case pair for {}-bit numbers (seed domain: {} bits)...".format(total_bits, approx_bits))
    candidate_a, candidate_b, gen_path = generate_candidate(total_bits, approx_bits, rounding_mode, integer_rounding)
    print("Generated candidate (from generative process):")
    print("  a = {}  (binary: {})".format(candidate_a, bin(candidate_a)))
    print("  b = {}  (binary: {})".format(candidate_b, bin(candidate_b)))
    if candidate_b != 0:
        print("  ratio = {:.3f}".format(candidate_a / candidate_b))
    else:
        print("  b is 0!")
    print("Generative path (bit choices per extension):")
    for step in gen_path:
        a_bit, b_bit = step['chosen_bits']
        b_str = str(b_bit) if b_bit is not None else "(no ext)"
        print("  At {} bits, chosen bits: a_bit = {}, b_bit = {}, update = {}".format(
            step['current_bits'] + 1, a_bit, b_str, step['update_type']))
    
    # Run full xgcd simulation on candidate:
    gcd_val, iter_count, avg_clears = full_xgcd(candidate_a, candidate_b, total_bits, approx_bits, rounding_mode, integer_rounding)
    print("\nFull xgcd simulation on candidate:")
    print("  GCD = {}".format(gcd_val))
    print("  Iteration count = {}".format(iter_count))
    print("  Average bit clears per iteration = {:.3f}".format(avg_clears))
    
    print("\n(For reference, brute-force worst-case for 16 bits is around 15 iterations.)")
