#!/usr/bin/env python3
"""
Lookahead Generative Worst-Case Candidate Generator for Approximate XGCD

This algorithm builds an n‑bit candidate pair (for example, 16‑bit numbers)
by starting from a seed pair (in a small “approx_bits” domain) and then extending
one bit at a time. Instead of choosing the next bit greedily, it uses a recursive
lookahead (future steps) to branch out over all possible bit choices for the next
few steps. The branch that yields the smallest total “bit clears” (i.e. minimal progress)
is chosen. Finally, the candidate pair is updated (using the xgcd update rule) and the
process continues until the desired bit width is reached.

You can control the lookahead depth via FUTURE_DEPTH.
"""

import math
import random

##########################
# PARAMETERS
##########################

TOTAL_BITS      = 16          # Target bit-width (e.g. 16-bit numbers)
APPROX_BITS     = 4           # Seed domain bit-width (numbers in [2^(APPROX_BITS-1), 2^APPROX_BITS - 1])
ROUNDING_MODE   = "truncate"  # or "round"
INTEGER_ROUNDING = True
# FUTURE_DEPTH controls how many future steps to look ahead (e.g. 2 means look ahead 2 steps)
FUTURE_DEPTH    = 3

##########################
# HELPER FUNCTIONS
##########################

def bit_length(x):
    return x.bit_length() if x else 0

def normalize(x, bits):
    """
    Force x into the normalized range for the given bit‑width.
    That is, ensure x is in [2^(bits-1), 2^bits – 1].
    """
    mask = (1 << bits) - 1
    x = x & mask
    if x < (1 << (bits - 1)):
        x |= (1 << (bits - 1))
    return x

def align_b(a_val, b_val):
    """
    Shift b_val left so its MSB aligns with that of a_val.
    Returns (b_shifted, shift_amount).
    """
    la = bit_length(a_val)
    lb = bit_length(b_val)
    shift_amount = la - lb
    if shift_amount > 0:
        return (b_val << shift_amount), shift_amount
    else:
        return b_val, 0

def get_fixed_top_bits(x_val, approx_bits):
    """
    Extract the top approx_bits bits of x_val.
      - If x_val has fewer than approx_bits bits, shift left.
      - Otherwise, shift right so that only the top approx_bits remain.
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
    Compute (a_top << approx_bits) / b_top using either truncation or rounding.
    """
    if b_top == 0:
        return 0
    numerator = a_top << approx_bits
    if rounding_mode == "round":
        return (numerator + (b_top >> 1)) // b_top
    else:
        return numerator // b_top

def remove_common_power_of_2(a, b):
    """
    Remove all common factors of 2 from a and b.
    """
    while a and b and (a % 2 == 0) and (b % 2 == 0):
        a //= 2
        b //= 2
    return a, b

##########################
# XGCD STEP SIMULATION
##########################

def simulate_xgcd_step(a, b, approx_bits, rounding_mode, integer_rounding=True):
    """
    Simulate one iteration of the approximate XGCD step for candidate pair (a, b).
    Returns:
      Q: the integer quotient computed,
      residual: |a - Q*b|,
      clears: the difference in bit-length (a vs. residual),
      shift_amount: from aligning b to a.
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

##########################
# UPDATE RULE
##########################

def update_pair(a, b, Q, residual, new_width):
    """
    Update the candidate pair using the rule:
      if residual > b: new_a = residual, new_b = b (non-swap)
      else: new_a = b, new_b = residual (swap)
    Then normalize to new_width and remove common factors of 2.
    """
    if residual > b:
        new_a = residual
        new_b = b
        up_type = "non-swap"
    else:
        new_a = b
        new_b = residual
        up_type = "swap"
    new_a = normalize(new_a, new_width)
    new_b = normalize(new_b, new_width)
    new_a, new_b = remove_common_power_of_2(new_a, new_b)
    return new_a, new_b, up_type

##########################
# LOOKAHEAD EXTENSION (RECURSIVE)
##########################

def lookahead_extend(a, b, current_bits, depth, approx_bits, rounding_mode, integer_rounding):
    """
    Recursively simulate extensions for 'depth' future steps.
    
    For the current candidate pair (a, b) at width = current_bits, try every possible
    extension (4 possibilities) by adding one new bit (for a and b), then:
      - Normalize the candidate pair to width = current_bits+1.
      - Simulate one xgcd step (which returns Q, residual, and clears).
      - (If residual==0, assign a huge penalty.)
      - Update the candidate pair via update_pair.
      - If depth > 1 and we haven't reached TOTAL_BITS, recursively call lookahead_extend
        for depth-1 with the updated candidate.
      - The total score for a branch is the immediate clears plus the recursive score.
    
    Returns a tuple (score, branch, candidate) where branch is a list of (a_bit, b_bit)
    choices made in each step and candidate is the final candidate pair reached.
    """
    # Base case: if depth==0 or no room to extend, return zero score.
    if depth == 0 or current_bits >= TOTAL_BITS:
        return 0, [], (a, b)
    
    best_score = None
    best_branch = None
    best_candidate = None
    new_width = current_bits + 1
    for a_bit in [0, 1]:
        for b_bit in [0, 1]:
            # Extend candidate by one bit
            a_candidate = (a << 1) | a_bit
            b_candidate = (b << 1) | b_bit
            a_candidate_norm = normalize(a_candidate, new_width)
            b_candidate_norm = normalize(b_candidate, new_width)
            # Simulate one xgcd step on this candidate:
            Q, residual, clears, shift = simulate_xgcd_step(a_candidate_norm, b_candidate_norm, approx_bits, rounding_mode, integer_rounding)
            # If the residual is 0 (i.e. the algorithm would finish), assign a high penalty:
            if residual == 0:
                immediate_score = 1e9
            else:
                immediate_score = clears  # immediate metric: how many bits got cleared
            # Update candidate pair using the rule:
            a_updated, b_updated, up_type = update_pair(a_candidate_norm, b_candidate_norm, Q, residual, new_width)
            # Recurse if we still have lookahead depth and room to extend:
            if depth - 1 > 0 and new_width < TOTAL_BITS:
                rec_score, rec_branch, rec_candidate = lookahead_extend(a_updated, b_updated, new_width, depth - 1, approx_bits, rounding_mode, integer_rounding)
            else:
                rec_score = 0
                rec_branch = []
                rec_candidate = (a_updated, b_updated)
            total_score = immediate_score + rec_score
            branch = [(a_bit, b_bit)] + rec_branch
            if best_score is None or total_score < best_score:
                best_score = total_score
                best_branch = branch
                best_candidate = rec_candidate
    return best_score, best_branch, best_candidate

##########################
# FULL GENERATIVE PROCESS
##########################

def generate_candidate(total_bits, approx_bits, rounding_mode, integer_rounding, future_depth):
    """
    Starting from a seed pair (in the approx_bits domain), extend the candidate pair
    one bit at a time until reaching total_bits. At each step, perform a lookahead
    extension of depth 'future_depth' (if possible) to choose the branch that minimizes
    the total bit clears.
    
    Returns the final candidate pair and a generative path (list of branch choices per step).
    """
    # Choose a seed pair. (For now, we simply use a=9 and b=8 normalized to seed_bits.)
    seed_bits = approx_bits
    a = normalize(9, seed_bits)
    b = normalize(8, seed_bits)
    current_bits = seed_bits
    generative_path = []  # record each step's chosen branch
    while current_bits < total_bits:
        # Use lookahead to decide on the best branch from here.
        score, branch, candidate_from_lookahead = lookahead_extend(a, b, current_bits, future_depth, approx_bits, rounding_mode, integer_rounding)
        # Apply the branch step by step:
        for (a_bit, b_bit) in branch:
            current_bits += 1
            a = (a << 1) | a_bit
            b = (b << 1) | b_bit
            a = normalize(a, current_bits)
            b = normalize(b, current_bits)
            Q, residual, clears, shift = simulate_xgcd_step(a, b, approx_bits, rounding_mode, integer_rounding)
            a, b, up_type = update_pair(a, b, Q, residual, current_bits)
        generative_path.append({
            'branch': branch,
            'score': score,
            'current_bits': current_bits
        })
    return a, b, generative_path

##########################
# FULL XGCD SIMULATION
##########################

def full_xgcd(a, b, total_bits, approx_bits, rounding_mode, integer_rounding):
    """
    Run the full approximate XGCD simulation on candidate pair (a,b) masked to total_bits.
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

##########################
# MAIN
##########################

if __name__ == "__main__":
    print("Generating candidate worst-case pair for {}-bit numbers (seed domain: {} bits)".format(TOTAL_BITS, APPROX_BITS))
    print("Using FUTURE_DEPTH =", FUTURE_DEPTH)
    candidate_a, candidate_b, gen_path = generate_candidate(TOTAL_BITS, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING, FUTURE_DEPTH)
    
    print("\nGenerated candidate (from generative process):")
    print("  A =", candidate_a, " (binary:", format(candidate_a, '0{}b'.format(TOTAL_BITS)) + ")")
    print("  B =", candidate_b, " (binary:", format(candidate_b, '0{}b'.format(TOTAL_BITS)) + ")")
    if candidate_b != 0:
        print("  Ratio =", candidate_a / candidate_b)
    else:
        print("  B is 0!")
    
    print("\nGenerative path (lookahead branch chosen per step):")
    for step in gen_path:
        print("  At {} bits, branch choices: {} with score {}".format(step['current_bits'], step['branch'], step['score']))
    
    gcd_val, iter_count, avg_clears = full_xgcd(candidate_a, candidate_b, TOTAL_BITS, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING)
    print("\nFinal XGCD simulation results:")
    print("  GCD =", gcd_val)
    print("  Iterations =", iter_count)
    print("  Avg bit clears =", avg_clears)
    print("\n(Note: Worst-case inputs for Euclid are typically coprime (GCD=1).)")
