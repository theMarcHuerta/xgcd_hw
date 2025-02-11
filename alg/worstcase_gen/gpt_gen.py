#!/usr/bin/env python3
"""
Generative Worst-Case Pair Generator for an Approximate XGCD Algorithm
(Targeting a specified quotient DESIRED_Q)

This script extends a seed pair one bit at a time (forcing normalization to
the range [2^(n-1), 2^n-1]) and at each extension:
  - Tries all 4 combinations for the new bit (for a and b),
  - Simulates one xgcd-style step using your fixed‑point LUT method,
  - Computes the integer quotient Q from that simulation,
  - Scores the candidate extension by how close Q is to DESIRED_Q (with a penalty
    if both candidate numbers are even, which would build up a power‑of‑2 factor),
  - Picks the best candidate extension and immediately “updates” the pair using the rule:
        if residual > b: (a_new, b_new) = (residual, b)
        else:            (a_new, b_new) = (b, residual)
  - Finally, common factors of 2 are removed so that the candidate will be coprime.
  
After building up to TOTAL_BITS, the candidate is fed into a full xgcd simulation.
(Recall that worst‑case Euclid inputs are nearly always coprime.)
"""

import math

##########################
# PARAMETERS
##########################

TOTAL_BITS      = 16          # Target bit-width (e.g. 16-bit numbers)
APPROX_BITS     = 4           # Seed domain bit‐width (numbers in [8,15])
ROUNDING_MODE   = "truncate"  # or "round"
INTEGER_ROUNDING = True
DESIRED_Q       = 1.436       # Target quotient for each simulated xgcd step

##########################
# HELPER FUNCTIONS
##########################

def bit_length(x):
    return x.bit_length() if x else 0

def normalize(x, bits):
    """
    Force x into the normalized range for the given bit‑width.
    (That is, ensure x is in [2^(bits-1), 2^bits–1].)
    """
    mask = (1 << bits) - 1
    x = x & mask
    if x < (1 << (bits - 1)):
        x |= (1 << (bits - 1))
    return x

def align_b(a_val, b_val):
    """
    Shift b_val left so its MSB lines up with that of a_val.
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
    (This is standard in Euclid algorithms to avoid building in extraneous factors.)
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
    Simulate one iteration of the approximate XGCD step for the candidate pair (a,b).
    Returns:
      Q: the integer quotient computed by the fixed‑point method,
      residual: |a - Q * b|,
      clears: the difference in bit‑length between a and the residual,
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
# GENERATIVE STEP (Extend One Bit)
##########################

def generative_step(a, b, current_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Extend the candidate pair (a,b) (currently with current_bits bits) by one extra bit.
    For each candidate extension:
      - Compute:
            a_candidate = (a << 1) | a_bit,
            b_candidate = (b << 1) | b_bit,
        and normalize both to new_width bits.
      - Simulate one xgcd step to obtain Q, residual, and bit clears.
      - Score the candidate using:
            score = abs(Q - DESIRED_Q)
      and add a penalty if (a_candidate, b_candidate) are both even.
      - Also, if residual is zero (i.e. the step “finishes”), assign a huge penalty.
    Returns the best candidate extension (as a tuple containing the candidate pair, Q, residual, etc.)
    along with the chosen extension bits.
    """
    best_candidate = None
    best_score = None
    best_bits = None
    new_width = current_bits + 1
    for a_bit in [0, 1]:
        a_candidate = (a << 1) | a_bit
        for b_bit in [0, 1]:
            b_candidate = (b << 1) | b_bit
            a_candidate_norm = normalize(a_candidate, new_width)
            b_candidate_norm = normalize(b_candidate, new_width)
            Q, residual, clears, shift = simulate_xgcd_step(a_candidate_norm,
                                                              b_candidate_norm,
                                                              approx_bits,
                                                              rounding_mode,
                                                              integer_rounding)
            if residual == 0:
                score_val = 1e9  # huge penalty if the step would finish immediately
            else:
                score_val = abs(Q - DESIRED_Q)
            # Penalize if both candidate numbers are even (to avoid building a common factor)
            if (a_candidate_norm % 2 == 0) and (b_candidate_norm % 2 == 0):
                score_val += 1.0
            score = (score_val, clears)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (a_candidate_norm, b_candidate_norm, Q, residual, clears, shift)
                best_bits = (a_bit, b_bit)
    return best_candidate, best_bits

##########################
# UPDATE RULE
##########################

def update_pair(a, b, Q, residual, new_width):
    """
    Update the candidate pair using:
         if residual > b: new_a = residual, new_b = b   (non‑swap)
         else:            new_a = b, new_b = residual   (swap)
    Then normalize both numbers to new_width and remove any common factors of 2.
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
    new_a, new_b = remove_common_power_of_2(new_a, new_b)
    return new_a, new_b, update_type

##########################
# FULL GENERATIVE PROCESS
##########################

def generate_candidate(total_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Starting from a seed pair (with approx_bits bits) extend a and b one bit at a time
    (using the generative_step) until they reach total_bits.
    Returns the final candidate pair and a record of the generative path.
    """
    # Seed pair: for example, use a=9 (binary 1001) and b=8 (binary 1000) in the seed domain.
    seed_bits = approx_bits
    a = normalize(9, seed_bits)
    b = normalize(8, seed_bits)
    current_bits = seed_bits
    generative_path = []
    
    while current_bits < total_bits:
        candidate_info, chosen_bits = generative_step(a, b, current_bits, approx_bits, rounding_mode, integer_rounding)
        new_width = current_bits + 1
        a_candidate_norm, b_candidate_norm, Q, residual, clears, shift = candidate_info
        a, b, up_type = update_pair(a_candidate_norm, b_candidate_norm, Q, residual, new_width)
        generative_path.append({
            'current_bits': current_bits,
            'chosen_bits': chosen_bits,
            'candidate_info': candidate_info,
            'update_type': up_type
        })
        current_bits = new_width
    return a, b, generative_path

##########################
# FULL XGCD SIMULATION
##########################

def full_xgcd(a, b, total_bits, approx_bits, rounding_mode, integer_rounding=True):
    """
    Run the full approximate xgcd simulation (using your fixed‑point approach)
    on the candidate pair (a,b) masked to total_bits.
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
    print("Target quotient (DESIRED_Q) =", DESIRED_Q)
    candidate_a, candidate_b, gen_path = generate_candidate(TOTAL_BITS, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING)
    
    print("\nGenerated candidate (from generative process):")
    print("  A =", candidate_a, " (binary:", format(candidate_a, '0{}b'.format(TOTAL_BITS)) + ")")
    print("  B =", candidate_b, " (binary:", format(candidate_b, '0{}b'.format(TOTAL_BITS)) + ")")
    if candidate_b != 0:
        print("  Ratio =", candidate_a / candidate_b)
    else:
        print("  B is 0!")
    
    print("\nGenerative path (bit choices per extension):")
    for step in gen_path:
        a_bit, b_bit = step['chosen_bits']
        A_cand, B_cand, Q, residual, clears, shift = step['candidate_info']
        print("  At {}->{} bits: chosen bits (a_bit={}, b_bit={}), Q={}, residual={}, clears={}, update={}".format(
            step['current_bits'], step['current_bits']+1, a_bit, b_bit, Q, residual, clears, step['update_type']))
    
    gcd_val, iter_count, avg_clears = full_xgcd(candidate_a, candidate_b, TOTAL_BITS, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING)
    print("\nFinal XGCD simulation results:")
    print("  GCD =", gcd_val)
    print("  Iterations =", iter_count)
    print("  Avg bit clears =", avg_clears)
    print("\n(Note: Worst-case inputs for Euclid are typically coprime (GCD=1), so check the iteration count.)")
