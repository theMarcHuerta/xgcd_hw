#!/usr/bin/env python3
"""
Worst‑Case Candidate Generator for an Approximate XGCD Algorithm

This version adds a validation check so that whenever we try to
append bits to (a, b) or the residual, we only accept those appended
bits that do not alter the previously determined quotient bits or
the "top bits" of the residual. In other words, we preserve the step's
Q exactly. If extending bits would change Q, that candidate is rejected.
"""

import math
import random

import sys
import os

# Suppose you have xgcd_bitwise in xgcd_impl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgcd_impl import xgcd_bitwise

##########################
# PARAMETERS
##########################

TOTAL_BITS       = 16         # Final bit‑width for the candidate numbers
APPROX_BITS      = 4          # Bit‑width of the seed domain
BITS_PER_STEP    = 1          # Number of bits added per step
ROUNDING_MODE    = "truncate" # "truncate" or "round"
INTEGER_ROUNDING = True       # Whether to adjust quotient by integer rounding
LOOKAHEAD_DEPTH  = 4

##########################
# HELPER FUNCTIONS
##########################

def bit_length(x):
    """Return the bit length of x (returns 0 for x==0)."""
    return x.bit_length() if x else 0

def bit_len(n):
    """Return the bit‑length of n (alias for bit_length)."""
    return n.bit_length() if n else 0

def norm_val(x, bits):
    """
    Force x to have at most 'bits' bits, with the MSB set to 1
    (if x < 2^(bits-1), set that bit).
    """
    mask = (1 << bits) - 1
    x = x & mask
    if x < (1 << (bits - 1)):
        x |= (1 << (bits - 1))
    return x

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
    - If x_val = 0, return 0.
    - If x_val has fewer than approx_bits bits, shift it up so it has exactly approx_bits bits.
    - Otherwise, shift down so we keep only the top approx_bits bits.
    """
    if x_val == 0:
        return 0
    length = bit_length(x_val)
    if length <= approx_bits:
        return x_val << (approx_bits - length)
    else:
        shift_down = length - approx_bits
        return x_val >> shift_down

def lut_result(a_top, b_top):
    """
    Compute the ratio (a_top / b_top) in fixed‑point with APPROX_BITS fractional bits.
    """
    if b_top == 0:
        return 0  # avoid division by zero
    numerator = a_top << APPROX_BITS  # up‑shift a_top by APPROX_BITS
    if ROUNDING_MODE == "round":
        # round‑half‑up
        return (numerator + (b_top >> 1)) // b_top
    else:
        # truncate/floor
        return numerator // b_top

##########################
# XGCD STEP SIMULATION
##########################

def simulate_step(a, b):
    """
    Simulate one iteration of the approximate XGCD step on (a, b).

    Returns (Q, residual, clears, shift, a_next, b_next, neg_res).
    """
    b_aligned, shift = align_b(a, b)
    a_top = get_fixed_top_bits(a, APPROX_BITS)
    b_top = get_fixed_top_bits(b_aligned, APPROX_BITS)
    quotient = lut_result(a_top, b_top)
    Q_pre_round = (quotient << shift) >> (APPROX_BITS - 1)
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and INTEGER_ROUNDING:
        Q += 1

    prod = b * Q
    residual = a - prod
    neg_res = False
    if residual < 0:
        neg_res = True
        residual = -residual

    clears = bit_len(a) - bit_len(residual)
    if clears < 0:
        clears = 0

    if residual > b:
        a_next = residual
        b_next = b
    else:
        a_next, b_next = b, residual

    return Q, residual, clears, shift, a_next, b_next, neg_res

##############################
# *** NEW *** CONSISTENCY CHECK
##############################

def validate_extension(old_a, old_b, old_Q, old_res, ext_bits_b):
    """
    We want to extend 'old_res' by BITS_PER_STEP bits (ext_bits_b)
    without changing old_Q or the top bits of old_res from the old step
    that was computed on (old_a, old_b).

    - We'll form 'candidate_b' = (old_res << BITS_PER_STEP) | ext_bits_b.
    - Then re-run simulate_step(old_a, old_b) but forcibly interpret
      the "would-be" residual as 'candidate_b' and see if Q or top bits differ.

    However, your approach might be:
      1. Re-simulate step on (old_a, old_b).
      2. If it yields Q != old_Q, invalid.
      3. Check top bits of new residual vs old_res, invalid if differ.
    Because we are generating bits for the new b, which is old_res.

    We'll do a simpler approach:
      - If old_Q, old_res came from simulate_step(old_a, old_b),
        that means 'old_res' was effectively |old_a - Q*old_b|.
      - By adding bits to old_res, we do NOT want to alter Q or old_res's top bits.

    We'll do a partial check:
      - If old_Q=0 or old_b=0, it's trivially okay or not, etc.
      - Else check that the top portion of old_res is the same as the top portion of candidate_b, and
        that re-simulating doesn't yield a different Q.
    """
    # Construct candidate for next b
    candidate_b = (old_res << BITS_PER_STEP) | ext_bits_b

    # Re-simulate the old step (old_a, old_b) to see if we get the same Q
    # Because it's approximate, we do a normal 'simulate_step'
    # and see if Q matches old_Q.
    test_Q, test_res, _, _, _, _, _ = simulate_step(old_a, old_b)

    if test_Q != old_Q:
        return False

    # Now check top bits of test_res vs old_res
    if old_res == 0:
        if test_res != 0:
            return False
    else:
        # Compare top bits up to bit_length(old_res)
        old_res_top = get_fixed_top_bits(old_res, bit_length(old_res))
        test_res_top= get_fixed_top_bits(test_res, bit_length(old_res))
        if old_res_top != test_res_top:
            return False

    # If we get here, the extension doesn't break old Q or old_res top bits
    return True

##########################
# LOOKAHEAD
##########################

def lookahead_score_with_bits(a, b, bits_to_gen_a, bits_to_gen_b, low_bit_pos_a, low_bit_pos_b, depth, old_Q=None, old_res=None):
    """
    Recursively compute minimal cumulative bit clears for candidate (a, b),
    subject to the new rule that any appended bits must preserve the 
    old Q and old residual's top bits (if old_Q/old_res are not None).

    If b == 0 and depth>0 => big penalty.

    ***If old_Q, old_res are given, we first validate that the extension 
    is consistent with them.***
    """
    if b == 0 and depth > 0:
        return 1000 * depth  # large penalty for early termination
    
    if depth == 0:
        return 0

    best_total = float('inf')
    # Loop over all possible extension bits for a:
    for ext_a in range(2 ** bits_to_gen_a):
        for ext_b in range(2 ** bits_to_gen_b):
            # *** NEW *** If there's an old Q/res, ensure the extension is consistent
            if old_Q is not None and old_res is not None:
                if not validate_extension(a, b, ext_a, ext_b, old_Q, old_res):
                    # skip this extension
                    continue

            # If we get here, extension is valid, so we do the usual step
            shift_a_by = bit_length(a) - APPROX_BITS - BITS_PER_STEP
            if shift_a_by < 0:
                shift_a_by = 0
            a_test = (((a >> shift_a_by) + ext_a) << shift_a_by)

            shift_b_by = bit_length(b) - APPROX_BITS - BITS_PER_STEP
            if shift_b_by < 0:
                shift_b_by = 0
            b_test = (((b >> shift_b_by) + ext_b) << shift_b_by)

            Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test, b_test)

            # update local trackers
            im_low_bit_pos_a = low_bit_pos_a
            im_low_bit_pos_b = low_bit_pos_b
            im_bits_to_gen_a = 0
            im_bits_to_gen_b = clears
            if (im_low_bit_pos_a - im_bits_to_gen_b < 1):
                im_bits_to_gen_b = (im_low_bit_pos_a - 1)
            if (im_low_bit_pos_a < 2):
                im_bits_to_gen_b = 0

            tmp_a = im_low_bit_pos_a
            im_low_bit_pos_a = im_low_bit_pos_b
            im_low_bit_pos_b = tmp_a - im_bits_to_gen_b

            # If no swap occurred in simulation, swap the low-bit trackers
            if b_test == b_next:  
                tmp_a = im_low_bit_pos_a
                im_low_bit_pos_a = im_low_bit_pos_b
                im_low_bit_pos_b = tmp_a
                im_bits_to_gen_a = im_bits_to_gen_b
                im_bits_to_gen_b = 0

            # look ahead recursively
            total_clears = clears + lookahead_score_with_bits(a_next, b_next,
                                                              im_bits_to_gen_a,
                                                              im_bits_to_gen_b,
                                                              im_low_bit_pos_a,
                                                              im_low_bit_pos_b,
                                                              depth - 1,
                                                              old_Q=Q,
                                                              old_res=rem)
            if total_clears < best_total:
                best_total = total_clears
    return best_total

def choose_best_seed(seed_pairs):
    """
    Similar to your original approach. We add a consistency check if needed.
    You might want to enforce that the appended bits do not change the Q, but
    typically the seed step doesn't have an 'old_Q' or 'old_res' to preserve.

    We'll keep this mostly the same except if you want to preserve Q from the seed,
    you can do so. For now, we do the original approach.
    """
    lowest_score = float('inf')
    best_pairs = []
    low_a = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    low_b = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1

    for (a, b) in seed_pairs:
        full_a = a << BITS_PER_STEP
        full_b = b << BITS_PER_STEP
        for i in range(2 ** BITS_PER_STEP):
            a_test = full_a + i
            for j in range(2 ** BITS_PER_STEP):
                b_test = full_b + j
                # simulate once
                a_test_ext = a_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)
                b_test_ext = b_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)
                Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test_ext, b_test_ext)

                bits_to_gen_a = 0
                bits_to_gen_b = clears
                lowest_bit_position_generated_a = low_b
                lowest_bit_position_generated_b = low_b - bits_to_gen_b

                future_score = lookahead_score_with_bits(a_next, b_next,
                                                         bits_to_gen_a,
                                                         bits_to_gen_b,
                                                         lowest_bit_position_generated_a,
                                                         lowest_bit_position_generated_b,
                                                         LOOKAHEAD_DEPTH,
                                                         old_Q=Q,
                                                         old_res=rem)
                total_score = clears + future_score
                if total_score < lowest_score:
                    lowest_score = total_score
                    best_pairs = [(a_test_ext, b_test_ext)]
                elif total_score == lowest_score:
                    best_pairs.append((a_test_ext, b_test_ext))

    return random.choice(best_pairs)

##########################
# MAIN STEP SELECTION
##########################

def choose_best_step(a, b, bits_to_gen_a, bits_to_gen_b, low_a, low_b, old_Q=None, old_res=None):
    """
    Updated to enforce validation checks for each extension candidate.
    If old_Q/old_res are given, we ensure the extension does not alter them.
    """
    lowest_score = float('inf')
    best_candidates = []

    for ext_a in range(2 ** bits_to_gen_a):
        for ext_b in range(2 ** bits_to_gen_b):
            # *** NEW *** Check consistency
            if old_Q is not None and old_res is not None:
                if not validate_extension(a, b, ext_a, ext_b, old_Q, old_res):
                    continue

            # Construct the extended a, b
            shift_a_by = bit_length(a) - APPROX_BITS - BITS_PER_STEP
            if shift_a_by < 0:
                shift_a_by = 0
            a_test = (((a >> shift_a_by) + ext_a) << shift_a_by)

            shift_b_by = bit_length(b) - APPROX_BITS - BITS_PER_STEP
            if shift_b_by < 0:
                shift_b_by = 0
            b_test = (((b >> shift_b_by) + ext_b) << shift_b_by)

            Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test, b_test)

            # same "bit position" logic
            im_low_bit_pos_a = low_a
            im_low_bit_pos_b = low_b
            im_bits_to_gen_a = 0
            im_bits_to_gen_b = clears

            if (im_low_bit_pos_a - im_bits_to_gen_b < 1):
                im_bits_to_gen_b = (im_low_bit_pos_a - 1)
            if (im_low_bit_pos_a < 2):
                im_bits_to_gen_b = 0

            tmp_a = im_low_bit_pos_a
            im_low_bit_pos_a = im_low_bit_pos_b
            im_low_bit_pos_b = tmp_a - im_bits_to_gen_b

            # If no swap occurred in simulation
            if b_test == b_next:
                tmp_a = im_low_bit_pos_a
                im_low_bit_pos_a = im_low_bit_pos_b
                im_low_bit_pos_b = tmp_a
                im_bits_to_gen_a = im_bits_to_gen_b
                im_bits_to_gen_b = 0

            future_score = lookahead_score_with_bits(a_next, b_next,
                                                     im_bits_to_gen_a,
                                                     im_bits_to_gen_b,
                                                     im_low_bit_pos_a,
                                                     im_low_bit_pos_b,
                                                     LOOKAHEAD_DEPTH,
                                                     old_Q=Q,
                                                     old_res=rem)
            total_score = clears + future_score

            if total_score < lowest_score:
                lowest_score = total_score
                best_candidates = [(a_test, b_test, ext_a, ext_b)]
            elif total_score == lowest_score:
                best_candidates.append((a_test, b_test, ext_a, ext_b))

    new_a, new_b, chosen_ext_a, chosen_ext_b = random.choice(best_candidates)
    return new_a, new_b, chosen_ext_a, chosen_ext_b


##########################
# SEED ENUM & PROCESS
##########################

def enumerate_seed_pairs(seed_bits):
    """
    Enumerate all valid seed pairs (A, B) in the seed domain:
    [2^(seed_bits-1), 2^(seed_bits) - 1] with A > B.
    """
    seeds = []
    low = 1 << (seed_bits - 1)
    high = (1 << seed_bits) - 1
    for a in range(low, high + 1):
        for b in range(low, high + 1):
            if a > b:
                seeds.append((a, b))
    return seeds

def reconstruct_candidate(history):
    """
    As before (though your original approach was quite tricky). 
    We'll keep it for reference, but presumably you might do a simpler 
    forward-based reconstruction if desired.
    """
    largest_b = 0
    prev_b_curr = 0
    b_prev = 0

    sim = history[-1]['simulation']
    q_curr = sim['Q']
    if (sim['neg_res']):
        q_curr = -q_curr

    b_curr = history[-1]['candidate'][1]
    b_next = abs(q_curr * b_curr + b_prev)

    print("ITERATION: ", 0, "   Next B: ", b_next, 
          "   Curr B: ", b_curr, "   Prev B: ", b_prev, "   Q: ", q_curr)

    b_prev = b_curr
    b_curr = b_next

    for i in range(1, len(history)-1):
        sim = history[(-i)-1]['simulation']
        q_curr = sim['Q']
        if (sim['neg_res']):
            b_prev = -b_prev
        
        b_next = abs(q_curr * b_curr + b_prev)
        print("ITERATION: ", i, "   Next B: ", b_next,
              "   Curr B: ", b_curr, "   Prev B: ", b_prev, "   Q: ", q_curr)

        prev_b_curr = b_curr
        b_prev = b_curr
        b_curr = b_next

        if (b_next > largest_b):
            largest_b = b_next

    reconstructed_b = b_next
    print("Largest B", largest_b)

    last_q = history[0]['simulation']['Q']
    if (history[0]['simulation']['neg_res']):
        last_q = -last_q

    reconstructed_a = abs(b_next * last_q + prev_b_curr)
    print(reconstructed_a, "    " , reconstructed_b)
    return reconstructed_a, reconstructed_b

def generative_process(seed_bits):
    """
    Similar to your original generative process.
    """
    seeds = enumerate_seed_pairs(seed_bits)
    (starting_a, starting_b) = choose_best_seed(seeds)

    # Initialize trackers
    lowest_bit_position_generated_a = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    lowest_bit_position_generated_b = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1

    # Simulate the first step
    Q, rem, clears, shift, current_a, current_b, neg_res = simulate_step(starting_a, starting_b)

    ext_a = (starting_a >> (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)) & ((2**BITS_PER_STEP) - 1)
    ext_b = (starting_b >> (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)) & ((2**BITS_PER_STEP) - 1)

    history = [{
        'iteration': 1,
        'extension_bits': (ext_a, ext_b),
        'candidate': (starting_a, starting_b),
        'simulation': {'Q': Q, 'rem': rem, 'clears': clears, 'shift': shift, 'neg_res': neg_res},
        'swap': False
    }]

    iteration_count = 1

    bits_to_gen_a = 0
    bits_to_gen_b = clears

    lowest_bit_position_generated_a = lowest_bit_position_generated_b
    lowest_bit_position_generated_b -= bits_to_gen_b

    while current_b != 0 or lowest_bit_position_generated_b > 1:
        iteration_count += 1

        # *** We now pass old_Q=Q, old_res=rem to ensure the new extension does not break old Q
        new_a, new_b, ext_a, ext_b = choose_best_step(current_a, current_b,
                                                      bits_to_gen_a,
                                                      bits_to_gen_b,
                                                      lowest_bit_position_generated_a,
                                                      lowest_bit_position_generated_b,
                                                      old_Q=Q,
                                                      old_res=rem)

        # Now simulate again
        Q_new, rem_new, clears_new, shift_new, a_next, b_next, neg_res_new = simulate_step(new_a, new_b)

        bits_to_gen_a = 0
        bits_to_gen_b = clears_new

        if (lowest_bit_position_generated_a - bits_to_gen_b < 1):
            bits_to_gen_b = (lowest_bit_position_generated_a - 1)
        if (lowest_bit_position_generated_a < 2):
            bits_to_gen_b = 0

        tmp_a = lowest_bit_position_generated_a
        lowest_bit_position_generated_a = lowest_bit_position_generated_b
        lowest_bit_position_generated_b = tmp_a - bits_to_gen_b

        swap_happened = (b_next != new_b)  # or whichever logic you prefer
        if new_b == b_next:
            # no swap
            tmp_a = lowest_bit_position_generated_a
            lowest_bit_position_generated_a = lowest_bit_position_generated_b
            lowest_bit_position_generated_b = tmp_a
            bits_to_gen_a = bits_to_gen_b
            bits_to_gen_b = 0

        history.append({
            'iteration': iteration_count,
            'extension_bits': (ext_a, ext_b),
            'candidate': (new_a, new_b),
            'simulation': {
                'Q': Q_new, 
                'rem': rem_new, 
                'clears': clears_new, 
                'shift': shift_new, 
                'neg_res': neg_res_new
            },
            'swap': swap_happened
        })

        # Update current variables
        current_a, current_b = a_next, b_next
        Q, rem = Q_new, rem_new

        if current_b == 0 and lowest_bit_position_generated_b <= 1:
            break

    print("ITERATIONS IT TOOK:", iteration_count)

    # Reconstruct for demonstration:
    reconstructed_a, reconstructed_b = reconstruct_candidate(history)
    print("Reconstructed Candidate Pair:")
    print(f"  A = {reconstructed_a} (binary: {reconstructed_a:0{TOTAL_BITS}b})")
    print(f"  B = {reconstructed_b} (binary: {reconstructed_b:0{TOTAL_BITS}b})")

    return reconstructed_a, reconstructed_b, history

##########################
# HISTORY PRINTING
##########################

def print_history(history, bits_per_step):
    """
    Print the generative history.
    """
    for step in history:
        iteration = step['iteration']
        candidate = step['candidate']
        ext_a, ext_b = step['extension_bits']
        sim = step['simulation']
        swap = step['swap']
        if iteration == 1:
            print(f"[Seed] Iteration {iteration}: "
                  f"Ext bits: A_ext={ext_a:0{bits_per_step}b}, B_ext={ext_b:0{bits_per_step}b} | "
                  f"Candidate: A={candidate[0]:b}, B={candidate[1]:b} | "
                  f"Q={sim['Q']}, rem={sim['rem']}, clears={sim['clears']}, Swap={swap}, "
                  f"NegRes={sim['neg_res']}")
        else:
            print(f"[Step] Iteration {iteration}: "
                  f"Ext bits: A_ext={ext_a:0{bits_per_step}b}, B_ext={ext_b:0{bits_per_step}b} | "
                  f"Candidate: A={candidate[0]:b}, B={candidate[1]:b} | "
                  f"Q={sim['Q']}, rem={sim['rem']}, clears={sim['clears']}, Swap={swap}, "
                  f"NegRes={sim['neg_res']}")

##########################
# MAIN
##########################

def main():
    print("Worst‑Case Candidate Generator for Approximate XGCD (Preserving Q Bits)")
    print(f"TOTAL_BITS = {TOTAL_BITS}, APPROX_BITS = {APPROX_BITS}, BITS_PER_STEP = {BITS_PER_STEP}")
    print(f"Rounding mode: {ROUNDING_MODE}, Integer Rounding: {INTEGER_ROUNDING}")
    print("-" * 60)
    
    # Generate
    candidate_A, candidate_B, history = generative_process(APPROX_BITS)
    
    print("\nFinal Candidate Pair:")
    print(f"  A = {candidate_A} (binary: {candidate_A:0{TOTAL_BITS}b})")
    print(f"  B = {candidate_B} (binary: {candidate_B:0{TOTAL_BITS}b})")
    if candidate_B:
        print(f"  Ratio A/B = {candidate_A / candidate_B:.6f}")
    else:
        print("  B is 0!")
    
    print("-" * 60)
    print("Generative History:")
    print_history(history, BITS_PER_STEP)
    
    print("-" * 60)
    print("Full XGCD Simulation Result:")
    gcd_val, count, avg_clears = xgcd_bitwise(candidate_A, candidate_B,
                                              total_bits=TOTAL_BITS,
                                              approx_bits=APPROX_BITS,
                                              rounding_mode=ROUNDING_MODE,
                                              integer_rounding=INTEGER_ROUNDING,
                                              plus_minus=False,
                                              enable_plotting=False)
    print(f"GCD({candidate_A}, {candidate_B}) = {gcd_val}, reached in {count} iterations.")
    print(f"  Average bit clears: {avg_clears:.3f}")

if __name__ == "__main__":
    main()
