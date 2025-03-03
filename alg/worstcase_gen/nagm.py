#!/usr/bin/env python3
"""
Worst‑Case Candidate Generator for an Approximate XGCD Algorithm

This script first enumerates all valid seed pairs (with APPROX_BITS‑bit numbers,
MSB=1 and A ≥ B). It then “grows” the candidate pair bit‑by‑bit (or by a fixed
number of bits per step, as specified by BITS_PER_STEP). At each extension, all
possible new bit‑combinations are tried; for each candidate extension we simulate
one xgcd‑style step and score it by the number of bit‐clears (i.e. how many bits are lost
from A). In the event of ties, one candidate is chosen randomly.
After selecting the best candidate extension, the candidate pair is updated (using
the usual swap/non‑swap rule).

The script also records a history of each generative step. Later, we use this
history to “reconstruct” an approximate candidate pair by combining the seed with
all of the extension bits.
"""

import math
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgcd_q_change import xgcd_bitwise

##########################
# PARAMETERS
##########################

TOTAL_BITS       = 18         # Final bit‑width for the candidate numbers
APPROX_BITS      = 4          # Bit‑width of the seed domain (numbers in [2^(APPROX_BITS–1), 2^(APPROX_BITS)-1])
BITS_PER_STEP    = 1          # Number of bits added per generative extension
ROUNDING_MODE    = "truncate" # "truncate" or "round" (for the fixed‑point division)
INTEGER_ROUNDING = True       # Whether to adjust quotient by integer rounding
LOOKAHEAD_DEPTH = 5

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
    Normalize x to have exactly 'bits' bits (i.e. force the MSB to 1).
    This is done by masking off any higher bits and ensuring x >= 2^(bits-1).
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
    - If x_val has fewer than approx_bits bits, left‑shift it so it becomes exactly approx_bits bits.
    - Otherwise, shift down so we only keep the top approx_bits bits.
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
        return 0  # avoid division by zero (should not happen if b != 0)
    numerator = a_top << APPROX_BITS  # up‑shift a_top by APPROX_BITS
    if ROUNDING_MODE == "round":
        # round‑half‑up division:
        return (numerator + (b_top >> 1)) // b_top
    else:
        # default: truncate/floor
        return numerator // b_top

##########################
# XGCD STEP SIMULATION
##########################

def simulate_step(a, b):
    """
    Simulate one iteration of the approximate XGCD step on (a, b).

    Process:
      1. Align b to a.
      2. Extract the top APPROX_BITS of a and b_aligned.
      3. Compute a fixed‑point quotient.
      4. Adjust the quotient by shifting and (if needed) integer rounding.
      5. Compute the residual: rem = |a - Q * b|.
      6. Compute bit clears = bit_len(a) - bit_len(rem).
      7. Apply the update rule (swap or not) based on whether rem > b.

    Returns a tuple: (Q, rem, clears, shift, a_next, b_next)
    """
    b_aligned, shift = align_b(a, b)
    a_top = get_fixed_top_bits(a, APPROX_BITS)
    b_top = get_fixed_top_bits(b_aligned, APPROX_BITS)
    quotient = lut_result(a_top, b_top)
    Q_pre_round = (quotient << shift) >> (APPROX_BITS - 1)
    Q = Q_pre_round >> 1
    # if (Q_pre_round & 1) and INTEGER_ROUNDING:
    Q += 1
    
    prod = b * Q
    prod_two = b * (Q_pre_round >> 1)
    residual = a - prod
    residual_two = a - prod_two

    neg_res = False
    neg_res2 = False
    

    if residual < 0:
        neg_res = True
        residual = -residual
    
    if residual_two < 0:
        neg_res2 = True
        residual_two = -residual_two

    if (residual_two < residual):
        neg_res = neg_res2
        residual = residual_two
        Q = (Q_pre_round >> 1)

    clears = bit_len(a) - bit_len(residual)
    if clears < 0:
        clears = 0

    if residual > b:
        # print("NO SWAP")
        a_next = residual
        b_next = b
    else:
        a_next, b_next = b, residual

    return Q, residual, clears, shift, a_next, b_next, neg_res

##########################
# GENERATIVE PROCESS FUNCTIONS
##########################

def enumerate_seed_pairs(seed_bits):
    """
    Enumerate all valid seed pairs (A, B) in the seed domain:
      A, B ∈ [2^(seed_bits-1), 2^(seed_bits)-1] and A > B.
    """
    seeds = []
    low = 1 << (seed_bits - 1)
    high = (1 << seed_bits) - 1
    for a in range(low, high + 1):
        for b in range(low, high + 1):
            if a > b:
                seeds.append((a, b))
    return seeds

def lookahead_score_with_bits(a, b, bits_to_gen_a, bits_to_gen_b, low_bit_pos_a, low_bit_pos_b, depth):
    """
    Recursively compute the minimal cumulative bit clears for candidate pair (a, b)
    over 'depth' future steps, using the actual bits-to-generate determined by:
      bits_to_gen = (APPROX_BITS + BITS_PER_STEP) - (bit_length(x) - low_x + 1)
    and updating the lowest bit positions accordingly.
    """
    if b == 0 and depth > 0:
        return 1000 * depth  # Adjust 1000 as needed so that early termination is strongly penalized.
    
    if depth == 0:
        return 0
    
    best_total = float('inf')
    # Loop over all possible extension bits for a.
    for ext_a in range(2 ** bits_to_gen_a):
        shift_a_by = bit_length(a) - APPROX_BITS - BITS_PER_STEP
        if shift_a_by < 0:
            shift_a_by = 0
        a_test = (((a >> shift_a_by) + ext_a) << shift_a_by)
        # Loop over all possible extension bits for b.
        for ext_b in range(2 ** bits_to_gen_b):
            shift_b_by = bit_length(b) - APPROX_BITS - BITS_PER_STEP
            if shift_b_by < 0:
                shift_b_by = 0
            b_test = (((b >> shift_b_by) + ext_b) << shift_b_by)
            Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test, b_test)
            #################################

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
                # print("NO SWAP AFTER SIM STEP")
                # if we keep b as b, we shouldnt generate anymore bit sso it becomes it orignal
                # the residual becomes A so now we generate bit son its residual
                tmp_a = im_low_bit_pos_a
                im_low_bit_pos_a = im_low_bit_pos_b
                im_low_bit_pos_b = tmp_a
                im_bits_to_gen_a = im_bits_to_gen_b
                im_bits_to_gen_b = 0
                
            if (im_bits_to_gen_b < 0):
                print("WAS NEGATIVE. STATS: ", )

            #################################
            total_clears = clears + lookahead_score_with_bits(a_next, b_next, im_bits_to_gen_a, im_bits_to_gen_b, im_low_bit_pos_a, im_low_bit_pos_b, depth - 1)
            if total_clears < best_total:
                best_total = total_clears
    return best_total


def choose_best_seed(seed_pairs):
    """
    For all seed pairs, simulate one extension using the current bits-to-generate
    (here implicitly APPROX_BITS + BITS_PER_STEP) and the initial lowest-bit positions,
    then use lookahead to choose the candidate with minimal cumulative clears.
    """
    lowest_score = float('inf')
    best_pairs = []
    # Define starting lowest-bit positions (as used in generative_process).
    low_a = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    low_b = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    for (a, b) in seed_pairs:
        full_a = a << BITS_PER_STEP
        full_b = b << BITS_PER_STEP
        for i in range(2 ** BITS_PER_STEP):
            a_test = full_a + i
            for j in range(2 ** BITS_PER_STEP):
                b_test = full_b + j
                a_test_ext = a_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)
                b_test_ext = b_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)
                Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test_ext, b_test_ext)
                ############################
                bits_to_gen_a = 0
                bits_to_gen_b = clears

                lowest_bit_position_generated_a = low_b
                lowest_bit_position_generated_b = low_b - bits_to_gen_b
                ############################
                # print("ABOUT TO LOOK AHEAD")
                future_score = lookahead_score_with_bits(a_next, b_next, bits_to_gen_a, bits_to_gen_b, lowest_bit_position_generated_a, lowest_bit_position_generated_b, LOOKAHEAD_DEPTH)
                # print("COMPLETED LOOK AHEAD")
                total_score = clears + future_score
                if total_score < lowest_score:
                    lowest_score = total_score
                    best_pairs = [(a_test_ext, b_test_ext)]
                elif total_score == lowest_score:
                    best_pairs.append((a_test_ext, b_test_ext))

    print("NUMBER OF CHOICES FROM BEST SEED: ", len(best_pairs))
    print("BEST SEED PAIRS: ", best_pairs, " \n")
    return random.choice(best_pairs)


def choose_best_step(a, b, bits_to_gen_a, bits_to_gen_b, low_a, low_b):
    """
    Given the current candidate pair (a, b) and the numbers of bits to generate for each,
    try all possible extensions by appending the appropriate number of bits.
    For each candidate extension, simulate one xgcd step and add the lookahead score (using
    the current lowest-bit positions) to determine the total score.
    Return the updated candidate pair along with the chosen extension bits.
    """
    lowest_score = float('inf')
    best_candidates = []
    for ext_a in range(2 ** bits_to_gen_a):
        shift_a_by = bit_length(a) - APPROX_BITS - BITS_PER_STEP
        if shift_a_by < 0:
            shift_a_by = 0
        a_test = (((a >> shift_a_by) + ext_a) << shift_a_by)
        for ext_b in range(2 ** bits_to_gen_b):
            shift_b_by = bit_length(b) - APPROX_BITS - BITS_PER_STEP
            if shift_b_by < 0:
                shift_b_by = 0
            b_test = (((b >> shift_b_by) + ext_b) << shift_b_by)
            Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a_test, b_test)
            #####################

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

            # If no swap occurred in simulation, swap the low-bit trackers
            if b_test == b_next:
                # print("NO SWAP AFTER SIM STEP")
                # if we keep b as b, we shouldnt generate anymore bit sso it becomes it orignal
                # the residual becomes A so now we generate bit son its residual
                tmp_a = im_low_bit_pos_a
                im_low_bit_pos_a = im_low_bit_pos_b
                im_low_bit_pos_b = tmp_a
                im_bits_to_gen_a = im_bits_to_gen_b
                im_bits_to_gen_b = 0

            #####################
            future_score = lookahead_score_with_bits(a_next, b_next, im_bits_to_gen_a, im_bits_to_gen_b, im_low_bit_pos_a, im_low_bit_pos_b, LOOKAHEAD_DEPTH)
            total_score = clears + future_score
            if total_score < lowest_score:
                lowest_score = total_score
                best_candidates = [(a_test, b_test, ext_a, ext_b)]
            elif total_score == lowest_score:
                best_candidates.append((a_test, b_test, ext_a, ext_b))

    print("NUMBER OF CHOICES FROM BEST STEP: ", len(best_candidates))
    print("BEST STEP PAIRS: ", best_candidates, " \n")

    new_a, new_b, chosen_ext_a, chosen_ext_b = random.choice(best_candidates)
    return new_a, new_b, chosen_ext_a, chosen_ext_b


def reconstruct_candidate(history):
    """
    Reconstruct the candidate pair (A, B) by traversing the generative history.
    Starting with the seed candidate, for every extension step, shift the candidate
    left by bits_per_step and OR in the extension bits. If a swap occurred in that step,
    swap the roles of A and B.
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

    #loop through all iterations but start with last iterations
    
    # print(" THIS IS THE SECOND B   ", history[1]['candidate'][1])

    # print("ITERATION: ", 0, "   Next B: ", b_next, "   Curr B: ", b_curr, "   Prev B: ", b_prev, "   Q: ", q_curr)

    b_prev = b_curr
    b_curr = b_next

    for i in range(1, len(history)-1):
        # b_prev = history[-i]['candidate'][1]

        # b_curr = history[-i-1]['candidate'][1]

        sim = history[(-i)-1]['simulation']

        q_curr = sim['Q']

        if (sim['neg_res']):
            b_prev = -b_prev
        
        b_next = abs(q_curr * b_curr + b_prev)
        # print("ITERATION: ", i, "   Next B: ", b_next, "   Curr B: ", b_curr, "   Prev B: ", b_prev, "   Q: ", q_curr)

        #used pretty much just to do the calc for A
        prev_b_curr = b_curr

        b_prev = b_curr
        b_curr = b_next

        # b_prev = history[-i]['candidate'][1]

        if (b_next > largest_b):
            largest_b = b_next

    reconstructed_b = b_next

    # print("Largest B", largest_b)

    last_q = history[0]['simulation']['Q']
    if (history[0]['simulation']['neg_res']):
        last_q = -last_q

    reconstructed_a = abs(b_next * last_q + prev_b_curr)

    print(reconstructed_a, "    " , reconstructed_b)

    return reconstructed_a, reconstructed_b

def generative_process(seed_bits):
    """
    Generate a candidate pair by:
      1. Enumerating all seed pairs.
      2. Choosing the best seed (minimal bit clears).
      3. Extending the candidate pair step‑by‑step until reaching total_bits.
      4. Recording the history of extensions.
      5. Reconstructing the candidate pair from the history.
    Returns the final candidate pair along with the generative history.
    """
    seeds = enumerate_seed_pairs(seed_bits)
    (starting_a, starting_b) = choose_best_seed(seeds)

    # print("GOT PAST SEED")

    # Initialize the lowest bit positions (heuristic; used for computing extension length)
    lowest_bit_position_generated_a = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    lowest_bit_position_generated_b = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1

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

    # print("LOWEST B BIT: ", lowest_bit_position_generated_b, "B IS: ", current_b, "   CURRENT A: ", current_a)

    while current_b != 0 or lowest_bit_position_generated_b > 1:
        iteration_count += 1

        # Choose best extension (now returns extension bits as well)
        current_a, current_b, ext_a, ext_b = choose_best_step(current_a, current_b, bits_to_gen_a, bits_to_gen_b, lowest_bit_position_generated_a, lowest_bit_position_generated_b)

        Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(current_a, current_b)

        # print(f"STEP INFO: Q: {Q}, rem: {rem}, clears: {clears}, shift: {shift}, current a: {current_a}, current b: {current_b}")

        bits_to_gen_a = 0
        bits_to_gen_b = clears

        if (lowest_bit_position_generated_a - bits_to_gen_b < 1):
            bits_to_gen_b = (lowest_bit_position_generated_a - 1)

        if (lowest_bit_position_generated_a < 2):
            bits_to_gen_b = 0

        tmp_a = lowest_bit_position_generated_a
        lowest_bit_position_generated_a = lowest_bit_position_generated_b
        lowest_bit_position_generated_b = tmp_a - bits_to_gen_b

        # If no swap occurred in simulation, swap the low-bit trackers
        if current_b == b_next:
            print("NO SWAP AFTER SIM STEP")
            # if we keep b as b, we shouldnt generate anymore bit sso it becomes it orignal
            # the residual becomes A so now we generate bit son its residual
            tmp_a = lowest_bit_position_generated_a
            lowest_bit_position_generated_a = lowest_bit_position_generated_b
            lowest_bit_position_generated_b = tmp_a
            bits_to_gen_a = bits_to_gen_b
            bits_to_gen_b = 0

        history.append({
            'iteration': iteration_count,
            'extension_bits': (ext_a, ext_b),
            'candidate': (current_a, current_b),
            'simulation': {'Q': Q, 'rem': rem, 'clears': clears, 'shift': shift, 'neg_res': neg_res},
            'swap': (current_b == b_next)
        })

        current_a = a_next
        current_b = b_next

        # print("LOWEST A BIT: ", lowest_bit_position_generated_a, "  BITS TO GEN A: ", bits_to_gen_a )
        # print("LOWEST B BIT: ", lowest_bit_position_generated_b, "     B IS: ", current_b, "   CURRENT A: ", current_a, "  BITS TO GEN B: ", bits_to_gen_b, "\n")

    print("ITERATIONS IT TOOK:")
    print(iteration_count)

    # Reconstruct candidate pair from generative history:
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
    Each history entry shows the iteration number, the extension bits (if any),
    the candidate pair (in binary), the simulation results, and whether a swap occurred.
    """
    for step in history:
        iteration = step['iteration']
        if iteration == 1:
            candidate = step['candidate']
            ext_a, ext_b = step['extension_bits']
            sim = step['simulation']
            swap = step['swap']
            print(f"[ Seed ] Iteration {iteration}: Extension bits: A_ext = {ext_a:0{bits_per_step}b}, B_ext = {ext_b:0{bits_per_step}b} | "
                  f"Candidate: A = {candidate[0]:b}, B = {candidate[1]:b} | Simulated: Q = {sim['Q']}, rem = {sim['rem']}, clears = {sim['clears']} | "
                  f"Swap: {swap}, Residual Sign Flip: {sim['neg_res']}")
        else:
            candidate = step['candidate']
            ext_a, ext_b = step['extension_bits']
            sim = step['simulation']
            swap = step['swap']
            print(f"[Extend] Iteration {iteration}: Extension bits: A_ext = {ext_a:0{bits_per_step}b}, B_ext = {ext_b:0{bits_per_step}b} | "
                  f"Candidate: A = {candidate[0]:b}, B = {candidate[1]:b} | Simulated: Q = {sim['Q']}, rem = {sim['rem']}, clears = {sim['clears']} | "
                  f"Swap: {swap}, Residual Sign Flip: {sim['neg_res']}")

##########################
# FULL XGCD SIMULATION
##########################

def full_xgcd_simulation(a, b, total_bits, appx_bits, mode, int_rounding):
    """
    Run the full approximate xgcd simulation on candidate pair (a, b).
    Returns the final GCD, number of iterations, and average bit clears.
    """
    mask = (1 << total_bits) - 1
    a, b = a & mask, b & mask
    if b > a:
        a, b = b, a
    iter_count = 0
    total_clears = 0
    while b:
        iter_count += 1
        Q, rem, clears, shift, a_next, b_next, neg_res = simulate_step(a, b)
        total_clears += clears
        if rem > b:
            a, b = rem, b
        else:
            a, b = b, rem
    avg_clears = total_clears / iter_count if iter_count else 0
    return a, iter_count, avg_clears

##########################
# MAIN
##########################

def main():
    print("Worst‑Case Candidate Generator for Approximate XGCD")
    print(f"TOTAL_BITS = {TOTAL_BITS}, APPROX_BITS = {APPROX_BITS}, BITS_PER_STEP = {BITS_PER_STEP}")
    print(f"Rounding mode: {ROUNDING_MODE}, Integer Rounding: {INTEGER_ROUNDING}")
    print("-" * 60)
    
    # Use APPROX_BITS as the seed bit-width here.
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
    print("Full XGCD Simulation Result 1:")
    gcd_val, count, avg_clears = xgcd_bitwise(candidate_A, candidate_B,
                                                           total_bits=TOTAL_BITS,
                                                           approx_bits=APPROX_BITS,
                                                           rounding_mode=ROUNDING_MODE,
                                                           integer_rounding=INTEGER_ROUNDING,
                                                           plus_minus=False,
                                                           enable_plotting=False)
                                                           
    print(f"(Medium) GCD of {candidate_A} and {candidate_B} is {gcd_val}, reached in {count} iterations.")
    print(f"  Average bit clears: {avg_clears:.3f}")
    


    print("-" * 60)
    # print("Full XGCD Simulation Result 2:")
    # gcd_val, count, avg_clears = xgcd_bitwise(candidate_A + gcd_val - 1, candidate_B + gcd_val - 2,
    #                                                        total_bits=TOTAL_BITS,
    #                                                        approx_bits=APPROX_BITS,
    #                                                        rounding_mode=ROUNDING_MODE,
    #                                                        integer_rounding=INTEGER_ROUNDING,
    #                                                        plus_minus=False,
    #                                                        enable_plotting=False)
                                                           
    # print(f"(ITER 2) GCD of {candidate_A + gcd_val - 1} and {candidate_B + gcd_val - 2} is {gcd_val}, reached in {count} iterations.")
    # print(f"  Average bit clears: {avg_clears:.3f}")


if __name__ == "__main__":
    main()

