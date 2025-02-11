#!/usr/bin/env python3
"""
Worst‑Case Candidate Generator for an Approximate XGCD Algorithm

This script first enumerates all valid seed pairs (with APPROX_BITS‑bit numbers,
MSB=1 and A ≥ B). It then “grows” the candidate pair bit‑by‑bit (or by a fixed
number of bits per step, as specified by BITS_PER_STEP). At each extension, all
possible new bit‑combinations are tried; for each candidate extension we simulate
one xgcd‑style step and score it by:
  - The number of bit‐clears (i.e. how many bits are lost from A)
  - A tie–breaker based on randomness
After selecting the best candidate extension, the candidate pair is updated (using
the usual swap/non‑swap rule)
"""

import math
import random

##########################
# PARAMETERS
##########################

TOTAL_BITS      = 16         # Final bit‑width for the candidate numbers
APPROX_BITS     = 4          # Bit‑width of the seed domain (numbers in [2^(APPROX_BITS–1), 2^(APPROX_BITS)-1])
BITS_PER_STEP   = 2          # Number of bits added per generative extension
ROUNDING_MODE   = "truncate" # "truncate" or "round" (for the fixed‑point division)
INTEGER_ROUNDING = True      # Whether to adjust quotient by integer rounding

##########################
# HELPER FUNCTIONS
##########################

def bit_length(x):
    """Return the bit length of x not counting leading 0s (in order to allign a and b)"""
    return x.bit_length() if x else 0

def bit_len(n):
    """Return the bit‑length of n."""
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

def lut_result(a_top, b_top):
    """
    Compute the ratio (a_top / b_top) in fixed-point with 'approx_bits' fractional bits
    """
    if b_top == 0:
        return 0  # avoid div by zero (should not happen if b!=0, but just in case)

    numerator = a_top << APPROX_BITS  # up-shift a_top by approx_bits
    if ROUNDING_MODE == "round":
        # do integer division with rounding:
        # equivalent to floor( (numerator + b_top/2 ) / b_top )
        return (numerator + (b_top >> 1)) // b_top
    else:
        # default: truncate / floor
        return numerator // b_top


##########################
# XGCD STEP SIMULATION
##########################

def simulate_step(a, b):
    """
    Simulate one iteration of the approximate XGCD step on (a, b).

    Process:
      1. Align b to a.
      2. Extract the top `appx_bits` of a and b_aligned.
      3. Compute a fixed‑point quotient.
      4. Adjust the quotient by shifting and (if needed) integer rounding.
      5. Compute the residual: rem = |a - Q*b|.
      6. Compute bit clears = bit_len(a) - bit_len(rem).

    Returns a tuple: (Q, rem, clears, shift)
    """
    b_aligned, shift = align_b(a, b)
    a_top = get_fixed_top_bits(a, APPROX_BITS)
    b_top = get_fixed_top_bits(b_aligned, APPROX_BITS)
    quotient = lut_result(a_top, b_top, APPROX_BITS)

    # Emulate the additional shifting/rounding of the quotient:
    Q_pre_round = (quotient << shift) >> (APPROX_BITS - 1)
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and INTEGER_ROUNDING:
        Q += 1

    prod = b * Q

    residual = a - prod

    if residual < 0:
        residual = -residual

    if residual > b:
        # then a_new becomes 'a', and b stays b
        residual = residual

    clears = bit_len(a) - bit_len(residual)
    if clears < 0:
        clears = 0

    if residual > b:
        # then a_new becomes 'a', and b stays b
        a_next = residual
        b_next = b
    else:
        # a_new is smaller => a = b, b = a_new
        a_next, b_next = b, residual

    return Q, residual, clears, shift, a_next, b_next


##########################
# GENERATIVE PROCESS
##########################

def enumerate_seed_pairs(seed_bits):
    """
    Enumerate all valid seed pairs (A, B) in the seed domain:
      A, B ∈ [2^(seed_bits-1), 2^(seed_bits)-1] and A ≥ B.
    """
    seeds = []
    low = 1 << (seed_bits - 1) 
    high = (1 << seed_bits) - 1
    for a in range(low, high + 1):
        for b in range(low, high + 1):
            # did not equality bc all worst case dont have it
            # plus dont haev to worry about validity of following bits generated and maybe b > a 
            if a > b:
                seeds.append((a, b))
    return seeds

def choose_best_step(a, b, bits_to_gen_a, bits_to_gen_b):
    """
    Given the current candidate pair (a, b) with bit‑width current_bits,
    try all possible extensions by appending bits_per_step new bits to both.
    For each candidate extension:
      1. Compute the new numbers (after left‑shifting and appending bits).
      2. Normalize them by shifting up and down.
      3. Simulate one xgcd step and score the candidates based off minimal bit clears
    Return the updated candidate pair (after applying the update rule)
    """
   
    lowest_clears = 100000

    best_pairs = []

    for gen_bits_a in range(2 ** bits_to_gen_a):
        shift_a_by = bit_length(a) - APPROX_BITS - BITS_PER_STEP
        # do the math to shift down then add in the bits at the gen steps
        # then shift back up to the orignal place the bits were at
        a_test = ((a >> shift_a_by) + gen_bits_a) << shift_a_by

        for gen_bits_b in range(2 ** bits_to_gen_b):
            shift_b_by = bit_length(b) - APPROX_BITS - BITS_PER_STEP
            # do the math to shift down then add in the bits at the gen steps
            # then shift back up to the orignal place the bits were at
            b_test = ((b >> shift_a_by) + gen_bits_b) << shift_b_by

            Q, rem, clears, shift, a_next, b_next = simulate_step(a_test, b_test)

            if (clears < lowest_clears):
                lowest_clears = clears
                best_pairs = []
                best_pairs.append((a_test, b_test))
            elif (clears == lowest_clears):
                best_pairs.append((a_test, b_test))
                
    (new_a, new_b) = random.choice(best_pairs)
    # Now update the candidate pair using the chosen extension.
    return new_a, new_b


def choose_best_seed(seed_pairs):
    """
    For all seed pairs, simulate one xgcd step and choose the candidate
    with the minimal score.
    """
    starting_a = None
    starting_b = None

    lowest_clears = 100000

    best_pairs = []

    best_score = None
    best_pair = None
    best_details = None
    for (a, b) in seed_pairs:
        # check each a and b to see which gives least bit clears and with what generated numbers
        # start with checking all permuations
        full_a = a << (BITS_PER_STEP)
        full_b = b << (BITS_PER_STEP)
        for i in range(BITS_PER_STEP):
            a_test = full_a + i
            for j in range(BITS_PER_STEP):
                b_test = full_b + j

                # use our temp a and b and extend it to make the msb 1/ full total bits long
                a_test = a_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)
                b_test = b_test << (TOTAL_BITS - APPROX_BITS - BITS_PER_STEP)

                #get the bit clears
                Q, rem, clears, shift, a_next, b_next = simulate_step(a_test, b_test)

                if (clears < lowest_clears):
                    lowest_clears = clears
                    best_pairs = []
                    best_pairs.append((a_test, b_test))
                elif (clears == lowest_clears):
                    best_pairs.append((a_test, b_test))
    
    (starting_a, starting_b) = random.choice(best_pairs)

    return (starting_a, starting_b)


def generative_process(total_bits, seed_bits, bits_per_step, appx_bits, mode, int_rounding):
    """
    Generate a candidate pair by first enumerating all seed pairs and then
    extending the best candidate step‑by‑step until the numbers have total_bits.
    Returns the final candidate pair along with the generative history.
    """
    # get all possible approx bits that are valid where a > b
    seeds = enumerate_seed_pairs(seed_bits)
    # for all the sides, find the one that clears the least amount of bits after generating bits
    # and randonmly from all ones with equal least bit clears pick one
    (starting_a, starting_b) = choose_best_seed(seeds)

    # will keep track of the lowest bit we've generated so far
    lowest_bit_position_generated_a = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1
    lowest_bit_position_generated_b = TOTAL_BITS - APPROX_BITS - BITS_PER_STEP + 1

    Q, rem, clears, shift, current_a, current_b = simulate_step(starting_a, starting_b)

    history = [{
        'iteration': 1,
        'Q': Q,
        'clears': clears,
        'starting': (starting_a, starting_b),
        'swap': False
    }]

    iteration_count = 1

    while current_b != 0:
        iteration_count += 1

        ############
        # MAKE IT SO WE PICK BEST CANDIDATE 
        # i.e. one with least bit clears and make that current_a, current b
        # here we add in the geneated bits and keep track of the lowest bit position
        # and should flip a and b
        bits_to_gen_a = (APPROX_BITS+BITS_PER_STEP) - (bit_length(current_a) - lowest_bit_position_generated_a + 1)
        bits_to_gen_b = (APPROX_BITS+BITS_PER_STEP) - (bit_length(current_b) - lowest_bit_position_generated_b + 1)

        current_a, current_b = choose_best_step(current_a, current_b, bits_to_gen_a, bits_to_gen_b)
        ############
        # keep the lowest bit we've updated updated
        lowest_bit_position_generated_a -= bits_to_gen_a
        lowest_bit_position_generated_b -= bits_to_gen_b

        # flip a and b because after this step, they essentially flip
        tmp_lowbit_a = lowest_bit_position_generated_a
        lowest_bit_position_generated_a = lowest_bit_position_generated_b
        lowest_bit_position_generated_b = tmp_lowbit_a
        ############

        Q, rem, clears, shift, a_next, b_next = simulate_step(current_a, current_b)

        # if the residual/remainder is the same as next a, that means there was no swap and we 
        # need to swap back the lower bit positions 
        if (rem == a_next):
            tmp_lowbit_a = lowest_bit_position_generated_a
            lowest_bit_position_generated_a = lowest_bit_position_generated_b
            lowest_bit_position_generated_b = tmp_lowbit_a

        history.append({
            'iteration': iteration_count,
            'Q': Q,
            'clears': clears,
            'starting': (starting_a, starting_b),
            'next': (current_a, current_b),
            'swap': rem == a_next
        })


    # TO DO, RECONSTRUCT a AND b 
    # MAKE FUNCTION CALL HERE TO RECONSTRUCT BY TRAVERSING history dicts FROM BOTTOM UP from the last iteration up until the first one
    # AND ADDING every "NEXT" A with the "next" A from the previous history log
    # Then do the same for b up up until the 1 entry. ( so dont add the first entry in)
    # By the end you should have a recreated a and b. And then keep a lookout for swaps, if there was a swap, then 
    # 
    #########################
    print("ITERATIONS IT TOOK: \n")
    print(iteration_count)

    return current_a, current_b, history

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
        Q, rem, clears, shift = simulate_step(a, b, appx_bits, mode, int_rounding)
        total_clears += clears
        if rem > b:
            a, b = rem, b
        else:
            a, b = b, rem
    avg_clears = total_clears / iter_count if iter_count else 0
    return a, iter_count, avg_clears

def print_history(history, bits_per_step):
    for step in history:
        bits = step['bits']
        A, B = step['candidate']
        if step.get('seed', False):
            print(f"[Seed] {bits}‑bit candidate: A = {A:0{bits}b}, B = {B:0{bits}b}, details: {step['details']}")
        else:
            ext_a, ext_b = step['details']['extension_bits']
            Q, rem, clears, _ = step['details']['simulation']
            print(f"[Extend] {bits}‑bit candidate: Extension bits: A_ext = {ext_a:0{bits_per_step}b}, B_ext = {ext_b:0{bits_per_step}b} | A = {A:0{bits}b}, B = {B:0{bits}b} | Simulated: Q = {Q}, rem = {rem}, clears = {clears}")

##########################
# MAIN
##########################

def main():
    print("Worst‑Case Candidate Generator for Approximate XGCD")
    print(f"TOTAL_BITS = {TOTAL_BITS}, APPROX_BITS = {APPROX_BITS}, BITS_PER_STEP = {BITS_PER_STEP}")
    print(f"Rounding mode: {ROUNDING_MODE}, Integer Rounding: {INTEGER_ROUNDING}")
    print("-" * 60)
    
    # Use APPROX_BITS as the approximation bits as well.
    candidate_A, candidate_B, history = generative_process(TOTAL_BITS, APPROX_BITS, BITS_PER_STEP, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING)
    
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
    gcd_val, iter_count, avg_clears = full_xgcd_simulation(candidate_A, candidate_B, TOTAL_BITS, APPROX_BITS, ROUNDING_MODE, INTEGER_ROUNDING)
    print("Full XGCD Simulation Result:")
    print(f"  GCD = {gcd_val}")
    print(f"  Iterations = {iter_count}")
    print(f"  Average Bit Clears = {avg_clears:.3f}")

if __name__ == "__main__":
    main()
