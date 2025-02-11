#!/usr/bin/env python3
import math
import sys

########################################
# Basic Helper Functions (as provided)
########################################

def bit_length(x):
    """Return the bit length of x (ignoring leading zeros)."""
    if (x==0):
        return 0
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
    If x_val has fewer than approx_bits bits, left-shift it so that it has exactly approx_bits.
    """
    if x_val == 0:
        return 0
    L = bit_length(x_val)
    if L <= approx_bits:
        return x_val << (approx_bits - L)
    else:
        shift_down = L - approx_bits
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

########################################
# Forward Simulation: Worst-Case Progression
########################################

def simulate_worst_case_progression(a, b, approx_bits, rounding_mode, threshold, integer_rounding=True):
    """
    Given an initial pair (a, b) (assumed normalized to approx_bits bits),
    simulate the modified xgcd-step progression until cumulative clears >= threshold.
    
    At each iteration:
      - Align b to a.
      - If shift_amount > 0, try all 2^(shift_amount) candidates:
             candidate = (b << shift_amount) + k  for k in 0 ... (2^(shift_amount)-1).
      - For each candidate, compute:
             a_top = get_fixed_top_bits(a, approx_bits)
             b_top = get_fixed_top_bits(candidate, approx_bits)
             quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
             Q_pre_round = (quotient << shift_amount) >> (approx_bits-1)
             Q = Q_pre_round >> 1; if (Q_pre_round & 1) and integer_rounding then Q += 1.
      - Then compute b_adjusted = b * Q and residual = |a - b_adjusted|.
      - Compute clears = bit_length(a) - bit_length(residual) (clamped to 0).
      - Choose the candidate (the value of k) that minimizes clears.
      - Also record the update type:
            if residual > b then update_type = "non-swap" (update a = residual, b stays)
            else update_type = "swap" (update a, b = (b, residual)).
      - Update cumulative_clears and set the new (a, b).
      
    The simulation stops when cumulative_clears >= threshold OR when b becomes 0.
    
    Returns a tuple:
       (steps, final_a, final_b, cumulative_clears, iteration_count)
    where steps is a list (one per iteration) recording:
         { iteration, a, b, shift, chosen_k, Q, residual, clears, update_type }.
    """
    cumulative_clears = 0
    iteration = 0
    steps = []
    prev_shift_add = 0
    # in case residual > b and we need to make up bits for a
    a_prev_shift_add = 0
    
    while cumulative_clears < threshold and b != 0:
        iteration += 1
        #num of a candidates
        a_candi = 1 << a_prev_shift_add
        a = a << a_prev_shift_add
        for j in range(a_candi):
            a += j
            if a < b:
                continue

            b_aligned, shift_amount = align_b(a, b)
            shift_amount += prev_shift_add
            if (shift_amount >= 4):
                print(shift_amount)
                print(prev_shift_add)
                print(b)
                
            possibilities = []
            num_candidates = 1 << shift_amount  # 2^(shift_amount) possibilities
            for k in range(num_candidates):
                candidate = (b << shift_amount) + k  # candidate for b_aligned
                a_top = get_fixed_top_bits(a, approx_bits)
                b_top = get_fixed_top_bits(candidate, approx_bits)

                quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)

                # save the .5 
                Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1)
                Q = Q_pre_round >> 1

                # do the integer rounding
                if (Q_pre_round & 1) and integer_rounding:
                    Q += 1

                b_adjusted = b * Q

                residual = a - b_adjusted

                if residual < 0:
                    residual = -residual
                
                a_tmp = a
                # if residual > b:
                #     a_tmp = residual

                clears = bit_length(a_tmp) - bit_length(residual)
                if clears < 0:
                    clears = 0

                possibilities.append({
                    'k': k,
                    'candidate': candidate,
                    'Q': Q,
                    'residual': residual,
                    'clears': clears,
                    'shift': shift_amount,
                    'a_top': a_top,
                    'b_top': b_top
                })

        # Choose the candidate with minimum clears.
        best = min(possibilities, key=lambda x: x['clears'])
        chosen = best
        cumulative_clears += chosen['clears']

        a = chosen['a_top']
        
        # Decide update type:
        if chosen['residual'] > b:
            update_type = "non-swap"  # a becomes residual; b stays
            a_new = chosen['residual']
            b_new = chosen['candidate']
            a_prev_shift_add = chosen['clears']
            prev_shift_add = 0
        else:
            update_type = "swap"      # a becomes b; b becomes residual
            a_new = chosen['candidate']
            b_new = chosen['residual']
            prev_shift_add = chosen['clears'] - (bit_length(b) - bit_length(b_new))
            a_prev_shift_add = 0
        
        step = {
            'iteration': iteration,
            'a': a,
            'b': b,
            'shift': shift_amount,
            'chosen_k': chosen['k'],
            'Q': chosen['Q'],
            'residual': chosen['residual'],
            'clears': chosen['clears'],
            'cumulative_clears': cumulative_clears,
            'update_type': update_type
        }
        steps.append(step)
        
        a, b = a_new, b_new
    
    return steps, a, b, cumulative_clears, iteration

########################################
# Reverse Reconstruction
########################################

def reverse_progression(steps, final_a, final_b):
    """
    Given the progression steps and the final (a, b) from the simulation,
    attempt to reconstruct the original (a, b) that would have produced this progression.
    
    We assume that at each forward step the following was done:
    
      Forward:
         Let the current pair be (a_old, b_old).
         Compute residual = |a_old - Q * b_old|.
         Then, if residual > b_old: update (a_new, b_new) = (residual, b_old)   [non-swap]
         else: update (a_new, b_new) = (b_old, residual)                         [swap]
         
      Reverse:
         For a non-swap step, we have:
             a_old = Q * b_old + a_new, with b_old = b_new.
         For a swap step, we have:
             a_old = Q * b_old + b_new, with b_old = a_new.
             
    We work backward from the final state.
    (Note: because the forward process is approximate and uses absolute values, this reverse
     is heuristic.)
    
    Returns (reconstructed_a, reconstructed_b).
    """
    cur_a, cur_b = final_a, final_b
    for step in reversed(steps):
        Q = step['Q']
        update_type = step['update_type']
        if update_type == "non-swap":
            # Forward: a_new = residual, b_new = b_old.
            # Reverse: prev_b = cur_b; prev_a = Q * cur_b + cur_a.
            prev_b = cur_b
            prev_a = Q * cur_b + cur_a
        else:  # swap
            # Forward: a_new = b_old, b_new = residual.
            # Reverse: prev_b = cur_a; prev_a = Q * cur_a + cur_b.
            prev_b = cur_a
            prev_a = Q * cur_a + cur_b
        cur_a, cur_b = prev_a, prev_b
    return cur_a, cur_b

########################################
# Main – Cycle over All Starting Pairs and Pick Worst-Case
########################################

if __name__ == "__main__":
    # Parameters:
    approx_bits = 4  # numbers of the form 1.xxx (for approx_bits=4, numbers 8..15)
    total_bits = 16  # target full-width (e.g., 16-bit worst-case)
    # Threshold for cumulative clears (tune this; if too high, progressions may terminate with b==0)
    threshold = 32  # try lowering threshold so we can capture progressions before termination
    rounding_mode = "truncate"  # or "round"
    integer_rounding = True

    lower = 1 << (approx_bits - 1)      # e.g., 8
    upper = (1 << approx_bits) - 1      # e.g., 15

    worst_progression = None
    worst_iter_count = -1
    worst_initial = None

    all_progressions = []  # store each progression

    for a0 in range(lower, upper + 1):
        for b0 in range(lower, a0 + 1):  # enforce a0 >= b0
            steps, final_a, final_b, cum_clears, iters = simulate_worst_case_progression(
                a0, b0, approx_bits, rounding_mode, threshold, integer_rounding)
            # Record the progression even if final_b == 0.
            record = {
                'initial_a': a0,
                'initial_b': b0,
                'steps': steps,
                'final_a': final_a,
                'final_b': final_b,
                'cum_clears': cum_clears,
                'iters': iters
            }
            all_progressions.append(record)
            if iters > worst_iter_count:
                worst_iter_count = iters
                worst_progression = record
                worst_initial = (a0, b0)

    if worst_progression is None:
        print("No valid progression found.")
        sys.exit(1)

    print("Worst-case progression among starting pairs with approx_bits = {}:".format(approx_bits))
    print("Initial pair: a = {}, b = {}  (ratio = {:.3f})".format(
          worst_progression['initial_a'],
          worst_progression['initial_b'],
          worst_progression['initial_a'] / worst_progression['initial_b']))
    print("Iterations: {}, Cumulative clears: {}\n".format(worst_progression['iters'], worst_progression['cum_clears']))
    print("Iteration-by-iteration details:")
    for step in worst_progression['steps']:
        print(" Iteration {}: a = {}, b = {}, shift = {}, chosen k = {} --> Q = {}, residual = {}, clears = {}, update = {}, cumulative clears = {}"
              .format(step['iteration'], step['a'], step['b'], step['shift'], step['chosen_k'],
                      step['Q'], step['residual'], step['clears'], step['update_type'], step['cumulative_clears']))
    
    # Report final state.
    if worst_progression['final_b'] != 0:
        ratio = worst_progression['final_a'] / worst_progression['final_b']
    else:
        ratio = float('inf')
    print("\nFinal state: a = {}, b = {}  (ratio = {})".format(
          worst_progression['final_a'], worst_progression['final_b'],
          ratio if ratio != float('inf') else "N/A (b=0)"))
    
    # Reverse reconstruction:
    reconstructed_a, reconstructed_b = reverse_progression(worst_progression['steps'],
                                                           worst_progression['final_a'],
                                                           worst_progression['final_b'])
    if reconstructed_b != 0:
        rec_ratio = reconstructed_a / reconstructed_b
    else:
        rec_ratio = float('inf')
    print("\nReverse reconstruction (approximate original full-precision candidate):")
    print("Reconstructed a = {}, b = {}  (ratio = {})".format(
          reconstructed_a, reconstructed_b,
          rec_ratio if rec_ratio != float('inf') else "N/A (b=0)"))
    
    print("\n=== End of Worst-Case Generation Experiment ===")
