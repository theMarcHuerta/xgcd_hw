#!/usr/bin/env python3
import math
import statistics

########################################
# Basic helper functions (unchanged)
########################################

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
# Simulation of one division iteration
########################################

def simulate_iteration1(a, b, approx_bits, rounding_mode, integer_rounding=True):
    """
    Simulate one division step (iteration 1) for fixed‐point numbers a and b.
    Both a and b are assumed to have approx_bits bits (normalized: msb=1).
    
    Returns a tuple:
       (Q, residual, clears)
    where Q is the integer quotient (computed via the LUT method and rounding/truncation),
    residual = |a - (Q * b)|, and clears = bit_length(a) - bit_length(residual) (clamped to 0).
    (Since a and b have the same bit‐length in iteration 1, shift_amount is 0.)
    """
    # Extract top bits
    a_top = get_fixed_top_bits(a, approx_bits)
    b_top = get_fixed_top_bits(b, approx_bits)
    # Compute LUT quotient (this yields a fixed-point value with approx_bits fractional bits)
    quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
    # For iteration 1, shift_amount is 0.
    # Emulate the “rounding” logic:
    Q_pre_round = quotient >> (approx_bits - 1)  # since shift_amount is 0
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and integer_rounding:
        Q += 1
    # Multiply b by Q and subtract from a
    b_adjusted = b * Q
    residual = a - b_adjusted
    if residual < 0:
        residual = -residual
    # Calculate bit clears:
    clears = bit_length(a) - bit_length(residual)
    if clears < 0:
        clears = 0
    return Q, residual, clears

def simulate_iteration2(a, b, branch_bit, approx_bits, rounding_mode, integer_rounding=True):
    """
    Simulate one division step (iteration 2) for a new pair:
       new dividend = a, new divisor = b.
    Here we first align b with a (as in the xgcd algorithm). Then we consider a branch:
       if branch_bit == 0, we leave b_aligned as is;
       if branch_bit == 1, we set the least-significant bit of b_aligned to 1.
       
    Then we compute the quotient and resulting residual in the same way.
    
    Returns a tuple: (Q, residual, clears, shift_amount, b_aligned_candidate)
    """
    # Align b with a:
    b_aligned, shift_amount = align_b(a, b)
    # Modify b_aligned according to branch_bit:
    if branch_bit == 1:
        b_aligned_candidate = b_aligned | 1
    else:
        b_aligned_candidate = b_aligned
    # Extract top bits:
    a_top = get_fixed_top_bits(a, approx_bits)
    b_top = get_fixed_top_bits(b_aligned_candidate, approx_bits)
    quotient = lut_result(a_top, b_top, approx_bits, rounding_mode)
    # Adjust quotient by the shift (as in the algorithm):
    Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1)
    Q = Q_pre_round >> 1
    if (Q_pre_round & 1) and integer_rounding:
        Q += 1
    # Multiply the original divisor b by Q (note: we use the unshifted b)
    b_adjusted = b * Q
    residual = a - b_adjusted
    if residual < 0:
        residual = -residual
    clears = bit_length(a) - bit_length(residual)
    if clears < 0:
        clears = 0
    return Q, residual, clears, shift_amount, b_aligned_candidate

########################################
# Extended LUT Analysis over 2 iterations
########################################

def lut_analysis_extended(approx_bits, rounding_mode="truncate", integer_rounding=True):
    """
    For all fixed-point numbers a and b of the form 1.f (with total approx_bits bits)
    (i.e. a, b in [2^(approx_bits-1), 2^(approx_bits)-1]),
    perform two iterations of the division process.
    
    In iteration 1, compute Q1, residual1, and bit clears.
    Then let new dividend = b and new divisor = residual1.
    
    In iteration 2, for each branch (branch_bit=0 and branch_bit=1) during the alignment step,
    compute Q2, residual2, and bit clears.
    
    Returns a list of records, where each record is a dictionary containing:
       - 'a': initial a
       - 'b': initial b
       - 'iter1': { 'Q': Q1, 'residual': residual1, 'clears': clears1 }
       - 'iter2_branch0': { 'branch': 0, 'Q': Q2_0, 'residual': residual2_0, 'clears': clears2_0,
                              'shift': shift_amount, 'b_aligned': b_aligned_candidate (for branch 0) }
       - 'iter2_branch1': { 'branch': 1, 'Q': Q2_1, 'residual': residual2_1, 'clears': clears2_1,
                              'shift': shift_amount, 'b_aligned': b_aligned_candidate (for branch 1) }
       - 'total_clears_branch0': clears1 + clears2_0
       - 'total_clears_branch1': clears1 + clears2_1
    """
    results = []
    lower = 1 << (approx_bits - 1)
    upper = (1 << approx_bits) - 1
    for a in range(lower, upper + 1):
        for b in range(lower, upper + 1):
            record = {}
            record['a'] = a
            record['b'] = b
            # Iteration 1:
            Q1, residual1, clears1 = simulate_iteration1(a, b, approx_bits, rounding_mode, integer_rounding)
            record['iter1'] = {'Q': Q1, 'residual': residual1, 'clears': clears1}
            # Prepare for iteration 2:
            new_a = b         # old b becomes new dividend
            new_b = residual1   # residual becomes new divisor
            # Only proceed if new_b is nonzero
            if new_b == 0:
                # Mark iteration 2 as N/A
                record['iter2_branch0'] = None
                record['iter2_branch1'] = None
                record['total_clears_branch0'] = clears1
                record['total_clears_branch1'] = clears1
            else:
                # Branch 0:
                Q2_0, residual2_0, clears2_0, shift0, b_aligned0 = simulate_iteration2(new_a, new_b, 0, approx_bits, rounding_mode, integer_rounding)
                # Branch 1:
                Q2_1, residual2_1, clears2_1, shift1, b_aligned1 = simulate_iteration2(new_a, new_b, 1, approx_bits, rounding_mode, integer_rounding)
                record['iter2_branch0'] = {'branch': 0, 'Q': Q2_0, 'residual': residual2_0, 'clears': clears2_0,
                                           'shift': shift0, 'b_aligned': b_aligned0}
                record['iter2_branch1'] = {'branch': 1, 'Q': Q2_1, 'residual': residual2_1, 'clears': clears2_1,
                                           'shift': shift1, 'b_aligned': b_aligned1}
                record['total_clears_branch0'] = clears1 + clears2_0
                record['total_clears_branch1'] = clears1 + clears2_1
            results.append(record)
    return results

########################################
# Statistics and Reporting
########################################

def report_lut_analysis(results):
    """
    Given the list of records from lut_analysis_extended, compute and print:
       - The distribution (mean, median, min, max) of clears for iteration 1,
         iteration 2 (for branch 0 and branch 1) and the total clears.
       - Also list the cases (a,b) with minimal total clears.
    """
    iter1_clears = []
    iter2_0_clears = []
    iter2_1_clears = []
    total_clears_0 = []
    total_clears_1 = []
    
    for rec in results:
        if rec['iter1'] is not None:
            iter1_clears.append(rec['iter1']['clears'])
        if rec['iter2_branch0'] is not None:
            iter2_0_clears.append(rec['iter2_branch0']['clears'])
            iter2_1_clears.append(rec['iter2_branch1']['clears'])
            total_clears_0.append(rec['total_clears_branch0'])
            total_clears_1.append(rec['total_clears_branch1'])
    
    print("=== LUT Extended Analysis ===")
    if iter1_clears:
        print("Iteration 1 Clears: mean = {:.3f}, median = {:.3f}, min = {}, max = {}"
              .format(statistics.mean(iter1_clears), statistics.median(iter1_clears),
                      min(iter1_clears), max(iter1_clears)))
    if iter2_0_clears:
        print("Iteration 2 Branch 0 Clears: mean = {:.3f}, median = {:.3f}, min = {}, max = {}"
              .format(statistics.mean(iter2_0_clears), statistics.median(iter2_0_clears),
                      min(iter2_0_clears), max(iter2_0_clears)))
    if iter2_1_clears:
        print("Iteration 2 Branch 1 Clears: mean = {:.3f}, median = {:.3f}, min = {}, max = {}"
              .format(statistics.mean(iter2_1_clears), statistics.median(iter2_1_clears),
                      min(iter2_1_clears), max(iter2_1_clears)))
    if total_clears_0:
        print("Total Clears (iter1 + branch0): mean = {:.3f}, median = {:.3f}, min = {}, max = {}"
              .format(statistics.mean(total_clears_0), statistics.median(total_clears_0),
                      min(total_clears_0), max(total_clears_0)))
    if total_clears_1:
        print("Total Clears (iter1 + branch1): mean = {:.3f}, median = {:.3f}, min = {}, max = {}"
              .format(statistics.mean(total_clears_1), statistics.median(total_clears_1),
                      min(total_clears_1), max(total_clears_1)))
    
    # Report cases with minimal total clears for each branch:
    min_total_0 = min(total_clears_0) if total_clears_0 else None
    min_total_1 = min(total_clears_1) if total_clears_1 else None
    print("\nCases with minimal total clears (Branch 0):")
    for rec in results:
        if rec.get('total_clears_branch0', None) == min_total_0:
            print("  a = {}, b = {}  -> iter1 clears = {}, iter2 clears = {}, total = {}"
                  .format(rec['a'], rec['b'], rec['iter1']['clears'],
                          rec['iter2_branch0']['clears'] if rec['iter2_branch0'] else "N/A",
                          rec['total_clears_branch0']))
    print("\nCases with minimal total clears (Branch 1):")
    for rec in results:
        if rec.get('total_clears_branch1', None) == min_total_1:
            print("  a = {}, b = {}  -> iter1 clears = {}, iter2 clears = {}, total = {}"
                  .format(rec['a'], rec['b'], rec['iter1']['clears'],
                          rec['iter2_branch1']['clears'] if rec['iter2_branch1'] else "N/A",
                          rec['total_clears_branch1']))

########################################
# Main – Run the extended LUT analysis
########################################

if __name__ == "__main__":
    # Set approx_bits (e.g., 4 means numbers of the form 1.xxx, with 3 fractional bits)
    approx_bits = 4
    rounding_mode = "truncate"  # or "round"
    # Run the extended analysis:
    results = lut_analysis_extended(approx_bits, rounding_mode, integer_rounding=True)
    report_lut_analysis(results)
