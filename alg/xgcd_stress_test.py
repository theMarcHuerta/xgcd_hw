#!/usr/bin/env python3
import random
import statistics
import sys
from xgcd_impl import xgcd_bitwise
from typing import Tuple, List

# -------------------------------------------------------------------------
def run_xgcd_stress_test(num_pairs=100,
                         total_bits=256,
                         approx_bits=4):
    """
    Generates 'num_pairs' random pairs (a, b) each 'total_bits' wide.
    Runs xgcd_bitwise on each pair in both truncate and round mode.
    Collects iteration counts and avg bit clears, then prints summary stats.

    Returns:
      None (prints results).
    """

    # We'll store results in lists:
    # For TRUNC
    trunc_iter_counts = []
    trunc_bit_clears = []
    # Keep track of max iteration pair, min bit-clear pair, etc.
    trunc_max_iter = -1
    trunc_max_iter_pair = (0,0)
    trunc_min_clears = float('inf')
    trunc_min_clears_pair = (0,0)

    # For ROUND
    round_iter_counts = []
    round_bit_clears = []
    # Keep track of max iteration pair, min bit-clear pair, etc.
    round_max_iter = -1
    round_max_iter_pair = (0,0)
    round_min_clears = float('inf')
    round_min_clears_pair = (0,0)

    # We'll generate the random pairs up front so we use the SAME pairs for both truncate & round
    pairs = []
    for _ in range(num_pairs):
        a = random.getrandbits(total_bits)
        b = random.getrandbits(total_bits)
        # ensure not both zero
        if a==0 and b==0:
            a=1
        pairs.append((a, b))

    print(f"Running XGCD stress test with {num_pairs} random pairs, {total_bits}-bit each, approx_bits={approx_bits}.")
    print("Comparing 'truncate' vs 'round' on the same pairs.")
    print("Progress: 0%", end="", flush=True)

    for i, (a, b) in enumerate(pairs, start=1):
        # 1) Truncate
        gcd_t, iter_t, clears_t = xgcd_bitwise(a, b,
                                              total_bits=total_bits,
                                              approx_bits=approx_bits,
                                              integer_rounding=True,
                                              plus_minus=False,
                                              rounding_mode='truncate')
        trunc_iter_counts.append(iter_t)
        trunc_bit_clears.append(clears_t)

        # Update best/worst
        if iter_t > trunc_max_iter:
            trunc_max_iter = iter_t
            trunc_max_iter_pair = (a, b)
        if clears_t < trunc_min_clears:
            trunc_min_clears = clears_t
            trunc_min_clears_pair = (a, b)

        # 2) Round
        gcd_r, iter_r, clears_r = xgcd_bitwise(a, b,
                                              total_bits=total_bits,
                                              approx_bits=approx_bits,
                                              integer_rounding=True,
                                              plus_minus=False,
                                              rounding_mode='round')
        round_iter_counts.append(iter_r)
        round_bit_clears.append(clears_r)

        # Update best/worst
        if iter_r > round_max_iter:
            round_max_iter = iter_r
            round_max_iter_pair = (a, b)
        if clears_r < round_min_clears:
            round_min_clears = clears_r
            round_min_clears_pair = (a, b)

        # Show a simple text progress bar
        # e.g. every 5% or final
        pct = 100*i//num_pairs
        if i == num_pairs or (pct % 5 == 0):
            print(f"\rProgress: {pct}%", end="", flush=True)

    print("\nDone.\n")

    # ------------------------------------------------------------------
    # Summarize results
    # ------------------------------------------------------------------
    # Means
    trunc_iter_mean = statistics.mean(trunc_iter_counts) if len(trunc_iter_counts)>0 else 0
    trunc_clears_mean = statistics.mean(trunc_bit_clears) if len(trunc_bit_clears)>0 else 0
    round_iter_mean = statistics.mean(round_iter_counts) if len(round_iter_counts)>0 else 0
    round_clears_mean = statistics.mean(round_bit_clears) if len(round_bit_clears)>0 else 0

    # Medians
    trunc_iter_median = statistics.median(trunc_iter_counts) if len(trunc_iter_counts)>0 else 0
    trunc_clears_median = statistics.median(trunc_bit_clears) if len(trunc_bit_clears)>0 else 0
    round_iter_median = statistics.median(round_iter_counts) if len(round_iter_counts)>0 else 0
    round_clears_median = statistics.median(round_bit_clears) if len(round_bit_clears)>0 else 0

    # Print them out
    print("===== RESULTS (TRUNCATE) =====")
    print(f"  Mean Iterations     : {trunc_iter_mean:.3f}")
    print(f"  Median Iterations   : {trunc_iter_median:.3f}")
    print(f"  Mean Bit Clears     : {trunc_clears_mean:.3f}")
    print(f"  Median Bit Clears   : {trunc_clears_median:.3f}")
    print(f"  Max Iterations      : {trunc_max_iter} for pair a={trunc_max_iter_pair[0]}, b={trunc_max_iter_pair[1]}")
    print(f"  Min Avg Bit Clears  : {trunc_min_clears:.3f} for pair a={trunc_min_clears_pair[0]}, b={trunc_min_clears_pair[1]}")

    print("\n===== RESULTS (ROUND) =====")
    print(f"  Mean Iterations     : {round_iter_mean:.3f}")
    print(f"  Median Iterations   : {round_iter_median:.3f}")
    print(f"  Mean Bit Clears     : {round_clears_mean:.3f}")
    print(f"  Median Bit Clears   : {round_clears_median:.3f}")
    print(f"  Max Iterations      : {round_max_iter} for pair a={round_max_iter_pair[0]}, b={round_max_iter_pair[1]}")
    print(f"  Min Avg Bit Clears  : {round_min_clears:.3f} for pair a={round_min_clears_pair[0]}, b={round_min_clears_pair[1]}")

    print("\n--- End of test ---\n")

def main():
    """
    Example command-line usage or just run defaults.
    """
    import argparse
    parser = argparse.ArgumentParser(description="XGCD Stress Test Script.")
    parser.add_argument("--num_pairs", type=int, default=1000000, help="Number of random pairs to test.")
    parser.add_argument("--total_bits", type=int, default=256, help="Bit-width of the random numbers.")
    parser.add_argument("--approx_bits", type=int, default=2, help="Approx bits for the XGCD.")
    args = parser.parse_args()

    run_xgcd_stress_test(num_pairs=args.num_pairs,
                         total_bits=args.total_bits,
                         approx_bits=args.approx_bits)

if __name__ == "__main__":
    main()



# # default usage:
# python3 xgcd_stress_test.py

# # or specify your own arguments, e.g. 500 pairs, 512-bit numbers, approx_bits=8
# python3 xgcd_stress_test.py --num_pairs 500 --total_bits 512 --approx_bits 8