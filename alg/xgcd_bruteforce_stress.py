#!/usr/bin/env python3
import sys
import statistics
import argparse
from xgcd_impl import xgcd_bitwise

def brute_force_xgcd(bits=8, approx_bits=4):
    """
    Enumerates all (a, b) with 'bits'-bit values (i.e., from 0..(2^bits - 1)),
    runs xgcd_bitwise in both truncate and round modes,
    and prints summary statistics (mean, median, worst iteration, etc.).
    
    NOTE: This is 2^(2*bits) total pairs minus (0,0). 
    So be careful with large 'bits'.
    """
    max_val = (1 << bits) - 1
    total_pairs = (1 << bits) * (1 << bits) - 1  # minus the (0,0) case

    # We'll store iteration counts and bit clears for each mode
    trunc_iters = []
    trunc_clears = []
    trunc_max_iter = -1
    trunc_max_iter_pair = (0, 0)
    trunc_min_clears = float('inf')
    trunc_min_clears_pair = (0, 0)

    round_iters = []
    round_clears = []
    round_max_iter = -1
    round_max_iter_pair = (0, 0)
    round_min_clears = float('inf')
    round_min_clears_pair = (0, 0)

    print(f"Brute forcing all pairs for {bits} bits (up to {max_val}) = {2**(2*bits)} total combos.")
    print(f"Skipping (0,0). Approx bits = {approx_bits}\n")

    counter = 0
    # Simple loop
    for a in range(max_val+1):
        for b in range(max_val+1):
            if a == 0 and b == 0:
                continue

            counter += 1

            # 1) Truncate
            gcd_t, it_t, clr_t = xgcd_bitwise(a, b,
                                             total_bits=bits,
                                             approx_bits=approx_bits,
                                             rounding_mode='truncate')
            trunc_iters.append(it_t)
            trunc_clears.append(clr_t)

            # update extremes
            if it_t > trunc_max_iter:
                trunc_max_iter = it_t
                trunc_max_iter_pair = (a, b)
            if clr_t < trunc_min_clears:
                trunc_min_clears = clr_t
                trunc_min_clears_pair = (a, b)

            # 2) Round
            gcd_r, it_r, clr_r = xgcd_bitwise(a, b,
                                             total_bits=bits,
                                             approx_bits=approx_bits,
                                             rounding_mode='round')
            round_iters.append(it_r)
            round_clears.append(clr_r)

            # update extremes
            if it_r > round_max_iter:
                round_max_iter = it_r
                round_max_iter_pair = (a, b)
            if clr_r < round_min_clears:
                round_min_clears = clr_r
                round_min_clears_pair = (a, b)

            # Progress update
            # Let's do a simple text progress every 2% or final
            if counter % (total_pairs // 50 + 1) == 0:  # roughly 2% steps
                pct = 100.0 * counter / total_pairs
                print(f"\rProcessed {counter} / {total_pairs}  ({pct:.1f}%)", end="", flush=True)

    print("\n\nDone.\n")

    # Summarize

    def safe_mean(x):
        return statistics.mean(x) if len(x) else 0
    def safe_median(x):
        return statistics.median(x) if len(x) else 0

    # Truncate stats
    trunc_iter_mean = safe_mean(trunc_iters)
    trunc_iter_median = safe_median(trunc_iters)
    trunc_clears_mean = safe_mean(trunc_clears)
    trunc_clears_median = safe_median(trunc_clears)

    # Round stats
    round_iter_mean = safe_mean(round_iters)
    round_iter_median = safe_median(round_iters)
    round_clears_mean = safe_mean(round_clears)
    round_clears_median = safe_median(round_clears)

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

    print("\n--- End of brute force test ---\n")

def main():
    parser = argparse.ArgumentParser(description="Brute force XGCD over all pairs for a given bit-width.")
    parser.add_argument("--bits", type=int, default=12, help="Bit width for enumerating all pairs (0..(2^bits -1)).")
    parser.add_argument("--approx_bits", type=int, default=4, help="Approximation bits for the XGCD.")
    args = parser.parse_args()

    # sanity check
    if args.bits > 16:
        print("WARNING: bits>16 => 2^(2*bits) can be extremely large! Proceed with caution.")

    brute_force_xgcd(bits=args.bits, approx_bits=args.approx_bits)

if __name__ == "__main__":
    main()
