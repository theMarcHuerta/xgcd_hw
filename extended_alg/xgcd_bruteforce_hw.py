#!/usr/bin/env python3
import sys
import statistics
import argparse
import math 
from xgcd_hw_lite import xgcd_bitwise  # Ensure xgcd_bitwise returns (gcd, iterations, avg_bit_clears)

def brute_force_xgcd(bits=8, 
                     approx_bits=4, 
                     skip_symmetry=True, 
                     skip_zeros=True,
                     force_a_msb=True,
                     int_rounding=True):
    """
    Enumerates (a, b) within 'bits'-bit range, applying custom filters:
      - skip_symmetry  => avoid duplicates (b,a) if we've done (a,b)
      - skip_zeros     => exclude cases where a=0 or b=0
      - force_a_msb    => ensure a >= 2^(bits-1), i.e. top bit of 'a' is 1

    Then for each valid (a,b), runs xgcd_bitwise in both 'truncate' and 'round' mode,
    collecting iteration counts & average bit clears to produce summary stats.
    """

    # Range for 'a'
    if force_a_msb:
        a_min = 1 << (bits - 1)       # e.g. for bits=8 => a_min=128
        a_max = (1 << bits) - 1       # e.g. for bits=8 => a_max=255
    else:
        a_min = 0
        a_max = (1 << bits) - 1
    
    # We'll define the same range for 'b'.
    b_min = 0
    b_max = (1 << bits) - 1

    # We'll figure out how many pairs this might be for progress reporting
    # We'll approximate the total by counting how many a's are valid times
    # how many b's are valid, factoring in skip_symmetry, skip_zeros, etc.
    # But let's just do a naive upper bound for a progress indicator.
    possible_a = (a_max - a_min + 1)
    possible_b = (b_max - b_min + 1)
    naive_total = possible_a * possible_b
    
    print(f"Brute forcing all pairs for {bits}-bit range.")
    print(f"Options: skip_symmetry={skip_symmetry}, skip_zeros={skip_zeros}, force_a_msb={force_a_msb}, integer_rounding={int_rounding}")
    print(f"Approx bits = {approx_bits}.\n")

    counter = 0
    valid_counter = 0  # how many (a,b) we actually test

    # Corrected naive_total computation
    naive_total = sum(a for a in range(a_min, a_max + 1)) - 13

    for a in range(a_min, a_max + 1):
        for b in range(1, a + 1):  # Ensure b is in [1, a] to uphold a >= b

            # No need to check for `skip_zeros` since b starts at 1.
            # No need to check for `skip_symmetry` since we limit b <= a.

            valid_counter += 1

            # Run XGCD in 'truncate' mode
            result = xgcd_bitwise(a, b,
                                total_bits=bits,
                                approx_bits=3)
            
            gcd_t, x, y = result

            # Verify against Python's built-in GCD function
            expected_gcd = math.gcd(a, b)
            if gcd_t != expected_gcd:
                print(f"ERROR: Mismatch for (a={a}, b={b}) → xgcd_bitwise() returned {gcd_t}, but math.gcd() says {expected_gcd}")

            # Track max/min cases
            # Progress indicator
            counter += 1
            if counter % (naive_total // 50 + 1) == 0:
                pct = 100.0 * counter / naive_total
                print(f"\rProgress: {counter} / {naive_total}  ({pct:.1f}%)", end="", flush=True)


    print("\n\nDone. Tested a total of", valid_counter, "valid pairs.\n")


def main():
    parser = argparse.ArgumentParser(description="Brute force XGCD over (a,b) for a given bit-width, with extra filters.")
    parser.add_argument("--bits", type=int, default=10, help="Bit width for enumerating all pairs.")
    parser.add_argument("--approx_bits", type=int, default=3, help="Approx bits for XGCD.")
    parser.add_argument("--skip_symmetry", action="store_true", 
                        help="If set, skip symmetrical pairs (b,a) if we've done (a,b).")
    parser.add_argument("--skip_zeros", action="store_true", 
                        help="If set, skip any pairs where a=0 or b=0.")
    parser.add_argument("--force_a_msb", action="store_true", 
                        help="If set, 'a' will always have top bit=1, i.e. range [2^(bits-1) .. (2^bits)-1].")

    args = parser.parse_args()

    brute_force_xgcd(bits=args.bits,
                     approx_bits=args.approx_bits,
                     skip_symmetry=True,
                     skip_zeros=True,
                     force_a_msb=True,
                     int_rounding=True)


if __name__ == "__main__":
    main()
