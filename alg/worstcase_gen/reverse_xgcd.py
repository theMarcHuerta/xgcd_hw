#!/usr/bin/env python3
"""
Searches for "worst-case" growth sequences for an XGCD-like iteration from the bottom up.

We start with b_prev and b_curr.
At each step, we try all possible values of q in [-q_max, ..., -1, 1, ..., q_max].
We compute b_next = b_prev + (b_curr * q).

We only keep transitions for which b_next >= b_curr (mimicking the top-down requirement "a >= b").
Then, for the next iteration, (b_prev, b_curr) := (b_curr, b_next).

We do this for 'max_iter' total iterations in a depth-first manner. 
Whenever we see that b_next and b_curr have the same bit-length, 
we print them (labeling them A = b_next, B = b_curr).

Usage Example:
  python3 bottom_up_worstcase.py <b_prev> <b_curr> <max_iter> <q_max>

Note:
  This search can explode combinatorially. For q_max=6 and max_iter=6, 
  you could have up to 12^6 = 2,985,984 paths explored.
  Use smaller params or add pruning if needed.
"""

import math
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgcd_impl import xgcd_bitwise

##########################
# PARAMETERS
##########################

MAX_Q            = 6          
MAX_ITER         = 8          

def explore_sequences(b_prev, b_curr, iteration, max_iter, q_max, q_trace):
    """
    Recursively explore all sequences up to max_iter depth, 
    building b_next from b_prev, b_curr, and q in +/-[1..q_max].
    """
    if iteration >= max_iter:
        return
    
    # Try all possible q in {±1, ±2, ..., ±q_max}, skipping 0
    for q in range(1, q_max+1):
        for sign in (1, -1):

            b_next = (b_curr * q) + (sign * b_prev)
            # Enforce 'b_next >= b_curr' as the valid path
            # (You could also require b_next >= 0 if needed.)
            if b_next < b_curr or b_next < 0:
                continue

            q_trace.append(q*sign)

            iteration_len = 0
            if (b_next.bit_length() == b_curr.bit_length()):
                gcd_val, iteration_len, avg_clears = xgcd_bitwise(b_next, b_curr,
                                                            b_next.bit_length(),
                                                            4,
                                                            'truncate',
                                                            True,
                                                            False,
                                                            False)

            # Check bit-length equality
            if b_curr != 0 and b_next.bit_length() == b_curr.bit_length() and b_next.bit_length() == (iteration_len+1):
                # Print if they share the same bit length
                print(f"Iteration {iteration+1}: A={b_next} (bitlen={b_next.bit_length()}), "
                      f"B={b_curr} (bitlen={b_curr.bit_length()}), q={q}, q_trace={q_trace}")
                q_trace.pop()
                continue

            # Recurse: shift "current" forward
            explore_sequences(b_curr, b_next, iteration+1, max_iter, q_max, q_trace)

            # Done exploring that path, so pop the q we appended
            q_trace.pop()


def main():

    b_prev  = 286258
    b_curr  = 655821

    q_trace = []

    explore_sequences(b_prev, b_curr, 0, MAX_ITER, MAX_Q, q_trace)


if __name__ == "__main__":
    main()
