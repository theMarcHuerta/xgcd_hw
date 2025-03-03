#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
from xgcd_impl import xgcd_bitwise
from xgcd_new import xgcd_bitwise_new

def largest_fibonacci_numbers(n_bits):
    """Find the two largest Fibonacci numbers that fit in an n-bit unsigned integer."""
    max_value = 2**n_bits - 1
    fib_numbers = [0, 1]
    while True:
        next_fib = fib_numbers[-1] + fib_numbers[-2]
        if next_fib > max_value:
            break
        fib_numbers.append(next_fib)
    # Return the two largest valid Fibonacci numbers.
    return fib_numbers[-2], fib_numbers[-1]

# Lists to hold the bit sizes and iteration counts for both implementations and rounding modes.
bit_sizes = []
impl_round_true = []
impl_round_false = []
new_round_true = []
new_round_false = []

# Loop over bit sizes (from 8 to 512 in steps of 8)
for n in range(64, 4097, 64):
    # Generate the two largest Fibonacci numbers for the current bit size.
    fib1, fib2 = largest_fibonacci_numbers(n)
    
    # Run xgcd_bitwise (original implementation) for both rounding modes.
    _, count_impl_true, _ = xgcd_bitwise(
        fib1, fib2, total_bits=n, approx_bits=4,
        rounding_mode='truncate', integer_rounding=True,
        plus_minus=False, enable_plotting=False
    )
    _, count_impl_false, _ = xgcd_bitwise(
        fib1, fib2, total_bits=n, approx_bits=4,
        rounding_mode='truncate', integer_rounding=False,
        plus_minus=False, enable_plotting=False
    )
    
    # Run xgcd_bitwise_new (new implementation) for both rounding modes.
    _, count_new_true, _ = xgcd_bitwise_new(
        fib1, fib2, total_bits=n, approx_bits=4,
        rounding_mode='truncate', integer_rounding=True,
        plus_minus=False, enable_plotting=False
    )
    
    bit_sizes.append(n)
    impl_round_true.append(count_impl_true)
    impl_round_false.append(count_impl_false)
    new_round_true.append(count_new_true)
    
    print(f"{n}-bit: impl: (True) = {count_impl_true}, (False) = {count_impl_false} | new: = {count_new_true}")

# Plot all four curves on the same figure.
plt.figure(figsize=(10, 6))
plt.plot(bit_sizes, impl_round_true, marker='o', color='blue', label="Impl: Integer Rounding True")
plt.plot(bit_sizes, impl_round_false, marker='x', color='blue', label="Impl: Integer Rounding False")
plt.plot(bit_sizes, new_round_true, marker='o', color='red', label="New")
plt.xlabel("Bit Size (n)")
plt.ylabel("XGCD Iteration Count")
plt.title("XGCD Iterations for Fibonacci Numbers vs. Bit Size")
plt.legend()
plt.grid(True)
plt.show()
