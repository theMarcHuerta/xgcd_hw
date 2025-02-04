import sys

def largest_fibonacci_numbers(n_bits):
    """Find the two largest Fibonacci numbers that fit in an n-bit unsigned integer."""
    max_value = 2**n_bits - 1

    # Start Fibonacci sequence
    fib_numbers = [0, 1]

    while True:
        next_fib = fib_numbers[-1] + fib_numbers[-2]
        if next_fib > max_value:
            break
        fib_numbers.append(next_fib)

    # The last two valid Fibonacci numbers
    largest_fib = fib_numbers[-1]
    second_largest_fib = fib_numbers[-2]

    return hex(second_largest_fib), hex(largest_fib)

if __name__ == "__main__":
    # Check if n_bits is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <n_bits>")
        sys.exit(1)

    try:
        n_bits = int(sys.argv[1])
        if n_bits <= 0 or n_bits > 1024:
            raise ValueError("n_bits must be between 1 and 1024.")

        fib1_hex, fib2_hex = largest_fibonacci_numbers(n_bits)
        print(f"Second largest Fibonacci number (hex): {fib1_hex}")
        print(f"Largest Fibonacci number (hex): {fib2_hex}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
