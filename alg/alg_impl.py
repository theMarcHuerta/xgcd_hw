#!/usr/bin/env python3

def xgcd_bitwise(a_in, b_in, total_bits=8, approx_bits=4, rounding_mode='truncate'):
    """
    Compute the GCD of a_in and b_in using the custom 'XGCD-style' bitwise approach.
    
    :param a_in: First integer (up to 1024 bits)
    :param b_in: Second integer (up to 1024 bits)
    :param total_bits: The fixed total bit-width we assume for a and b.
                       (For safety, Python can handle more, but you can limit.)
    :param approx_bits: Number of bits for the approximate division step.
                        The first bit is treated as integer '1', and the next (approx_bits-1) bits are fractional.
    :param rounding_mode: How to handle the fractional part. Can be 'truncate', 'floor', or 'round' (to nearest),
                          or 'ceil' as you see fit.
    :return: The GCD of a_in and b_in according to the custom iteration.
    """

    #--- 1) Normalize inputs (ensure within requested bit size, pick a >= b) ---
    
    # Just to be safe, mask off anything above total_bits; user can remove if not wanted
    mask = (1 << total_bits) - 1
    a = a_in & mask
    b = b_in & mask

    # Ensure a >= b initially
    if b > a:
        a, b = b, a

    # Quick check for trivial cases
    if b == 0:
        return a
    if a == 0:
        return b
    
    #--- Helpers ---

    def bit_length(x):
        """Return the bit length of x (like x.bit_length(), but we use a helper to keep it explicit)."""
        return x.bit_length()

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
    
    def get_fixed_point_approx(x_val, approx_bits):
        """
        Extract the top `approx_bits` from x_val's leading bits and interpret as:
            1 . (approx_bits-1 fractional bits)
        If x_val has fewer than approx_bits bits, we pad the fractional bits with zero.

        Returns the *floating* or *fractional* representation (depending on rounding_mode).
        """
        # If x_val == 0, return 0.0 directly:
        if x_val == 0:
            return 0.0

        length = bit_length(x_val)
        if length <= approx_bits:
            # All bits fit inside approx_bits
            # The topmost bit is '1' (assuming x_val>0), remainder are fraction
            # Example: x_val=0b101 => length=3 => approx_bits=4 => top=101 => treat as 1.01 in binary
            top_bits = x_val  # no shift needed
            fractional_length = approx_bits - 1  # number of fractional bits
            int_bit = 1  # The leading '1'
            frac_part = top_bits & ((1 << (fractional_length)) - 1)  # just the fractional bits
            # But we might have fewer bits than fractional_length. We'll interpret accordingly.
            frac_shift = fractional_length - (length - 1)  # how many bits are actually used for fraction
            # Now build the fractional value:
            # binary fraction = frac_part / 2^(fractional_length)
            # but frac_part includes the leftover bits after the topmost '1'.
            # For instance, if top_bits=5(=0b101), fractional_length=3 => leading bit=1, fraction=0b01 => 1 + 1/4
            approx_value = 1.0 + (frac_part / (1 << fractional_length))
        else:
            # length > approx_bits => we need to cut the top 'approx_bits' bits out
            shift_down = length - approx_bits
            top_bits = x_val >> shift_down  # This extracts the top 'approx_bits' bits
            # The next bits that got shifted out are for fractional rounding decisions if needed
            leftover_bits = x_val & ((1 << shift_down) - 1)

            # The top bit is the integer '1' in the fixed-point sense, rest are fractional
            fractional_length = approx_bits - 1  # # of fraction bits
            int_bit = 1
            frac_part = top_bits & ((1 << fractional_length) - 1)
            
            # Now interpret top_bits as: 1.<frac_part> in binary
            # fraction = frac_part / 2^(fractional_length)
            fraction_value = frac_part / float(1 << fractional_length)
            approx_value = 1.0 + fraction_value

            # -- Rounding/Truncation (looking at leftover_bits) --
            # leftover_bits is what's beyond the approx_bits. If leftover_bits > 0, it might push us up or not.
            if leftover_bits > 0:
                if rounding_mode == 'truncate':
                    # do nothing; we keep approx_value as is
                    pass
                elif rounding_mode == 'floor':
                    # same as truncate for positive values
                    pass
                elif rounding_mode == 'round':
                    # round to nearest; if leftover_bits >= 2^(shift_down-1), then round up
                    half_point = 1 << (shift_down - 1) if shift_down > 0 else 0
                    if shift_down > 0 and leftover_bits >= half_point:
                        # increment fraction
                        # The full 1.XXX is stored in approx_value, so increment it by 1/(2^(fractional_length))
                        step = 1.0 / (1 << fractional_length)
                        approx_value += step
                elif rounding_mode == 'ceil':
                    # if leftover_bits > 0, push it up by 1 increment
                    step = 1.0 / (1 << fractional_length)
                    approx_value += step

        return approx_value

    while b != 0:
        #--- 2) Align b so that the leading 1 matches a's leading 1 ---
        b_aligned, shift_amount = align_b(a, b)

        # If aligning yields b_aligned = 0 (only if b=0 originally), break
        if b_aligned == 0:
            break

        #--- 3) Approximate division with approx_bits ---
        # Extract top approx_bits from a => a_approx
        a_approx = get_fixed_point_approx(a, approx_bits)
        # Extract top approx_bits from b_aligned => b_approx
        # (We use b_aligned rather than b so that the leading '1' of b_aligned lines up with a)
        b_approx = get_fixed_point_approx(b_aligned, approx_bits)

        if b_approx == 0.0: 
            # avoid division by zero
            quotient = 0.0
        else:
            quotient = a_approx / b_approx  # floating or fractional approximation
        
        #--- 4) Shift the quotient by shift_amount => multiply by 2^(shift_amount) ---
        # i.e. shifting left in binary is multiply by 2^(shift_amount)
        shifted_q = quotient * (1 << shift_amount)
        
        # The integer part of shifted_q => Q
        Q = int(shifted_q)  # floor/truncate to get the integer portion

        # b_adjusted = Q * b (the original b, not b_aligned)
        b_adjusted = b * Q

        # a_new = a - b_adjusted
        # (In normal gcd we also do positivity checks, but let's keep it direct.)
        a_new = a - b_adjusted
        if a_new < 0:
            # If negative, we might consider flipping Q or adjusting logic,
            # but in typical gcd-likes we just keep subtracting.
            # For now, just do absolute value or keep negative? We'll do absolute:
            a_new = abs(a_new)

        #--- 5) Prepare next iteration: 
        #    "the old b becomes new a, the result becomes new b"
        #    if the result > old b, swap them.
        # old b is 'b', new residual is 'a_new'
        if a_new > b:
            # then a_new becomes 'a', and b stays b
            a = a_new
        else:
            # a_new is smaller => a = b, b = a_new
            a, b = b, a_new

    # When b=0, a is the GCD
    return a

# -------------------------------------------------------------------------
# A small demo/test
if __name__ == "__main__":
    # Example usage:
    import sys

    # If you want interactive input:
    # a_in = int(input("Enter first number: "))
    # b_in = int(input("Enter second number: "))

    # Or hard-code a small example
    a_in = 128
    b_in = 56

    gcd_val = xgcd_bitwise(a_in, b_in,
                           total_bits=8,
                           approx_bits=4,
                           rounding_mode='truncate')
    print(f"GCD of {a_in} and {b_in} (via custom XGCD) is {gcd_val}")
