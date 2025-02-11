import math

def xgcd(a, b, constant_time=False, bitwidth=64):
    """
    Inputs:
      a, b (int): integers to find the gcd for (assume any factors of 2 have already been divided out, so at most one is even)
      constant_time (bool): whether the execution should pad out to 'constant time'
                            iterations
      bitwidth (int): maximum bitwidth of the inputs (used if constant_time=True)
    
    Outputs:
      gcd  (int): gcd(a, b)
      x0, y0 (int): Bézout coefficients satisfying a*x0 + b*y0 = gcd(a, b)
    """
    # -------------------------------------------------------------------------
    # Stage 1: Pre-processing
    #
    # Since common factors of two have already been removed from the inputs,
    # at most one input can be even. This step aligns inputs for the reduction
    # stage so that (a0, b0) are odd if needed:
    # - If we need to make 'a' odd, adding 1 to an even 'a' and an odd 'b' 
    #   still results in an odd a0.
    # - If we need to make 'b' odd, similarly add 1 to 'b'.
    # -------------------------------------------------------------------------

    if (a % 2 == 0) and (b % 2 == 1):
        a0 = a + 1
        b0 = b
    elif (b % 2 == 0) and (a % 2 == 1):
        a0 = a
        b0 = b + 1
    else:
        # Both are already odd or both even in some edge case
        a0 = a
        b0 = b

    # If we want a strictly constant number of loop iterations:
    if constant_time:
        iteration = 0
        constant_time_iterations = int(math.ceil(1.51 * bitwidth + 1))
    else:
        iteration = None
        constant_time_iterations = None

    # -------------------------------------------------------------------------
    # Stage 2: Reduction
    #
    # We'll track the Bézout coefficients in (u1,u2) and (v1,v2), so that:
    #    a0 * u1 + b0 * u2   and   a0 * v1 + b0 * v2
    # transform as we do divisions/shifts.
    #
    # 'delta' sometimes represents an approximation of log2(a0) - log2(b0)
    # to guess if a0 > b0.  We'll just initialize it to 0 (or something) here.
    #
    # We'll loop until either (a0 == 0 or b0 == 0) in the normal (non-constant) case,
    # or until we've done 'constant_time_iterations' loops if constant_time is True.
    # -------------------------------------------------------------------------
    u1, u2 = 1, 0
    v1, v2 = 0, 1
    delta = 0
    end_loop = False

    # Helper function to update XGCD coefficients by dividing them by 2^num_bits_reduced
    # while keeping track of the fact that each might be odd (so we can add b0 or a0 before halving).
    def xgcd_update(num_bits_reduced, a0_val, b0_val, u1_val, u2_val, v1_val, v2_val):
        """
        num_bits_reduced times, do:
          - if u1 is odd, add b0 then halve
          - if u1 is even, just halve
          - if v1 is odd, add a0 then halve
          - if v1 is even, just halve
          - similarly for u2, v2 if needed
        (The screenshot mostly focuses on adjusting (u1,v1), but you may
         need symmetrical logic for (u2,v2) as well, depending on your use‐case.)
        """
        for _ in range(num_bits_reduced):
            # If u1 is odd, add b0 before dividing by 2
            if (u1_val % 2) == 1:
                u1_val = (u1_val + b0_val) // 2
            else:
                u1_val = u1_val // 2

            # If v1 is odd, add a0 before dividing by 2
            if (v1_val % 2) == 1:
                v1_val = (v1_val + a0_val) // 2
            else:
                v1_val = v1_val // 2

            # For completeness, do the same with u2, v2 if needed
            if (u2_val % 2) == 1:
                u2_val = (u2_val + b0_val) // 2
            else:
                u2_val = u2_val // 2

            if (v2_val % 2) == 1:
                v2_val = (v2_val + a0_val) // 2
            else:
                v2_val = v2_val // 2

        return (u1_val, u2_val, v1_val, v2_val)

    # Main reduction loop
    while not end_loop:
        # ---------------------------------------------------------------------
        # tries various "cases" based on a0 mod 4, b0 mod 4,
        # or the sign of (a0 - b0). 
        # They do partial steps (div by 2, or +/- some multiple)
        # and then call xgcd_update(...) to fix up the coefficient vectors.
        # ---------------------------------------------------------------------

        # Case A: a0 % 4 == 0
        if (not constant_time) and (a0 % 4 == 0):
            # Halve a0 by 1 bit, and update coefficients by 1 bit
            (u1, u2, v1, v2) = xgcd_update(1, a0, b0, u1, u2, v1, v2)
            a0 //= 2
            delta -= 1

        # Case B: a0 % 4 == 1
        elif (not constant_time) and (a0 % 4 == 1):
            # Subtract b0, then halve by 2 bits
            a0 = (a0 - b0) // 2
            (u1, u2, v1, v2) = xgcd_update(2, a0, b0, u1, u2, v1, v2)
            delta += 1

        # Case C: a0 % 4 == 2
        elif (not constant_time) and (a0 % 4 == 2):
            # Just halve a0 by 1 bit, no +/- b0 first
            a0 //= 2
            (u1, u2, v1, v2) = xgcd_update(1, a0, b0, u1, u2, v1, v2)
            # delta might stay the same or adjust if the screenshot indicates so
            # e.g. delta += 0

        # Case D: a0 % 4 == 3
        elif (not constant_time) and (a0 % 4 == 3):
            # Add b0, then halve by 2 bits
            a0 = (a0 + b0) // 2
            (u1, u2, v1, v2) = xgcd_update(2, a0, b0, u1, u2, v1, v2)
            delta -= 1

        # Case E: b0 % 4 == 0
        elif (not constant_time) and (b0 % 4 == 0):
            (u1, u2, v1, v2) = xgcd_update(1, a0, b0, u1, u2, v1, v2)
            b0 //= 2
            delta += 1

        # Case F: b0 % 4 == 1
        elif (not constant_time) and (b0 % 4 == 1):
            b0 = (b0 - a0) // 2
            (u1, u2, v1, v2) = xgcd_update(2, a0, b0, u1, u2, v1, v2)
            delta -= 1

        # Case G: b0 % 4 == 2
        elif (not constant_time) and (b0 % 4 == 2):
            b0 //= 2
            (u1, u2, v1, v2) = xgcd_update(1, a0, b0, u1, u2, v1, v2)
            # maybe delta += 0

        # Case H: b0 % 4 == 3
        elif (not constant_time) and (b0 % 4 == 3):
            b0 = (b0 + a0) // 2
            (u1, u2, v1, v2) = xgcd_update(2, a0, b0, u1, u2, v1, v2)
            delta += 1

        # Case I: if a0 < b0 (sometimes the screenshot does a swap or modifies)
        elif (not constant_time) and (a0 < b0):
            # Possibly we do a swap of a0,b0 and also swap (u1,u2), (v1,v2).
            # This depends on your exact screenshot logic.
            a0, b0 = b0, a0
            u1, v1 = v1, u1
            u2, v2 = v2, u2
            # Adjust delta if needed
            delta = -delta

        # Termination condition
        if constant_time:
            # We do exactly 'constant_time_iterations' loops
            iteration += 1
            end_loop = (iteration >= constant_time_iterations)
        else:
            # Normal termination: if either a0=0 or b0=0, we are done
            end_loop = (a0 == 0 or b0 == 0)

    # -------------------------------------------------------------------------
    # Stage 3: Post-processing
    #
    # After the loop ends
    #   gcd = a0 + b0
    #   x_final = u1 + v1
    #   y_final = u2 + v2
    # Then it possibly halves them if exactly one is even, etc.
    # And it adjusts signs if needed.
    # -------------------------------------------------------------------------
    gcd_val = a0 + b0
    x_final = u1 + v1
    y_final = u2 + v2

    # Account for making x_final or y_final odd if required
    if x_final % 2 == 0:
        x_final //= 2
    elif y_final % 2 == 0:
        y_final //= 2

    # Fix results that may be negative due to approximations
    if gcd_val < 0:
        gcd_val = -gcd_val
    if x_final < 0:
        x_final = -x_final
    if y_final < 0:
        y_final = -y_final

    return gcd_val, x_final, y_final


# -----------------------------------------------------------------------------
# Example usage / quick test:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Just a small test
    a_test, b_test = 99, 69
    g, x0, y0 = xgcd(a_test, b_test, constant_time=False, bitwidth=8)
    print(f"a={a_test}, b={b_test} => gcd={g}, x0={x0}, y0={y0}")
    # You can verify that x0*a + y0*b == gcd in typical usage.
