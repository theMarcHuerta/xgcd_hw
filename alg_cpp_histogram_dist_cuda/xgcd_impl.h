#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>

using namespace std;

// Compute the bit length of a 32-bit unsigned integer.
// (For x==0, we return 0.)
int bit_length(uint32_t x) {
    if (x == 0) return 0;
    // __builtin_clz returns the number of leading 0s in a 32-bit value.
    return 32 - __builtin_clz(x);
}

// Align b_val with a_val: shift b_val left so that its MSB
// lines up with the MSB of a_val.
// Returns a pair: (b_shifted, shift_amount)
pair<uint32_t, int> align_b(uint32_t a_val, uint32_t b_val) {
    int len_a = bit_length(a_val);
    int len_b = bit_length(b_val);
    int shift_amount = len_a - len_b;
    uint32_t b_shifted = (shift_amount > 0) ? (b_val << shift_amount) : b_val;
    if (shift_amount < 0) shift_amount = 0;
    return make_pair(b_shifted, shift_amount);
}

// Extract the top approx_bits from x_val.  
// If x_val has fewer than approx_bits bits, we left-shift it.
uint32_t get_fixed_top_bits(uint32_t x_val, int approx_bits) {
    if (x_val == 0)
        return 0;
    int length = bit_length(x_val);
    if (length <= approx_bits)
        return x_val << (approx_bits - length);
    else {
        int shift_down = length - approx_bits;
        return x_val >> shift_down;
    }
}

// Compute the ratio a_top / b_top in fixed-point arithmetic
// with approx_bits fractional bits.
// If rounding_mode=="round" then we add half of b_top before dividing.
uint32_t lut_result(uint32_t a_top, uint32_t b_top, int approx_bits) {
    if (b_top == 0)
        return 0; // safeguard (should not occur if b != 0)
    uint64_t numerator = static_cast<uint64_t>(a_top) << approx_bits;  // up-shift a_top
    return static_cast<uint32_t>(numerator / b_top);
}

// Structure to hold the result of the xgcd_bitwise algorithm.
struct XgcdResult {
    uint32_t gcd;         // the final GCD
    int iterations;       // number of iterations
    double avgBitClears;  // average number of bit clears per iteration
};

// The main XGCD bitwise function. (All numbers are 32-bit.)
XgcdResult xgcd_bitwise(uint32_t a_in, uint32_t b_in,
                        int total_bits = 32,
                        int approx_bits = 4)
{
    // --- STEP 1: Normalize the inputs ---
    // Mask off any bits above total_bits.
    uint32_t mask = (total_bits >= 32) ? 0xFFFFFFFF : ((1u << total_bits) - 1);
    uint32_t a = a_in & mask;
    uint32_t b = b_in & mask;

    // Ensure a >= b (swap if needed).
    if (b > a)
        swap(a, b);

    // Quick checks for trivial cases.
    if (b == 0)
        return { a, 0, 0.0 };
    if (a == 0)
        return { b, 0, 0.0 };

    int iteration_count = 0;
    vector<int> bit_clears_list;
    // uint64_t avg_q = 0; // (Optional: used for tracking the average Q)

    bool was_one_clear = false;
    bool two_one_clear = false;

    int count_twice = 0;

    uint32_t prev_a = 0;
    uint32_t prev_b = 0;

    // --- Main loop ---
    while (b != 0) {
        iteration_count++;

        // STEP 2: Align b so that its MSB lines up with a's MSB.
        auto [b_aligned, shift_amount] = align_b(a, b);

        // STEP 3: Extract the top approx_bits from a and from b_aligned.
        uint32_t a_top = get_fixed_top_bits(a, approx_bits);
        uint32_t b_top = get_fixed_top_bits(b_aligned, approx_bits);

        // STEP 4: Compute the approximate quotient.
        uint32_t quotient = lut_result(a_top, b_top, approx_bits);

        // STEP 5: Shift the quotient by shift_amount.
        uint32_t Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1);
        uint32_t Q = Q_pre_round >> 1;
        Q++;


        // STEP 6: Compute the adjusted b and the residual.
        // Use 64-bit arithmetic for multiplication.
        uint64_t product = static_cast<uint64_t>(b) * Q;
        uint64_t product_two = static_cast<uint64_t>(b) * (Q_pre_round >> 1);

        uint32_t residual;
        uint32_t residual_two;

        if (a >= product)
            residual = a - static_cast<uint32_t>(product);
        else
            residual = static_cast<uint32_t>(product - a);

        if (a >= product_two)
            residual_two = a - static_cast<uint32_t>(product_two);
        else
            residual_two = static_cast<uint32_t>(product_two - a);

        if (residual_two < residual){
            residual = residual_two;
            Q = Q_pre_round >> 1;
        }

        // Count how many “leading bits” got cleared in this iteration.
        int msb_a = bit_length(a);
        int msb_res = bit_length(residual);
        int clears_this_iter = msb_a - msb_res;
        if (clears_this_iter < 0)
            clears_this_iter = 0;
        bit_clears_list.push_back(clears_this_iter);

        // STEP 7: Prepare for the next iteration.
        // Swap: if the residual is larger than b, then a becomes residual;
        // otherwise, set a = b and b = residual.
        if (residual > b)
            a = residual;
        else {
            prev_a = a;
            prev_b = b;
            uint32_t temp = b;
            b = residual;
            a = temp;
        }
    }

    // Adjust the last iteration’s bit clears (as in the Python code).
    if (!bit_clears_list.empty())
        bit_clears_list.back() += bit_length(a);

    // Compute the average bit clears per iteration.
    double avg_bit_clears = 0.0;
    for (int clears : bit_clears_list)
        avg_bit_clears += clears;
    if (iteration_count > 0)
        avg_bit_clears /= iteration_count;

    return { a, iteration_count, avg_bit_clears };
}