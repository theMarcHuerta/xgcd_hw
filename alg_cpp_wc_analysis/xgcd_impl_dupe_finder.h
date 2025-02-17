#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>

using namespace std;

// Custom struct
struct matchInfo {
    uint32_t a, b, bit_size, iteration;
};

// Custom hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<uint32_t>()(p.first) * 31 + std::hash<uint32_t>()(p.second);
    }
};

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
uint32_t lut_result(uint32_t a_top, uint32_t b_top, int approx_bits, const string &rounding_mode) {
    if (b_top == 0)
        return 0; // safeguard (should not occur if b != 0)
    uint64_t numerator = static_cast<uint64_t>(a_top) << approx_bits;  // up-shift a_top
    if (rounding_mode == "round")
        return static_cast<uint32_t>((numerator + (b_top >> 1)) / b_top);
    else
        return static_cast<uint32_t>(numerator / b_top);
}

// Structure to hold the result of the xgcd_bitwise algorithm.
struct XgcdResult {
    uint32_t gcd;         // the final GCD
    int iterations;       // number of iterations
    double avgBitClears;  // average number of bit clears per iteration
};

// The main XGCD bitwise function. (All numbers are 32-bit.)
XgcdResult xgcd_bitwise(uint32_t a_in,
                        uint32_t b_in,
                        int total_bits,
                        int approx_bits,
                        const std::string &rounding_mode,
                        bool integer_rounding,
                        std::unordered_map<std::pair<uint32_t, uint32_t>,
                                           std::vector<matchInfo>, pair_hash>& data_map,
                        std::vector<std::vector<int>>& bitclears,
                        std::vector<std::vector<int>>& q_vals,
                        std::vector<std::vector<int>>& swaps,
                        std::vector<std::vector<int>>& negatives)
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

    // --- Main loop ---
    while (b != 0) {

        std::pair<uint32_t, uint32_t> curr_pair = {a, b};
        matchInfo iter_info = {a_in, b_in, 
                    static_cast<uint32_t>(total_bits), 
                    static_cast<uint32_t>(iteration_count)};
        data_map[curr_pair].push_back(iter_info); 

        iteration_count++;

        // STEP 2: Align b so that its MSB lines up with a's MSB.
        auto [b_aligned, shift_amount] = align_b(a, b);

        // STEP 3: Extract the top approx_bits from a and from b_aligned.
        uint32_t a_top = get_fixed_top_bits(a, approx_bits);
        uint32_t b_top = get_fixed_top_bits(b_aligned, approx_bits);

        // STEP 4: Compute the approximate quotient.
        uint32_t quotient = lut_result(a_top, b_top, approx_bits, rounding_mode);

        // STEP 5: Shift the quotient by shift_amount.
        // The Python code does:
        //   Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1)
        //   Q = Q_pre_round >> 1
        //   if (Q_pre_round & 1) and integer_rounding then Q++
        uint32_t Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1);
        uint32_t Q = Q_pre_round >> 1;
        if ((Q_pre_round & 1) && integer_rounding)
            Q++;

        // avg_q += Q; // (For debugging/statistics, if desired.)

        // STEP 6: Compute the adjusted b and the residual.
        // Use 64-bit arithmetic for multiplication.
        uint64_t product = static_cast<uint64_t>(b) * Q;
        uint32_t residual;
        if (a >= product){
            residual = a - static_cast<uint32_t>(product);
            negatives[iteration_count-1][1] += 1;
            negatives[total_bits - 1 - iteration_count][3] += 1;
        }
        else {
            residual = static_cast<uint32_t>(product - a);
            negatives[iteration_count-1][0] += 1;
            negatives[total_bits - 1 - iteration_count][2] += 1;
        }

        // Count how many “leading bits” got cleared in this iteration.
        int msb_a = bit_length(a);
        int msb_res = bit_length(residual);
        int clears_this_iter = msb_a - msb_res;
        if (clears_this_iter < 0)
            clears_this_iter = 0;
        bit_clears_list.push_back(clears_this_iter);

        bitclears[iteration_count-1].push_back(clears_this_iter);
        q_vals[iteration_count-1].push_back(int(Q));

        if (clears_this_iter > 4 || Q > 7){
            std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
            "   BIT CLEARS: " << clears_this_iter << std::endl;
        }

        // STEP 7: Prepare for the next iteration.
        // Swap: if the residual is larger than b, then a becomes residual;
        // otherwise, set a = b and b = residual.
        if (residual > b) {
            a = residual;
            swaps[iteration_count-1][0] += 1;
            swaps[total_bits - 1 - iteration_count][2] += 1;
        }
        else {
            uint32_t temp = b;
            b = residual;
            a = temp;
            swaps[iteration_count - 1][1] += 1;
            swaps[total_bits - 1 - iteration_count][3] += 1;
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