#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>
#include "xgcd_impl.h"

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
    bool prev_clr = false;
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
        uint64_t product_two = static_cast<uint64_t>(b) * (Q_pre_round >> 1);

        uint32_t residual;
        uint32_t residual_two;

        bool product_one_neg = false; 
        bool product_two_neg = false; 

        if (a >= product){
            residual = a - static_cast<uint32_t>(product);
        }
        else {
            product_one_neg = true;
            residual = static_cast<uint32_t>(product - a);
        }

        if (a >= product_two){
            residual_two = a - static_cast<uint32_t>(product_two);
        }
        else{
            product_two_neg = true;
            residual_two = static_cast<uint32_t>(product_two - a);
        }

        if (residual_two < residual){
            residual = residual_two;
            Q = Q_pre_round >> 1;
            if (product_two_neg){
                negatives[iteration_count-1][0] += 1;
                negatives[total_bits - 1 - iteration_count][2] += 1;
            }
            else {
                negatives[iteration_count-1][1] += 1;
                negatives[total_bits - 1 - iteration_count][3] += 1;
            }
        }
        else {
            if (product_one_neg){
                negatives[iteration_count-1][0] += 1;
                negatives[total_bits - 1 - iteration_count][2] += 1;
            }
            else {
                negatives[iteration_count-1][1] += 1;
                negatives[total_bits - 1 - iteration_count][3] += 1;
            }
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


        // if (clears_this_iter == 1 && iteration_count != 1){
        //     prev_clr = true;

        //     std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
        //     "   BIT CLEARS: " << clears_this_iter << "    ITERATION: " << iteration_count << std::endl;
        //     std::cout << "curr a: " << a << "\n   curr b: " << b << std::endl;
        // }
        // else if ( prev_clr == true ) {
        //     std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
        //     "   BIT CLEARS: " << clears_this_iter << "    ITERATION: " << iteration_count << std::endl;
        //     std::cout << "curr a: " << a << "\n   curr b: " << b << std::endl << std::endl;

        //     prev_clr = false;
        // }
        // if (clears_this_iter > 4 || Q > 7){
        //     std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
        //     "   BIT CLEARS: " << clears_this_iter << std::endl;
        // }

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