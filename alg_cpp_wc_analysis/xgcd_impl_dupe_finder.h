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


        if (clears_this_iter == 1 && iteration_count != 1){
            prev_clr = true;

            std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
            "   BIT CLEARS: " << clears_this_iter << "    ITERATION: " << iteration_count << std::endl;
            std::cout << "curr a: " << a << "\n   curr b: " << b << std::endl;
        }
        else if ( prev_clr == true ) {
            std::cout << "A_IN: " << a_in << "   B_IN: " << b_in << "   Q: " << Q <<
            "   BIT CLEARS: " << clears_this_iter << "    ITERATION: " << iteration_count << std::endl;
            std::cout << "curr a: " << a << "\n   curr b: " << b << std::endl << std::endl;
            
            prev_clr = false;
        }
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


===== RESULTS (TRUNCATE) =====
  Mean Iterations     : 5.595
  Median Iterations   : 6.000
  Mean Bit Clears     : 4.144
  Median Bit Clears   : 4.000
  Max Iterations      : 11 for pairs [(3647, 2564), (3695, 2568), (3697, 2575), (3699, 2582), (3707, 2581), (3758, 2611), (3776, 2629), (3840, 2809), (3841, 2784), 
  (3859, 2797), (3865, 2801), (3927, 2726), (4009, 2568), (4028, 2575), (4036, 2581),
   (4045, 2564), (4047, 2582), (4075, 2611)]
  Min Avg Bit Clears  : 2.000 for pair a=1, b=1

18

===== RESULTS (ROUND) =====
  Mean Iterations     : 5.543
  Median Iterations   : 6.000
  Mean Bit Clears     : 4.208
  Median Bit Clears   : 4.000
  Max Iterations      : 11 for pairs [(1675, 1168), (1829, 1168), (1855, 1303), (1893, 1318), (2054, 1303), 
  (2061, 1318), (2182, 1675), (2407, 1855), (2468, 1893), (2603, 1815), (2689, 2182), (2842, 1815), (2843, 1675), 
  (2869, 1258), (2913, 2125), (2955, 2078), (2959, 2407), (2997, 1168), (3043, 2468), (3057, 2359), (3068, 2163), 
  (3073, 2248), (3077, 2251), (3089, 2272), (3091, 2264), (3106, 2291), (3118, 2281), (3126, 2267), (3140, 2297), 
  (3158, 1303), (3158, 1855), (3173, 2572), (3196, 2689), (3211, 1318), (3211, 1893), (3267, 2362), (3279, 2078), 
  (3313, 2312), (3339, 2320), (3350, 2336), (3353, 2336), (3353, 2371), (3356, 2337), (3357, 1303), (3361, 2342), 
  (3379, 1318), (3379, 2357), (3389, 2396), (3391, 2603), (3393, 2386), (3407, 2396), (3409, 2396), (3421, 2163), 
  (3421, 2403), (3437, 2414), (3462, 2125), (3469, 2823), (3511, 2959), (3513, 2531), (3521, 2539), (3549, 2512), 
  (3559, 2293), (3566, 2513), (3582, 2809), (3594, 2227), (3618, 3043), (3621, 2320), (3623, 2312), (3655, 2336), 
  (3655, 2337), (3658, 2336), (3665, 2342), (3674, 2819), (3683, 2826), (3692, 2357), (3692, 2995), (3694, 2601), 
  (3695, 2568), (3697, 2574), (3697, 2575), (3701, 2913), (3703, 3196), (3707, 2581), (3708, 2611), (3710, 2606), 
  (3710, 2673), (3711, 2606), (3727, 2621), (3728, 2591), (3736, 2579), (3746, 2887), (3755, 3057), (3760, 2371), 
  (3765, 2386), (3766, 2645), (3773, 2628), (3779, 2396), (3779, 2894), (3781, 2396), (3786, 2636), (3788, 2403), 
  (3789, 2663), (3793, 2664), (3794, 2665), (3797, 2669), (3799, 2396), (3805, 2414), (3805, 2752), (3819, 2362), 
  (3832, 2955), (3834, 2695), (3835, 2696), (3841, 2784), (3843, 2803), (3856, 2795), (3856, 2815), (3857, 1675), 
  (3857, 2182), (3859, 2796), (3859, 2797), (3865, 2801), (3869, 2842), (3871, 1636), (3871, 2806), (3881, 2813), 
  (3947, 3087), (3973, 2513), (3973, 3068), (3987, 2512), (3992, 2779), (3999, 3082), (4001, 2579), (4009, 2568), 
  (4011, 1168), (4011, 2843), (4016, 2823), (4017, 2824), (4017, 3097), (4020, 2359), (4024, 3089), (4025, 2574), 
  (4026, 3103), (4027, 3086), (4028, 2575), (4031, 2839), (4035, 3092), (4036, 2581), (4037, 2771), (4043, 2843), 
  (4044, 3103), (4045, 2591), (4047, 2822), (4049, 2851), (4051, 2820), (4052, 3109), (4053, 3125), (4055, 2852), 
  (4055, 3113), (4057, 3113), (4063, 3511), (4065, 3118), (4069, 2833), (4071, 2834), (4073, 2840), (4078, 3129), 
  (4080, 2531), (4087, 2874), (4087, 3136), (4091, 2960), (4091, 3140), (4093, 2850), (4095, 3142)]