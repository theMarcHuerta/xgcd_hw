#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include "xgcd_impl.h"

// For C++17 std::gcd
#include <numeric>

using namespace std;

// ================================================================
// STRUCTURES FOR THREAD RESULTS
// ================================================================
struct ThreadResult {
    uint64_t sum_iters_trunc = 0;
    double sum_trunc_clears = 0;
    int trunc_max_iter;
    vector<pair<int, int>> trunc_max_iter_pairs;
    double trunc_min_clears;
    pair<int, int> trunc_min_clears_pair;

    uint64_t sum_iters_round = 0;
    double sum_round_clears = 0;
    int round_max_iter;
    vector<pair<int, int>> round_max_iter_pairs;
    double round_min_clears;
    pair<int, int> round_min_clears_pair;

    int valid_pairs;

    ThreadResult()
        : trunc_max_iter(-1),
          trunc_min_clears(numeric_limits<double>::max()),
          round_max_iter(-1),
          round_min_clears(numeric_limits<double>::max()),
          valid_pairs(0)
    {}
};

// ================================================================
// GLOBAL VARIABLES FOR PROGRESS REPORTING
// ================================================================
mutex print_mutex;
atomic<uint64_t> global_counter(0);

// ================================================================
// Helper function: count total pairs in [a_min, a_max] x [b=a_min..a].
// ================================================================
static inline uint64_t countPairs(int a_min, int a_max)
{
    // Summation of (a - a_min + 1) as a goes from a_min to a_max
    // i.e. sum_{i=0..(a_max - a_min)} of (i+1) = (n*(n+1))/2, where n = (a_max - a_min + 1).
    uint64_t n = static_cast<uint64_t>(a_max - a_min + 1);
    return (n * (n + 1)) / 2;
}

// ================================================================
// Helper function: given a global index k, find which 'a' block it belongs to.
// We enumerate in ascending a, then b from a_min..a.
// For block i (i.e., a = a_min + i), there are (i+1) pairs.
// So we want the largest i such that i*(i+1)/2 <= k.
// ================================================================
uint64_t findAIndex(uint64_t k)
{
    // Approximate i from k ~ i*(i+1)/2 => i^2 ~ 2k => i ~ sqrt(2k)
    // We'll do a quick approach with sqrt, then fix with a small loop.
    double guess = std::floor((std::sqrt(8.0 * k + 1) - 1) / 2.0);
    uint64_t i = static_cast<uint64_t>( (guess < 0.0) ? 0.0 : guess );

    // Adjust downward if we're too high:
    while ((i * (i + 1)) / 2 > k) {
        i--;
    }
    // Adjust upward in case we're too low:
    while (((i + 1) * (i + 2)) / 2 <= k) {
        i++;
    }
    return i;
}

// ================================================================
// Convert global index k -> (a,b). We assume b runs from a_min..a
// in ascending order, for a = a_min..a_max in ascending order.
// ================================================================
pair<int,int> indexToPair(uint64_t k, int a_min)
{
    // i = block index, i*(i+1)/2 <= k < (i+1)*(i+2)/2
    uint64_t i = findAIndex(k);
    // The start index of block i is blockStart = i*(i+1)/2
    uint64_t blockStart = (i * (i + 1)) / 2;

    // offset within that block
    uint64_t offset = k - blockStart;

    // a = a_min + i
    // b = a_min + offset
    int a = a_min + static_cast<int>(i);
    int b = a_min + static_cast<int>(offset);
    return make_pair(a,b);
}

// ================================================================
// THREAD FUNCTION: Process a subrange of the global index space.
// ================================================================
void bruteForceThread(int thread_id,
                      uint64_t startIndex, uint64_t endIndex,
                      int bits, int approx_bits,
                      bool int_rounding,
                      int a_min,
                      uint64_t total_pairs,
                      ThreadResult &result)
{
    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Thread " << thread_id << " starting. "
             << "Global index range: [" << startIndex << ", " << endIndex << "]\n";
    }

    for (uint64_t k = startIndex; k <= endIndex; k++) {
        // Convert k to (a,b)
        auto [a, b] = indexToPair(k, a_min);

        // (Optional) If you skip zeros, and a_min could be 0, you could do:
        //   if (a == 0 || b == 0) continue;

        result.valid_pairs++;

        // Truncate mode
        XgcdResult res_trunc = xgcd_bitwise(a, b, bits, approx_bits, "truncate", int_rounding);
        result.sum_iters_trunc += res_trunc.iterations;
        result.sum_trunc_clears += res_trunc.avgBitClears;
        int expected_gcd = std::gcd(a, b);
        if (res_trunc.gcd != static_cast<uint32_t>(expected_gcd)) {
            lock_guard<mutex> lock(print_mutex);
            cerr << "ERROR (truncate): mismatch for (a=" << a << ", b=" << b
                 << ") => xgcd_bitwise returned " << res_trunc.gcd
                 << ", but std::gcd is " << expected_gcd << "\n";
        }

        if (res_trunc.iterations == bits - 2) {
            result.trunc_max_iter_pairs.push_back({a, b});
        }

        // Track max/min for truncate
        // if (res_trunc.iterations > result.trunc_max_iter) {
        //     result.trunc_max_iter = res_trunc.iterations;
        //     result.trunc_max_iter_pairs.clear();
        //     result.trunc_max_iter_pairs.push_back({a, b});
        // } else if (res_trunc.iterations == result.trunc_max_iter) {
        //     result.trunc_max_iter_pairs.push_back({a, b});
        // }
        if (res_trunc.avgBitClears < result.trunc_min_clears) {
            result.trunc_min_clears = res_trunc.avgBitClears;
            result.trunc_min_clears_pair = {a, b};
        }

        // // Round mode
        // XgcdResult res_round = xgcd_bitwise(a, b, bits, approx_bits, "round", int_rounding);
        // result.sum_iters_round += res_round.iterations;
        // result.sum_round_clears += res_round.avgBitClears;
        // if (res_round.gcd != static_cast<uint32_t>(expected_gcd)) {
        //     lock_guard<mutex> lock(print_mutex);
        //     cerr << "ERROR (round): mismatch for (a=" << a << ", b=" << b
        //          << ") => xgcd_bitwise returned " << res_round.gcd
        //          << ", but std::gcd is " << expected_gcd << "\n";
        // }
        // // Track max/min for round
        // if (res_round.iterations > result.round_max_iter) {
        //     result.round_max_iter = res_round.iterations;
        //     result.round_max_iter_pairs.clear();
        //     result.round_max_iter_pairs.push_back({a, b});
        // } else if (res_round.iterations == result.round_max_iter) {
        //     result.round_max_iter_pairs.push_back({a, b});
        // }
        // if (res_round.avgBitClears < result.round_min_clears) {
        //     result.round_min_clears = res_round.avgBitClears;
        //     result.round_min_clears_pair = {a, b};
        // }

        // Update global progress
        uint64_t current = ++global_counter;

        if (thread_id == 0 && current % (total_pairs / (total_pairs / 1000) + 1) == 0) {
            double pct = 100.0 * current / total_pairs;
            lock_guard<mutex> lock(print_mutex);
            cout << "\rProgress: " << current << " / " << total_pairs
                << " (" << fixed << setprecision(1) << pct << "%)   " << flush;
        }

    }

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Thread " << thread_id << " finished. Processed "
             << (endIndex - startIndex + 1) << " pairs.\n";
    }
}

// ================================================================
// MAIN
// ================================================================
int main(int argc, char* argv[])
{
    // Default parameters
    int bits = 12;
    int approx_bits = 4;
    bool skip_symmetry = true;   // (In this code, we always do b in [a_min..a], so effectively skipping symmetry.)
    bool skip_zeros = true;      // If you'd like to skip a=0 or b=0, you'd handle that logic in the indexToPair or inside the thread loop
    bool force_a_msb = false;
    bool int_rounding = true;
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--approx_bits") == 0 && i + 1 < argc) {
            approx_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--skip_symmetry") == 0) {
            skip_symmetry = true;
        } else if (strcmp(argv[i], "--skip_zeros") == 0) {
            skip_zeros = true;
        } else if (strcmp(argv[i], "--force_a_msb") == 0) {
            force_a_msb = true;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--int_rounding") == 0) {
            int_rounding = true;
        }
    }

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Brute forcing all pairs for " << bits << "-bit range.\n"
             << "Options: skip_symmetry=" << (skip_symmetry ? "true" : "false")
             << ", skip_zeros=" << (skip_zeros ? "true" : "false")
             << ", force_a_msb=" << (force_a_msb ? "true" : "false")
             << ", integer_rounding=" << (int_rounding ? "true" : "false") << "\n"
             << "Approx bits = " << approx_bits << ".\n"
             << "Using " << num_threads << " threads.\n";
    }

    // Determine a_min, a_max
    int global_a_min, global_a_max;
    if (force_a_msb) {
        // e.g. for 12 bits, a_min = 1<<(12-1) = 2048, a_max = 4095
        global_a_min = 1 << (bits - 1);
        global_a_max = (1 << bits) - 1;
    } else {
        global_a_min = 0;
        global_a_max = (1 << bits) - 1;
    }

    // If skipping zeros, enforce a_min >= 1
    if (skip_zeros && global_a_min < 1) {
        global_a_min = 1;
    }

    // Count total pairs
    uint64_t total_pairs = countPairs(global_a_min, global_a_max);

    // Prepare thread vectors
    vector<thread> threads;
    vector<ThreadResult> thread_results(num_threads);

    // We split total_pairs among num_threads
    uint64_t chunk = total_pairs / num_threads;
    uint64_t remainder = total_pairs % num_threads;

    // Launch threads
    uint64_t nextStart = 0;
    for (int t = 0; t < num_threads; t++) {
        // Each thread gets chunk or chunk+1 indices
        uint64_t extra = (t < (int) remainder) ? 1 : 0;
        uint64_t startIndex = nextStart;
        uint64_t endIndex   = startIndex + chunk + extra - 1;
        if (endIndex >= total_pairs) endIndex = total_pairs - 1;

        if (startIndex > endIndex) {
            // If total_pairs < num_threads, some threads do no work
            {
                lock_guard<mutex> lock(print_mutex);
                cout << "Thread " << t << " gets empty range.\n";
            }
            break;
        }

        nextStart = endIndex + 1;

        threads.emplace_back(bruteForceThread,
                             t,
                             startIndex, endIndex,
                             bits, approx_bits,
                             int_rounding,
                             global_a_min,
                             total_pairs,
                             ref(thread_results[t]));
    }

    // Wait
    for (auto &th : threads) {
        if (th.joinable()) th.join();
    }

    cout << "\nAll threads completed.\n";

    // Aggregate results
    uint64_t total_trunc_iters = 0;
    double total_trunc_clears = 0;
    int global_trunc_max_iter = -1;
    vector<pair<int, int>> global_trunc_max_iter_pairs;
    double global_trunc_min_clears = numeric_limits<double>::max();
    pair<int, int> global_trunc_min_clears_pair;

    // uint64_t total_round_iters = 0;
    // double total_round_clears = 0;
    // int global_round_max_iter = -1;
    // vector<pair<int, int>> global_round_max_iter_pairs;
    // double global_round_min_clears = numeric_limits<double>::max();
    // pair<int, int> global_round_min_clears_pair;

    int total_valid_pairs = 0;

    for (auto &res : thread_results) {
        total_valid_pairs += res.valid_pairs;

        // Truncate
        total_trunc_iters  += res.sum_iters_trunc;
        total_trunc_clears += res.sum_trunc_clears;
        if (res.trunc_max_iter > global_trunc_max_iter) {
            global_trunc_max_iter = res.trunc_max_iter;
            global_trunc_max_iter_pairs = res.trunc_max_iter_pairs;
        } else if (res.trunc_max_iter == global_trunc_max_iter) {
            global_trunc_max_iter_pairs.insert(global_trunc_max_iter_pairs.end(),
                                               res.trunc_max_iter_pairs.begin(),
                                               res.trunc_max_iter_pairs.end());
        }
        if (res.trunc_min_clears < global_trunc_min_clears) {
            global_trunc_min_clears = res.trunc_min_clears;
            global_trunc_min_clears_pair = res.trunc_min_clears_pair;
        }

        // // Round
        // total_round_iters  += res.sum_iters_round;
        // total_round_clears += res.sum_round_clears;
        // if (res.round_max_iter > global_round_max_iter) {
        //     global_round_max_iter = res.round_max_iter;
        //     global_round_max_iter_pairs = res.round_max_iter_pairs;
        // } else if (res.round_max_iter == global_round_max_iter) {
        //     global_round_max_iter_pairs.insert(global_round_max_iter_pairs.end(),
        //                                        res.round_max_iter_pairs.begin(),
        //                                        res.round_max_iter_pairs.end());
        // }
        // if (res.round_min_clears < global_round_min_clears) {
        //     global_round_min_clears = res.round_min_clears;
        //     global_round_min_clears_pair = res.round_min_clears_pair;
        // }
    }

    // Compute means. global_counter is the total processed pairs
    // which should match total_pairs if everything is enumerated fully.
    uint64_t pairs_processed = global_counter.load();
    double trunc_iter_mean  = (pairs_processed == 0) ? 0.0
                                   : double(total_trunc_iters) / double(pairs_processed);
    // double round_iter_mean  = (pairs_processed == 0) ? 0.0
    //                                : double(total_round_iters) / double(pairs_processed);

    // For average bit-clears, interpret `sum_trunc_clears` as
    // sum_of_avgBitClears. Often you'd want "mean of avgBitClears" or
    // a weighted approach. Here we do a simpler approach similar to your code:
    double trunc_clears_mean = (pairs_processed == 0) ? 0.0
                                    : (total_trunc_clears / double(total_trunc_iters))
                                      * trunc_iter_mean;
    // double round_clears_mean = (pairs_processed == 0) ? 0.0
    //                                 : (total_round_clears / double(total_round_iters))
    //                                   * round_iter_mean;

    // Print final results
    cout << "\n===== RESULTS (TRUNCATE MODE) =====\n"
         << "  Mean Iterations     : " << fixed << setprecision(3) << trunc_iter_mean << "\n"
         << "  Mean Bit Clears     : " << trunc_clears_mean << "\n"
         << "  Max Iterations      : " << global_trunc_max_iter << " for pairs: ";
    for (auto &p : global_trunc_max_iter_pairs) {
        cout << "{" << p.first << ", " << p.second << "}, ";
    }
    cout << "\n"
         << "  Min Avg Bit Clears  : " << global_trunc_min_clears << " for pair: ("
         << global_trunc_min_clears_pair.first << "," << global_trunc_min_clears_pair.second << ")\n";

    // cout << "\n===== RESULTS (ROUND MODE) =====\n"
    //      << "  Mean Iterations     : " << round_iter_mean << "\n"
    //      << "  Mean Bit Clears     : " << round_clears_mean << "\n"
    //      << "  Max Iterations      : " << global_round_max_iter << " for pairs: ";
    // for (auto &p : global_round_max_iter_pairs) {
    //     cout << "(" << p.first << "," << p.second << ") ";
    // }
    // cout << "\n"
    //      << "  Min Avg Bit Clears  : " << global_round_min_clears << " for pair: ("
    //      << global_round_min_clears_pair.first << "," << global_round_min_clears_pair.second << ")\n";

    cout << "\nTested a total of " << total_valid_pairs << " valid pairs.\n"
         << "global_counter = " << pairs_processed << ", total_pairs = " << total_pairs << "\n"
         << "\n--- End of brute force test ---\n";

    return 0;
}
