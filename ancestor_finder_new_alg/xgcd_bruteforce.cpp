#pragma once

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
#include <map>   // Added for tracking top iteration counts
#include "xgcd_impl.h"

// For C++17 std::gcd
#include <numeric>

using namespace std;

// ================================================================
// STRUCTURES FOR THREAD RESULTS
// ================================================================

struct ThreadResult {
    // For truncate mode:
    uint64_t sum_iters_trunc = 0;
    double sum_trunc_clears = 0;
    // Instead of tracking only one worst-case, we now track a map:
    // Key: iteration count; Value: all (a, b) pairs that produced that count.
    // Sorted in descending order so that the highest (worst) iteration counts come first.
    std::map<int, vector<pair<uint32_t, uint32_t>>, std::greater<int>> trunc_top_iters;

    // For round mode (if needed, similar changes could be applied)
    uint64_t sum_iters_round = 0;
    double sum_round_clears = 0;

    int valid_pairs = 0;

    ThreadResult() {}
};

// ================================================================
// GLOBAL VARIABLES FOR PROGRESS REPORTING
// ================================================================
mutex print_mutex;
atomic<uint64_t> global_counter(0);

// ================================================================
// THREAD FUNCTION: Process a subset of 'a' values.
// ================================================================

void bruteForceThread(int thread_id, int num_threads, int bits, int approx_bits,
                      bool force_a_msb, bool int_rounding,
                      uint64_t total_pairs, bool progress_thread, int worstX,
                      ThreadResult &result)
{
    // Determine range for 'a' based on force_a_msb.
    uint32_t a_min, a_max;
    uint32_t b_min = 1;

    if (force_a_msb) {
        a_min = 1 << (bits - 1);
        a_max = (1 << bits) - 1;
    } else {
        a_min = 0;
        a_max = (1 << bits) - 1;
    }
    if (a_min < 1) a_min = 1;

    int total_a = a_max - a_min + 1;
    int chunk = total_a / num_threads;
    int remainder = total_a % num_threads;
    int start_a = a_min + thread_id * chunk + min(thread_id, remainder);
    int end_a = start_a + chunk - 1;
    if (thread_id < remainder)
        end_a++;

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Thread " << thread_id << " launching. Processing a from " << start_a << " to " << end_a << ".\n";
    }

    // Variable to hold the current minimum threshold among the worstX groups.
    // Initialize it to the lowest possible value so that early on, we add everything.
    int minThreshold = std::numeric_limits<int>::min();

    for (uint32_t a = start_a; a <= end_a; a++) {
        for (uint32_t b = b_min; b <= a; b++) {
            result.valid_pairs++;

            XgcdResult res_trunc = xgcd_bitwise(a, b, bits, approx_bits, "truncate", int_rounding);
            result.sum_iters_trunc += res_trunc.iterations;
            result.sum_trunc_clears += res_trunc.avgBitClears;

            int expected_gcd = std::gcd(a, b);
            if (res_trunc.gcd != static_cast<uint32_t>(expected_gcd)) {
                lock_guard<mutex> lock(print_mutex);
                cerr << "ERROR (truncate): Mismatch for (a=" << a << ", b=" << b 
                     << ") → xgcd_bitwise() returned " << res_trunc.gcd 
                     << ", but std::gcd() says " << expected_gcd << "\n";
            }

            // Only add if we haven't reached worstX groups yet, or if this iteration count is >= minThreshold.
            if (result.trunc_top_iters.size() < static_cast<size_t>(worstX) || 
                res_trunc.iterations >= minThreshold) 
            {
                result.trunc_top_iters[res_trunc.iterations].push_back({a, b});
                
                // If adding this pair causes us to exceed worstX distinct iteration counts,
                // remove the lowest one and update minThreshold.
                if (result.trunc_top_iters.size() > static_cast<size_t>(worstX)) {
                    auto it = result.trunc_top_iters.end();
                    --it;  // Points to the smallest iteration count.
                    result.trunc_top_iters.erase(it);
                }
                
                // Update minThreshold if we have worstX groups.
                if (result.trunc_top_iters.size() == static_cast<size_t>(worstX)) {
                    auto it = result.trunc_top_iters.end();
                    --it;
                    minThreshold = it->first;
                }
            }

            // uint64_t current = ++global_counter;
            // if (progress_thread && current % (total_pairs / 2000000 + 1) == 0) {
            //     double pct = 100.0 * current / total_pairs;
            //     lock_guard<mutex> lock(print_mutex);
            //     cout << "\rProgress: " << current << " / " << total_pairs << " (" 
            //          << fixed << setprecision(1) << pct << "%)   " << flush;
            // }
        }
    }

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "\nThread " << thread_id << " finished. Processed " << result.valid_pairs << " pairs.\n";
    }
}


// ================================================================
// MAIN: Parse command-line arguments, spawn threads, and aggregate results.
// ================================================================
// ================================================================
// MAIN: Spawn threads and aggregate results.
// ================================================================

std::map<int, std::vector<std::pair<uint32_t, uint32_t>>, std::greater<int>> bruteForce(int bits,
                                                        int approx_bits,
                                                        bool skip_symmetry,
                                                        bool skip_zeros,
                                                        bool force_a_msb,
                                                        bool int_rounding,
                                                        int num_threads,
                                                        int worstX)
{
    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Brute forcing all pairs for " << bits << "-bit range.\n";
        cout << "Options: skip_symmetry=" << (skip_symmetry ? "true" : "false")
             << ", skip_zeros=" << (skip_zeros ? "true" : "false")
             << ", force_a_msb=" << (force_a_msb ? "true" : "false")
             << ", integer_rounding=" << (int_rounding ? "true" : "false") << "\n";
        cout << "Approx bits = " << approx_bits << ".\n";
        cout << "Using " << num_threads << " threads.\n";
        cout << "Keeping top " << worstX << " worst-case iteration categories.\n";
    }

    // Determine overall range for 'a' to compute total pairs.
    int global_a_min, global_a_max;
    if (force_a_msb) {
        global_a_min = 1 << (bits - 1);
        global_a_max = (1 << bits) - 1;
    } else {
        global_a_min = 0;
        global_a_max = (1 << bits) - 1;
    }
    if (global_a_min < 1) global_a_min = 1;
    uint64_t total_pairs = 0;
    for (int a = global_a_min; a <= global_a_max; a++) {
        total_pairs += a; // since for each a, b runs from 1 to a.
    }

    // Create threads and results container.
    vector<thread> threads;
    vector<ThreadResult> thread_results(num_threads);

    // Launch threads.
    for (int t = 0; t < num_threads; t++) {
        bool progress_thread = (t == num_threads - 1);
        threads.emplace_back(bruteForceThread, t, num_threads, bits, approx_bits,
                             force_a_msb, int_rounding,
                             total_pairs, progress_thread, worstX, ref(thread_results[t]));
    }

    for (auto &th : threads)
        th.join();
    cout << "\nAll threads completed.\n";

    // Merge thread results into a global map.
    std::map<int, vector<pair<uint32_t, uint32_t>>, std::greater<int>> global_trunc_top_iters;
    for (const auto &res : thread_results) {
        for (const auto &entry : res.trunc_top_iters) {
            int iter_count = entry.first;
            for (const auto &p : entry.second) {
                global_trunc_top_iters[iter_count].push_back(p);
            }
        }
    }

    // Trim to worstX distinct iteration keys.
    while (global_trunc_top_iters.size() > static_cast<size_t>(worstX)) {
        auto it = global_trunc_top_iters.end();
        --it;
        global_trunc_top_iters.erase(it);
    }

    return global_trunc_top_iters;
}