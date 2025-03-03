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
#include <map>    // for histogram
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
    int trunc_max_iter;
    vector<pair<int, int>> trunc_max_iter_pairs;
    double trunc_min_clears;
    pair<int, int> trunc_min_clears_pair;

    // Histogram: iteration count → frequency
    std::map<int, uint64_t> trunc_iters_hist;

    int valid_pairs;

    ThreadResult() : trunc_max_iter(-1), trunc_min_clears(numeric_limits<double>::max()),
                     valid_pairs(0) {}
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
                      uint64_t total_pairs, bool progress_thread, ThreadResult &result)
{
    // Determine range for 'a' based on force_a_msb.
    int a_min, a_max;
    if (force_a_msb) {
        a_min = 1 << (bits - 1);
        a_max = (1 << bits) - 1;
    } else {
        a_min = 0;
        a_max = (1 << bits) - 1;
    }
    // We skip a==0 because inner loop starts at b=1.
    if (a_min < 1) a_min = 1;

    int total_a = a_max - a_min + 1;
    // Partition the [a_min, a_max] range among threads.
    int chunk = total_a / num_threads;
    int remainder = total_a % num_threads;
    int start_a = a_min + thread_id * chunk + min(thread_id, remainder);
    int end_a = start_a + chunk - 1;
    if (thread_id < remainder)
        end_a++;

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "Thread " << thread_id << " launching. Processing a from " 
             << start_a << " to " << end_a << ".\n";
    }

    // Loop over a values in the assigned range.
    for (int a = start_a; a <= end_a; a++) {
        // For each a, let b run from 1 to a (ensuring a>=b and skipping symmetry).
        for (int b = 1; b <= a; b++) {
            result.valid_pairs++;

            // Run xgcd_bitwise in "truncate" mode.
            XgcdResult res_trunc = xgcd_bitwise(a, b, bits, approx_bits, "truncate", int_rounding);
            result.sum_iters_trunc += res_trunc.iterations;
            result.sum_trunc_clears += res_trunc.avgBitClears;

            // Update the histogram.
            result.trunc_iters_hist[res_trunc.iterations]++;

            // Verify against std::gcd.
            int expected_gcd = std::gcd(a, b);
            if (res_trunc.gcd != static_cast<uint32_t>(expected_gcd)) {
                lock_guard<mutex> lock(print_mutex);
                cerr << "ERROR (truncate): Mismatch for (a=" << a << ", b=" << b 
                     << ") → xgcd_bitwise() returned " << res_trunc.gcd 
                     << ", but std::gcd() says " << expected_gcd << "\n";
            }

            // Track worst-case (max iteration) for truncate.
            if (res_trunc.iterations > result.trunc_max_iter) {
                result.trunc_max_iter = res_trunc.iterations;
                result.trunc_max_iter_pairs.clear();
                result.trunc_max_iter_pairs.push_back({a, b});
            } else if (res_trunc.iterations == result.trunc_max_iter) {
                result.trunc_max_iter_pairs.push_back({a, b});
            }
            // Track minimum average bit clears for truncate.
            if (res_trunc.avgBitClears < result.trunc_min_clears) {
                result.trunc_min_clears = res_trunc.avgBitClears;
                result.trunc_min_clears_pair = {a, b};
            }

            // Update the global progress counter.
            ++global_counter;
        } // end for b
    } // end for a

    {
        lock_guard<mutex> lock(print_mutex);
        cout << "\nThread " << thread_id << " finished. Processed " 
             << result.valid_pairs << " pairs.\n";
    }
}

// ================================================================
// MAIN: Parse command-line arguments, spawn threads, and aggregate results.
// ================================================================
int main(int argc, char* argv[]) {
    // Default parameters (similar to your Python defaults).
    int bits = 12;
    int approx_bits = 4;
    bool skip_symmetry = true;
    bool skip_zeros = true;
    bool force_a_msb = false;
    bool int_rounding = true;
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // fallback



    // Simple command-line argument parsing.
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
        cout << "Brute forcing all pairs for " << bits << "-bit range.\n";
        cout << "Options: skip_symmetry=" << (skip_symmetry ? "true" : "false")
             << ", skip_zeros=" << (skip_zeros ? "true" : "false")
             << ", force_a_msb=" << (force_a_msb ? "true" : "false")
             << ", integer_rounding=" << (int_rounding ? "true" : "false") << "\n";
        cout << "Approx bits = " << approx_bits << ".\n";
        cout << "Using " << num_threads << " threads.\n";
    }

    // Determine overall range for 'a' to compute total number of pairs.
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
        total_pairs += a; // since for each a, b goes from 1 to a.
    }

    // Create a vector to hold thread objects and a vector for their results.
    vector<thread> threads;
    vector<ThreadResult> thread_results(num_threads);

    // Launch threads.
    for (int t = 0; t < num_threads; t++) {
        // Designate the last thread (t == num_threads - 1) to be the progress reporter.
        bool progress_thread = (t == num_threads - 1);
        threads.emplace_back(bruteForceThread, t, num_threads, bits, approx_bits,
                             force_a_msb, int_rounding,
                             total_pairs, progress_thread, ref(thread_results[t]));
    }

    // Wait for all threads to finish.
    for (auto &th : threads) {
        th.join();
    }
    cout << "\nAll threads completed.\n";

    // Aggregate results from all threads.
    uint64_t total_trunc_iters = 0;
    double total_trunc_clears = 0;
    int global_trunc_max_iter = -1;
    vector<pair<int, int>> global_trunc_max_iter_pairs;
    double global_trunc_min_clears = numeric_limits<double>::max();
    pair<int, int> global_trunc_min_clears_pair;
    int total_valid_pairs = 0;
    // Global histogram for iteration counts.
    map<int, uint64_t> global_hist;

    for (const auto &res : thread_results) {
        total_valid_pairs += res.valid_pairs;
        total_trunc_iters += res.sum_iters_trunc;
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
        // Combine each thread's histogram into a global histogram.
        for (const auto &entry : res.trunc_iters_hist) {
            global_hist[entry.first] += entry.second;
        }
    }

    double trunc_iter_mean = double(total_trunc_iters) / double(global_counter);
    double trunc_clears_mean = (total_trunc_clears / total_trunc_iters) * trunc_iter_mean;

    // Print final results.
    cout << "\n===== RESULTS (TRUNCATE MODE) =====\n";
    cout << "  Mean Iterations     : " << fixed << setprecision(3) << trunc_iter_mean << "\n";
    cout << "  Mean Bit Clears     : " << trunc_clears_mean << "\n";
    cout << "  Max Iterations      : " << global_trunc_max_iter << " for pairs: ";
    for (auto &p : global_trunc_max_iter_pairs)
        cout << "{" << p.first << ", " << p.second << "}, ";
    cout << "\n";
    cout << "  Min Avg Bit Clears  : " << global_trunc_min_clears << " for pair: (" 
         << global_trunc_min_clears_pair.first << "," << global_trunc_min_clears_pair.second << ")\n";

    cout << "\nTested a total of " << total_valid_pairs << " valid pairs.\n";

    // Print histogram for iterations.
    cout << "\n===== HISTOGRAM (TRUNCATE MODE ITERATIONS) =====\n";
    for (const auto &entry : global_hist) {
        cout << "Iterations " << entry.first << ": " << entry.second << "\n";
    }

    cout << "\n--- End of brute force test ---\n";

    return 0;
}
