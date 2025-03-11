#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>
#include "xgcd_impl_dupe_finder.h"
#include "xgcd_bruteforce.cpp"
#include <unordered_map>
#include <map> // For ordered map
#include <limits>
#include <iomanip> // For formatting output

using namespace std;


/// Advanced statistics printing function.
/// collision_threshold: number of iterations (e.g. 4) within which a collision is counted.
void printAdvancedStats(
    const std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<matchInfo>, pair_hash>& data_map,
    int collision_threshold = 4)
{
    // Group the matchInfo vectors by:
    //   Outer key: bit size (matchInfo.bit_size) [only consider bit_size >= 6]
    //   Inner key: worst-case iteration count (matchInfo.iterations_to_completion)
    //   Value: a vector of the entire matchInfo vector (one per (a,b) pair)
    std::map<int, std::map<int, std::vector<std::vector<matchInfo>>, std::greater<int>> > groups;
    
    for (const auto& [key, matchInfos] : data_map) {
        if (matchInfos.empty())
            continue;
        // For this pair, we assume all matchInfo entries share the same total bit_size
        int bs = matchInfos.front().bit_size;
        if (bs < 6)
            continue; // Skip pairs with bit sizes lower than 6.
        int worstCase = matchInfos.front().iterations_to_completion;
        groups[bs][worstCase].push_back(matchInfos);
    }
    
    int total_pairs = 0;
    std::cout << "\n=== Advanced Statistics ===\n";
    // Iterate over groups by bit size (ordered by ascending bit size for clarity)
    for (const auto& [bitSize, worstGroups] : groups) {
        std::cout << bitSize << " bit worst-case statistics:\n";
        int scenarioIndex = 0;
        // worstGroups is a map with keys sorted in descending order (std::greater<int>),
        // so the first group is the absolute worst-case.
        for (const auto& [worstCase, vecOfMatchInfoVecs] : worstGroups) {
            int group_total = vecOfMatchInfoVecs.size();
            total_pairs += group_total;
            int collision_count = 0;
            std::unordered_map<int,int> matchBitSizeFrequency; // key: bit_size from matchInfo, value: count
            
            // For each (a,b) pair (its matchInfo vector) in this worst-case group:
            for (const auto& mvec : vecOfMatchInfoVecs) {
                bool hadCollision = false;
                // Look for any matchInfo entry with iteration less than collision_threshold.
                for (const auto& mi : mvec) {
                    if (mi.iteration < collision_threshold) {
                        hadCollision = true;
                        // Tally the bit_size of this collision record.
                        matchBitSizeFrequency[mi.bit_size]++;
                    }
                }
                if (hadCollision)
                    collision_count++;
            }
            
            double collision_percent = group_total > 0 ? (collision_count * 100.0 / group_total) : 0.0;
            
            // Determine the most common matching bit size among collisions.
            int most_common_bit = 0;
            int highest_freq = 0;
            for (const auto& [mb, freq] : matchBitSizeFrequency) {
                if (freq > highest_freq) {
                    highest_freq = freq;
                    most_common_bit = mb;
                }
            }
            
            // Prepare a label: if scenarioIndex is 0, it's the "worst case scenario",
            // otherwise, append " - scenarioIndex"
            std::stringstream label;
            label << bitSize << " bit worst case scenario";
            if (scenarioIndex > 0)
                label << " - " << scenarioIndex;
            
            std::cout << label.str() << " (worst-case iterations: " << worstCase << "): total pairs = " << group_total;
            std::cout << "   % that had collisions (within first " << collision_threshold << " iterations): " 
                      << std::fixed << std::setprecision(1) << collision_percent << "%, ";
            if (highest_freq > 0)
                std::cout << "most common match: " << most_common_bit << " bit worst case";
            else
                std::cout << "no collisions";
            std::cout << "\n";
            
            scenarioIndex++;
        }
        std::cout << "\n";
    }
    
    std::cout << "Total number of pairs (across all bit sizes): " << total_pairs << "\n";
}


void analyzeAndPrint(
    const std::vector<std::vector<int>>& bitclears,
    const std::vector<std::vector<int>>& q_vals,
    const std::vector<std::vector<int>>& swaps,
    const std::vector<std::vector<int>>& negatives) 
{
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== Bit Clears and Q Values ===\n";
    for (size_t i = 0; i < bitclears.size(); ++i) {
        if (bitclears[i].empty()) continue; // Skip empty vectors
        double avg_bitclear = 0;
        int max_bitclear = std::numeric_limits<int>::min();
        int min_bitclear = std::numeric_limits<int>::max();
        for (int val : bitclears[i]) {
            avg_bitclear += val;
            if (val > max_bitclear) max_bitclear = val;
            if (val < min_bitclear) min_bitclear = val;
        }
        avg_bitclear /= bitclears[i].size();

        double avg_q = 0;
        int max_q = std::numeric_limits<int>::min();
        int min_q = std::numeric_limits<int>::max();
        for (int val : q_vals[i]) {
            avg_q += val;
            if (val > max_q) max_q = val;
            if (val < min_q) min_q = val;
        }
        avg_q /= q_vals[i].size();

        std::cout << "Index " << i 
                  << " | Avg Bit Clears: " << avg_bitclear 
                  << ", Min Bit Clears: " << min_bitclear 
                  << ", Max Bit Clears: " << max_bitclear
                  << " | Avg Q: " << avg_q 
                  << ", Min Q: " << min_q 
                  << ", Max Q: " << max_q << "\n";
    }

    std::cout << "\n=== Swaps and Negatives Analysis ===\n";
    for (size_t i = 0; i < swaps.size(); ++i) {
        if (swaps[i].size() < 4 || negatives[i].size() < 4) continue; // Ensure size safety
        double swap_ratio_1 = (swaps[i][0] + swaps[i][1]) > 0
            ? static_cast<double>(swaps[i][0]) / (swaps[i][0] + swaps[i][1])
            : 0;
        double swap_ratio_2 = (swaps[i][2] + swaps[i][3]) > 0
            ? static_cast<double>(swaps[i][2]) / (swaps[i][2] + swaps[i][3])
            : 0;
        double negative_ratio_1 = (negatives[i][0] + negatives[i][1]) > 0
            ? static_cast<double>(negatives[i][0]) / (negatives[i][0] + negatives[i][1])
            : 0;
        double negative_ratio_2 = (negatives[i][2] + negatives[i][3]) > 0
            ? static_cast<double>(negatives[i][2]) / (negatives[i][2] + negatives[i][3])
            : 0;

        std::cout << "Index " << i
                  << " | Swaps: (Top Down " << swap_ratio_1 * 100 << "%, Bottom Up " << swap_ratio_2 * 100 << "%)"
                  << " | Negatives: (Top Down " << negative_ratio_1 * 100 << "%, Bottom Up " << negative_ratio_2 * 100 << "%)\n";
    }
}



void printPairsWithMultipleValues(const std::unordered_map<std::pair<uint32_t, uint32_t>, 
                                   std::vector<matchInfo>, pair_hash>& data_map) {
    // Group by bit-length of key.first.
    std::map<int, std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<matchInfo>>>> grouped_data;
    for (const auto& [key, values] : data_map) {
        int bit_size = bit_length(key.first);
        if (bit_size < 6)
            continue; // Skip pairs with bit size lower than 6.
        if (values.size() > 2) { // Only consider keys with more than 2 items.
            grouped_data[bit_size].push_back({key, values});
        }
    }
    // Print grouped data.
    for (const auto& [bit_size, pairs] : grouped_data) {
        std::cout << "=== Bit-Length: " << bit_size << " ===\n";
        for (const auto& [key, values] : pairs) {
            std::cout << "Pair (" << key.first << ", " << key.second << ") had " 
                      << values.size() << " items:\n";
            for (const auto& val : values) {
                std::cout << "   - matchInfo { a: " << val.a 
                          << ", b: " << val.b 
                          << ", bit_size: " << val.bit_size 
                          << ", iteration: " << val.iteration
                          << ", iterations_to_completion: " << val.iterations_to_completion
                          << " }\n";
            }
            std::cout << "------------------------\n";
        }
    }
}



int main(int argc, char* argv[]) {
    // Default parameters.
    int bits = 12;
    int approx_bits = 4;
    bool skip_symmetry = true;
    bool skip_zeros = true;
    bool force_a_msb = false;
    bool int_rounding = true;
    int additional_bits = 3;
    int collect_x_slowest_iters = 3; // Example: collect top 3 worst iteration counts.
    int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;  // fallback

    // Command-line argument parsing.
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc)
            bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--approx_bits") == 0 && i + 1 < argc)
            approx_bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip_symmetry") == 0)
            skip_symmetry = true;
        else if (strcmp(argv[i], "--skip_zeros") == 0)
            skip_zeros = true;
        else if (strcmp(argv[i], "--force_a_msb") == 0)
            force_a_msb = true;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--int_rounding") == 0)
            int_rounding = true;
        else if (strcmp(argv[i], "--additional_bits") == 0)
            additional_bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--collect_x_slowest_iters") == 0)
            collect_x_slowest_iters = atoi(argv[++i]);
    }

    // Data structures for xgcd_bitwise statistics.
    std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<matchInfo>, pair_hash> data_map;
    std::vector<std::vector<int>> bitclears(21);
    std::vector<std::vector<int>> q_vals(21);
    std::vector<std::vector<int>> swaps(21, std::vector<int>(4, 0));
    std::vector<std::vector<int>> negatives(21, std::vector<int>(4, 0));

    // Build a master map of worst-case pairs across different bit sizes.
    // Key: iterations_to_completion, Value: vector of (a, b) pairs.
    std::map<int, std::vector<std::pair<uint32_t, uint32_t>>, std::greater<int>> masterWorstCases;

    for (int i = bits; i < bits + additional_bits + 1; i++) {
        auto tmpWorstCases = bruteForce(i, approx_bits, skip_symmetry, skip_zeros, force_a_msb, int_rounding, num_threads, collect_x_slowest_iters);
        for (const auto& entry : tmpWorstCases) {
            int iterCount = entry.first;
            masterWorstCases[iterCount].insert(masterWorstCases[iterCount].end(), entry.second.begin(), entry.second.end());
        }
    }

    // Now, for every worst-case pair in the master map, run xgcd_bitwise.
    for (const auto &entry : masterWorstCases) {
        int iterations_to_complete = entry.first;
        for (const auto &p : entry.second) {
            uint32_t a_in = p.first;
            uint32_t b_in = p.second;
            // Pass the iteration key as iterations_to_completion.
            xgcd_bitwise(a_in, b_in, bit_length(a_in), 4, "truncate", true, iterations_to_complete, 
                         data_map, bitclears, q_vals, swaps, negatives);
        }
    }

    printPairsWithMultipleValues(data_map);
    analyzeAndPrint(bitclears, q_vals, swaps, negatives);
    // For example, using a collision threshold of 4 iterations:
    printAdvancedStats(data_map, 4);


    return 0;
}
