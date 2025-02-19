#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>
#include "xgcd_impl_dupe_finder.h"
#include <unordered_map>

using namespace std;

#include <map> // For ordered map
#include <limits>
#include <iomanip> // For formatting output

void analyzeAndPrint(
    const std::vector<std::vector<int>>& bitclears,
    const std::vector<std::vector<int>>& q_vals,
    const std::vector<std::vector<int>>& swaps,
    const std::vector<std::vector<int>>& negatives) 
{
    std::cout << std::fixed << std::setprecision(2); // Format floating-point output

    std::cout << "=== Bit Clears and Q Values ===\n";
    for (size_t i = 0; i < bitclears.size(); ++i) {
        if (bitclears[i].empty()) continue; // Skip empty vectors to avoid division by zero

        double avg_bitclear = 0;
        int max_bitclear = std::numeric_limits<int>::max();
        for (int val : bitclears[i]) {
            avg_bitclear += val;
            if (val < max_bitclear) max_bitclear = val;
        }
        avg_bitclear /= bitclears[i].size();

        double avg_q = 0;
        int max_q = std::numeric_limits<int>::min();
        for (int val : q_vals[i]) {
            avg_q += val;
            if (val > max_q) max_q = val;
        }
        avg_q /= q_vals[i].size();

        std::cout << "Index " << i 
                  << " | Avg Bit Clears: " << avg_bitclear 
                  << ", Max Bit Clears: " << max_bitclear
                  << " | Avg Q: " << avg_q 
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
    // Use std::map to store keys grouped by bit-length
    std::map<int, std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<matchInfo>>>> grouped_data;

    // Group by bit-length of key.first
    for (const auto& [key, values] : data_map) {
        if (values.size() > 2) { // Only consider keys with more than 2 items
            int bit_size = bit_length(key.first); // Compute bit length of key.first
            grouped_data[bit_size].push_back({key, values});
        }
    }

    // Print by increasing bit-size order
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
                          << " }\n";
            }
            std::cout << "------------------------\n"; // Separator for readability
        }
    }
}


int main() {

    // Define unordered_map with a custom hash function
    std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<matchInfo>, pair_hash> data_map;

    // Worst case scenarios
    std::vector<std::pair<uint32_t, uint32_t>> worstCases = {
    // 8 bit
    /////////
    {185, 129}, {193, 142}, {202, 129}, {207, 151}, 
    {211, 147}, {229, 95}, {229, 161}, {230, 147}, 
    {233, 162}, {234, 163}, {235, 164}, {236, 163}, 
    {237, 173}, {239, 168}, {239, 169}, {241, 185}, 
    {246, 151}, {251, 181}, {251, 207}, {253, 162}, 
    {253, 163}, {253, 179}, {254, 161}, {254, 179}, 
    {255, 163},
    // 9 bit
    /////////
    {324, 229}, {363, 229}, {387, 283}, {389, 284}, 
    {391, 282}, {393, 287}, {398, 287}, {419, 324}, 
    {421, 348}, {438, 317}, {443, 314}, {455, 282}, 
    {460, 331}, {461, 321}, {462, 283}, {463, 284}, 
    {464, 323}, {466, 325}, {468, 287}, {468, 329}, 
    {477, 335}, {499, 314}, {502, 321}, {505, 323}, 
    {505, 358}, {509, 325}, {509, 358},
    // 10 bit
    /////////
    {772, 565}, {780, 569}, {782, 553}, {821, 592}, 
    {877, 553}, {898, 623}, {927, 569}, {934, 649}, 
    {935, 651}, {937, 651}, {951, 664}, {953, 670}, 
    {955, 592}, {955, 673}, {957, 673}, {961, 703}, 
    {963, 698}, {970, 703}, {1011, 782}, {1013, 649}, 
    {1016, 651}, {1018, 651}, {1019, 737},
    // 11 bit
    /////////
    {1579, 1142}, {1843, 1283}, {1849, 1287}, {1849, 1525}, 
    {1853, 1290}, {1888, 1335}, {1889, 1328}, {1893, 1318}, 
    {1902, 1337}, {1918, 1349}, {1930, 1397}, {1983, 1430}, 
    {2005, 1413}, {2006, 1283}, {2012, 1287}, {2017, 1290},
    // 12 bit
    /////////
    {3647, 2564}, {3695, 2568}, {3697, 2575}, {3699, 2582}, 
    {3707, 2581}, {3758, 2611}, {3776, 2629}, {3840, 2809}, 
    {3841, 2784}, {3859, 2797}, {3865, 2801}, {3927, 2726}, 
    {4009, 2568}, {4028, 2575}, {4036, 2581}, {4045, 2564}, 
    {4047, 2582}, {4075, 2611},
    // 13 bit
    /////////
    {6332, 4579}, {7362, 5125}, {7368, 5129}, {7501, 5222}, 
    {7549, 5256}, {7576, 5275}, {7582, 5279}, {7686, 5569}, 
    {7691, 5626}, {7736, 5607}, {8013, 5125}, {8019, 5129}, 
    {8165, 5222},
    // 14 bit
    /////////
    {12370, 9053}, {14905, 10376}, {14963, 10418}, 
    {15009, 10451}, {15065, 10489}, {15105, 10619}, 
    {16223, 10376}, {16291, 10418}, {16344, 10451},
    // 15 bit
    /////////'
    {25363, 18342}, {29551, 20571}, {29731, 20697}, 
    {30476, 21423}, {32162, 20571}, {32360, 20697},
    // 16 bit
    /////////
    {59057, 41112}, {59439, 41378}, {59653, 41527},
    {59699, 41559}, {64279, 41112}, {64695, 41378},
    {64928, 41527}, {64978, 41559},
    // 17 bit
    /////////
    {101705, 73551}, {117815, 82016}, {118184, 82273}, 
    {118618, 82575}, {119648, 83305}, {128233, 82016},
    {128635, 82273}, {129107, 82575}, {130267, 83305},
    // 18 bit
    /////////
    {241389, 168067}
    };

    // max iterations we'll see
    std::vector<std::vector<int>> bitclears(17);
    std::vector<std::vector<int>> q_vals(17);

    std::vector<std::vector<int>> swaps(17, std::vector<int>(4, 0));
    std::vector<std::vector<int>> negatives(17, std::vector<int>(4, 0));

    for (const auto& pair : worstCases) {
        uint32_t a_in = pair.first;
        uint32_t b_in = pair.second;
    
        xgcd_bitwise(a_in, b_in, bit_length(a_in), 4, "truncate", true, data_map, bitclears, q_vals, swaps, negatives);
    }

    printPairsWithMultipleValues(data_map);
    analyzeAndPrint(bitclears, q_vals, swaps, negatives);


    
    return 0;
}
