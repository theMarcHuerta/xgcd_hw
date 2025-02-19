#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include <cassert>
#include <algorithm>
#include "xgcd_impl.h"

using namespace std;

// ----------------------
// Demo / Test main()
// ----------------------
int main() {
    // Example 1: Small example.
    {
        uint32_t a_in = 14, b_in = 9;
        // Here we set total_bits = 4, approx_bits = 4.
        XgcdResult res = xgcd_bitwise(a_in, b_in, 4, 4, "truncate", true);
        cout << "(Small) GCD of " << a_in << " and " << b_in << " is " << res.gcd
             << ", reached in " << res.iterations << " iterations.\n";
        cout << "  Average bit clears: " << res.avgBitClears << "\n";
    }

    // Example 2: Medium example.
    {
        uint32_t a_in = 14905, b_in = 10376;
        // Using total_bits = 14, approx_bits = 4.
        XgcdResult res = xgcd_bitwise(a_in, b_in, 14, 4, "truncate", true);
        cout << "(Medium) GCD of " << a_in << " and " << b_in << " is " << res.gcd
             << ", reached in " << res.iterations << " iterations.\n";
        cout << "  Average bit clears: " << res.avgBitClears << "\n";
    }

    // Example 3: A larger 32-bit example.
    {
        uint32_t a_in = 0xF2345678, b_in = 0x9ABCDEF0;  // Example hex numbers within 32 bits.
        XgcdResult res = xgcd_bitwise(a_in, b_in, 32, 4, "truncate", true);
        cout << "(Large 32-bit) GCD of " << a_in << " and " << b_in << " is " << res.gcd
             << ", reached in " << res.iterations << " iterations.\n";
        cout << "  Average bit clears: " << res.avgBitClears << "\n";
    }

    return 0;
}
