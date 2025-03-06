#include <iostream>
#include <mutex>
#include <cstdint>
#include <numeric>
#include <map>
#include <limits>
#include <atomic>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// 1) Define how big your histogram can be.
//    We've changed this to 32, as you said your max iterations won't exceed 32.
// #define MAX_ITERS 256
#define MAX_ITERS 32

// We'll store partial results in this struct on the GPU.
struct GpuThreadResult {
    // For truncate mode
    uint64_t sum_iters_trunc;
    double   sum_trunc_clears;
    int      trunc_max_iter;
    int      valid_pairs;
    // We'll skip storing the entire histogram here (we'll store it in deviceHistogram below).
};

// A struct to hold results of the device-based XGCD step
struct XgcdResultDevice {
    uint32_t gcd;
    int iterations;
    double avgBitClears;
};

__device__ 
void d_updateThreadResult(GpuThreadResult &res, const XgcdResultDevice &xg, int a, int b)
{
    // Summation
    res.sum_iters_trunc     += xg.iterations;
    res.sum_trunc_clears    += xg.avgBitClears;
    // Track max iteration
    if (xg.iterations > res.trunc_max_iter) {
        res.trunc_max_iter = xg.iterations;
    }
    // Count the pair
    res.valid_pairs++;
}

// Return the bit-length of x (0 if x==0).
__device__ int d_bit_length(uint32_t x) {
    if (x == 0) return 0;
    // Use built-in on GPU
    return 32 - __clz(x);
}

// The device version of xgcd_bitwise that doesn't use C++ STL.
__device__ XgcdResultDevice xgcd_bitwise_device(uint32_t a_in, uint32_t b_in,
                                               int total_bits, int approx_bits,
                                               bool integer_rounding /* or bool use_truncate */)
{
    // 1) Mask off any bits above total_bits
    uint32_t mask = (total_bits >= 32) ? 0xFFFFFFFF : ((1u << total_bits) - 1);
    uint32_t a = a_in & mask;
    uint32_t b = b_in & mask;

    if (b > a) {
        uint32_t temp = a; a = b; b = temp;
    }
    if (b == 0) {
        return {a, 0, 0.0};
    }
    if (a == 0) {
        return {b, 0, 0.0};
    }

    int iteration_count = 0;
    double total_bit_clears = 0.0;

    while (b != 0) {
        iteration_count++;

        // align b's MSB with a's MSB
        int len_a = d_bit_length(a);
        int len_b = d_bit_length(b);
        int shift_amount = len_a - len_b;
        uint32_t b_aligned = (shift_amount > 0) ? (b << shift_amount) : b;

        // get top bits
        auto get_fixed_top_bits = [&](uint32_t x_val) {
            if (x_val == 0) return 0u;
            int length = d_bit_length(x_val);
            if (length <= approx_bits) {
                return x_val << (approx_bits - length);
            } else {
                int shift_down = length - approx_bits;
                return x_val >> shift_down;
            }
        };

        uint32_t a_top = get_fixed_top_bits(a);
        uint32_t b_top = get_fixed_top_bits(b_aligned);

        // approximate Q
        uint64_t numerator = (uint64_t)a_top << approx_bits;
        uint32_t quotient = (b_top == 0) ? 0 : (uint32_t)(numerator / b_top);

        // ------------------------------------------------
        // STEP 5: Shift the quotient, then do Q++ (unconditional)
        // ------------------------------------------------
        uint32_t Q_pre_round = (quotient << shift_amount) >> (approx_bits - 1);
        uint32_t Q = (Q_pre_round >> 1);
        Q++;

        // compute product
        uint64_t product = (uint64_t)b * Q;

        // also compute product_two, picking whichever residual is smaller
        uint64_t product_two = (uint64_t)b * (Q_pre_round >> 1);

        uint32_t residual  = (a >= product) ? (a - (uint32_t)product)
                                            : ((uint32_t)product - a);
        uint32_t residual2 = (a >= product_two) ? (a - (uint32_t)product_two)
                                                : ((uint32_t)product_two - a);

        if (residual2 < residual) {
            residual = residual2;
            Q = (Q_pre_round >> 1);
        }

        // Count how many bits got cleared (msb_a - msb_res)
        int msb_a = d_bit_length(a);
        int msb_r = d_bit_length(residual);
        int cleared = msb_a - msb_r;
        if (cleared < 0) cleared = 0;
        total_bit_clears += (double)cleared;

        // prepare next iteration
        if (residual > b) {
            a = residual;
        } else {
            // swap
            uint32_t tmp = b;
            b = residual;
            a = tmp;
        }

        // If it's taking too long, break and indicate an error.
        // For 6-bit input, let's say 200 is already suspiciously large.
        if (iteration_count > 200) {
            // we can return a sentinel gcd or do something to show an error
            printf("INFINITE LOOP DETECTED: a_in=%u b_in=%u a=%u b=%u iteration_count=%d\n",
                   a_in, b_in, a, b, iteration_count);
            return { 0, iteration_count, 0.0 };
        }
    }

    // final iteration's bit clearing adjustment? (If needed, replicate your Python logic.)
    // We'll skip for brevity or just do something simpler:
    // total_bit_clears += d_bit_length(a);

    double avg_clears = 0.0;
    if (iteration_count > 0) {
        avg_clears = total_bit_clears / iteration_count;
    }

    return {a, iteration_count, avg_clears};
}

// A CUDA kernel that directly enumerates all (a, b) pairs for the assigned range of `a`.
// Each block processes one range chunk of `a`s in a grid-stride loop style.
__global__
void bruteForceKernel(int bits, int approx_bits,
                      bool force_a_msb,
                      bool int_rounding,
                      int a_min, int a_max, 
                      GpuThreadResult *deviceResults,
                      unsigned int *deviceHistogram)
{
    // -----------------
    // 1) Create a block-local histogram in shared memory
    // -----------------
    __shared__ unsigned int blockHist[MAX_ITERS];
    // Initialize the shared histogram to zero
    for (int i = threadIdx.x; i < MAX_ITERS; i += blockDim.x) {
        blockHist[i] = 0;
    }
    __syncthreads();

    // Each thread accumulates partial sums:
    GpuThreadResult myPartial;
    myPartial.sum_iters_trunc   = 0;
    myPartial.sum_trunc_clears  = 0.0;
    myPartial.trunc_max_iter    = -1;
    myPartial.valid_pairs       = 0;

    // Use a grid-stride loop over 'a':
    for (int a = blockIdx.x * blockDim.x + threadIdx.x + a_min;
         a <= a_max;
         a += gridDim.x * blockDim.x)
    {
        // printf("Thread %d (global %d) enumerating a=%d ...\n", 
        //        threadIdx.x, blockIdx.x*blockDim.x + threadIdx.x, a);
        for (int b = 1; b <= a; b++) {
            // CALL THE DEVICE VERSION (not the old xgcd_bitwise from xgcd_impl)
            XgcdResultDevice xg = xgcd_bitwise_device(
                a, b, bits, approx_bits, int_rounding
            );

            // accumulate partial data
            myPartial.sum_iters_trunc += xg.iterations;
            myPartial.sum_trunc_clears += xg.avgBitClears;
            if (xg.iterations > myPartial.trunc_max_iter) {
                myPartial.trunc_max_iter = xg.iterations;
            }
            myPartial.valid_pairs++;

            // -----------------
            // 2) Update the block-local histogram instead of the global one
            // -----------------
            int iters = xg.iterations;
            if (iters < MAX_ITERS) {
                atomicAdd(&blockHist[iters], 1);
            }
        }
    }

    // Sync so that all updates to blockHist are done before we aggregate
    __syncthreads();

    // Store partial sums in device memory
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    deviceResults[tId] = myPartial;

    // -----------------
    // 3) One final pass to merge blockHist into the global histogram
    //    Only do so for threads within the valid range
    // -----------------
    for (int i = threadIdx.x; i < MAX_ITERS; i += blockDim.x) {
        if (blockHist[i] > 0) {
            atomicAdd(&deviceHistogram[i], blockHist[i]);
        }
    }
}

// Host function to run GPU kernel, gather partial results, and print them.
int main(int argc, char* argv[])
{
    // ~~~ Reuse same argument parsing as CPU code ~~~
    int bits = 12;
    int approx_bits = 4;
    bool force_a_msb = false;
    bool int_rounding = true;

    // Minimal parse:
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc) {
            bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--approx_bits") == 0 && i + 1 < argc) {
            approx_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--force_a_msb") == 0) {
            force_a_msb = true;
        } else if (strcmp(argv[i], "--int_rounding") == 0) {
            int_rounding = true;
        }
    }

    // Decide the range for 'a'
    int a_min, a_max;
    if (force_a_msb) {
        a_min = 1 << (bits - 1);
        a_max = (1 << bits) - 1;
    } else {
        a_min = 1;  // skipping 0
        a_max = (1 << bits) - 1;
    }
    int total_a = (a_max - a_min + 1);

    // Choose a block and grid size
    int blockSize = 256; 
    int gridSize = (total_a + blockSize - 1) / blockSize;
    // But to avoid huge grids, clamp it
    if (gridSize > 65535) gridSize = 65535;

    // Allocate device array for partial results
    int totalThreads = blockSize * gridSize;
    GpuThreadResult *deviceResults;
    cudaMalloc(&deviceResults, totalThreads * sizeof(GpuThreadResult));

    // 4) Allocate device array for the global histogram
    unsigned int *deviceHistogram;
    cudaMalloc(&deviceHistogram, MAX_ITERS * sizeof(unsigned int));
    // Zero out the histogram on the device
    cudaMemset(deviceHistogram, 0, MAX_ITERS * sizeof(unsigned int));

    // Launch kernel (add our new deviceHistogram argument)
    bruteForceKernel<<<gridSize, blockSize>>>(
        bits, approx_bits, force_a_msb, int_rounding, 
        a_min, a_max, deviceResults,
        deviceHistogram
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "[DEBUG] After kernel, about to gather results..." << std::endl;

    // Copy partial results back to host
    std::vector<GpuThreadResult> hostResults(totalThreads);
    cudaMemcpy(hostResults.data(), deviceResults, totalThreads * sizeof(GpuThreadResult), cudaMemcpyDeviceToHost);
    cudaFree(deviceResults);

    // Also copy the histogram back from GPU
    std::vector<unsigned int> hostHistogram(MAX_ITERS);
    cudaMemcpy(
        hostHistogram.data(), deviceHistogram,
        MAX_ITERS * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    cudaFree(deviceHistogram);

    // Aggregate results on the host
    uint64_t total_trunc_iters = 0;
    double   total_trunc_clears = 0.0;
    int      global_trunc_max_iter = -1;
    int      total_valid_pairs = 0;

    for (auto &res : hostResults) {
        total_trunc_iters     += res.sum_iters_trunc;
        total_trunc_clears    += res.sum_trunc_clears;
        if (res.trunc_max_iter > global_trunc_max_iter) {
            global_trunc_max_iter = res.trunc_max_iter;
        }
        total_valid_pairs     += res.valid_pairs;
    }

    // Print some final info
    std::cout << "\n===== GPU RESULTS (TRUNCATE MODE) =====\n";
    uint64_t totalPairs = (uint64_t)0;
    for(int a=a_min; a<=a_max; a++){
        totalPairs += a; 
    }
    double trunc_iter_mean = double(total_trunc_iters) / double(totalPairs);
    // As a quick hack, we interpret sum_trunc_clears as a total, but we must 
    // also consider how you originally computed average. You might refine this as needed.

    double trunc_clears_mean = (total_trunc_clears / total_trunc_iters) * trunc_iter_mean;

    std::cout << "  Mean Iterations    : " << trunc_iter_mean << "\n";
    std::cout << "  Mean Bit Clears    : " << trunc_clears_mean << "\n";
    std::cout << "  Max Iterations     : " << global_trunc_max_iter << "\n";
    std::cout << "  Valid Pairs        : " << total_valid_pairs << "\n";
    std::cout << "Tested a total of " << totalPairs << " pairs.\n";

    // 7) Print out the histogram
    std::cout << "\n===== HISTOGRAM (TRUNCATE MODE) =====\n";
    for (int i = 0; i < MAX_ITERS; i++) {
        unsigned int count = hostHistogram[i];
        if (count > 0) {
            std::cout << "Iterations " << i << ": " << count << "\n";
        }
    }

    std::cout << "\n--- End of GPU brute force test ---\n";
    return 0;
} 