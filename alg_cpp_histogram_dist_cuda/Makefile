# Compiler and flags for the CPU build
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread

# Compiler/flags for the GPU build
NVCC = nvcc
NVCCFLAGS = -std=c++17 -arch=sm_86 -Xcompiler -Wall,-Wextra,-pthread \
    -I/usr/local/cuda/include

# CPU targets
TARGET = bruteforce
SOURCES = xgcd_bruteforce.cpp

# GPU targets
TARGET_GPU = bruteforce_gpu
SOURCES_GPU = xgcd_bruteforce_gpu.cu

# Default rule: build + run CPU version
all: $(TARGET)
	./$(TARGET) --bits 12 --approx_bits 4 --force_a_msb --threads 14

# CPU build rule
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# GPU build rule
$(TARGET_GPU): $(SOURCES_GPU) xgcd_impl.h
	$(NVCC) $(NVCCFLAGS) -o $(TARGET_GPU) $(SOURCES_GPU)
# Optional shortcut to build & run GPU
gpu: $(TARGET_GPU)
	./$(TARGET_GPU) --bits 20 --approx_bits 3 --force_a_msb

# Clean up
clean:
	rm -f $(TARGET) $(TARGET_GPU) *.o *.out *.exe
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
