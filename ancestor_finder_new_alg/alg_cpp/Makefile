# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread

# Target and source files
TARGET = bruteforce
SOURCES = xgcd_bruteforce.cpp

# Default rule to build and run the project
all: $(TARGET)
	./$(TARGET) --bits 15 --approx_bits 4 --force_a_msb --threads 24

# Rule to compile the target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean up generated files
clean:
	rm -f $(TARGET) *.o *.out *.exe
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
