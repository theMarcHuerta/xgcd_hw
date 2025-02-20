# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread

# Target and source files
TARGET = xgcd_impl_dupe_finder
SOURCES = xgcd_impl_dupe_finder.cpp

# Default rule to build and run the project
all: $(TARGET)
	./$(TARGET)

# Rule to compile the target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Clean up generated files
clean:
	rm -f $(TARGET) *.o *.out *.exe
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
