# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jshin/csc200/multi-vector-retrieval

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jshin/csc200/multi-vector-retrieval

# Include any dependencies generated for this target.
include CMakeFiles/BruteForceProgressive.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/BruteForceProgressive.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/BruteForceProgressive.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BruteForceProgressive.dir/flags.make

CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o: CMakeFiles/BruteForceProgressive.dir/flags.make
CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o: src/app/BruteForceProgressive.cpp
CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o: CMakeFiles/BruteForceProgressive.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jshin/csc200/multi-vector-retrieval/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o -MF CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o.d -o CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o -c /home/jshin/csc200/multi-vector-retrieval/src/app/BruteForceProgressive.cpp

CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jshin/csc200/multi-vector-retrieval/src/app/BruteForceProgressive.cpp > CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.i

CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jshin/csc200/multi-vector-retrieval/src/app/BruteForceProgressive.cpp -o CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.s

# Object files for target BruteForceProgressive
BruteForceProgressive_OBJECTS = \
"CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o"

# External object files for target BruteForceProgressive
BruteForceProgressive_EXTERNAL_OBJECTS =

BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: CMakeFiles/BruteForceProgressive.dir/src/app/BruteForceProgressive.cpp.o
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: CMakeFiles/BruteForceProgressive.dir/build.make
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libspdlog.so.1.9.2
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopenblas.so
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.a
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1
BruteForceProgressive.cpython-310-x86_64-linux-gnu.so: CMakeFiles/BruteForceProgressive.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jshin/csc200/multi-vector-retrieval/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module BruteForceProgressive.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BruteForceProgressive.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BruteForceProgressive.dir/build: BruteForceProgressive.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/BruteForceProgressive.dir/build

CMakeFiles/BruteForceProgressive.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BruteForceProgressive.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BruteForceProgressive.dir/clean

CMakeFiles/BruteForceProgressive.dir/depend:
	cd /home/jshin/csc200/multi-vector-retrieval && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jshin/csc200/multi-vector-retrieval /home/jshin/csc200/multi-vector-retrieval /home/jshin/csc200/multi-vector-retrieval /home/jshin/csc200/multi-vector-retrieval /home/jshin/csc200/multi-vector-retrieval/CMakeFiles/BruteForceProgressive.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/BruteForceProgressive.dir/depend

