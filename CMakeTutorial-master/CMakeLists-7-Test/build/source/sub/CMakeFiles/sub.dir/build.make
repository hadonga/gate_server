# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /config/workspace/CMakeTutorial-master/CMakeLists-7-Test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build

# Include any dependencies generated for this target.
include source/sub/CMakeFiles/sub.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include source/sub/CMakeFiles/sub.dir/compiler_depend.make

# Include the progress variables for this target.
include source/sub/CMakeFiles/sub.dir/progress.make

# Include the compile flags for this target's objects.
include source/sub/CMakeFiles/sub.dir/flags.make

source/sub/CMakeFiles/sub.dir/sub.cpp.o: source/sub/CMakeFiles/sub.dir/flags.make
source/sub/CMakeFiles/sub.dir/sub.cpp.o: ../source/sub/sub.cpp
source/sub/CMakeFiles/sub.dir/sub.cpp.o: source/sub/CMakeFiles/sub.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/sub/CMakeFiles/sub.dir/sub.cpp.o"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT source/sub/CMakeFiles/sub.dir/sub.cpp.o -MF CMakeFiles/sub.dir/sub.cpp.o.d -o CMakeFiles/sub.dir/sub.cpp.o -c /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/sub/sub.cpp

source/sub/CMakeFiles/sub.dir/sub.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sub.dir/sub.cpp.i"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/sub/sub.cpp > CMakeFiles/sub.dir/sub.cpp.i

source/sub/CMakeFiles/sub.dir/sub.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sub.dir/sub.cpp.s"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/sub/sub.cpp -o CMakeFiles/sub.dir/sub.cpp.s

# Object files for target sub
sub_OBJECTS = \
"CMakeFiles/sub.dir/sub.cpp.o"

# External object files for target sub
sub_EXTERNAL_OBJECTS =

../lib/libsub.a: source/sub/CMakeFiles/sub.dir/sub.cpp.o
../lib/libsub.a: source/sub/CMakeFiles/sub.dir/build.make
../lib/libsub.a: source/sub/CMakeFiles/sub.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../../lib/libsub.a"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && $(CMAKE_COMMAND) -P CMakeFiles/sub.dir/cmake_clean_target.cmake
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sub.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/sub/CMakeFiles/sub.dir/build: ../lib/libsub.a
.PHONY : source/sub/CMakeFiles/sub.dir/build

source/sub/CMakeFiles/sub.dir/clean:
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub && $(CMAKE_COMMAND) -P CMakeFiles/sub.dir/cmake_clean.cmake
.PHONY : source/sub/CMakeFiles/sub.dir/clean

source/sub/CMakeFiles/sub.dir/depend:
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /config/workspace/CMakeTutorial-master/CMakeLists-7-Test /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/sub /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/sub/CMakeFiles/sub.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/sub/CMakeFiles/sub.dir/depend

