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
include source/add/CMakeFiles/add.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include source/add/CMakeFiles/add.dir/compiler_depend.make

# Include the progress variables for this target.
include source/add/CMakeFiles/add.dir/progress.make

# Include the compile flags for this target's objects.
include source/add/CMakeFiles/add.dir/flags.make

source/add/CMakeFiles/add.dir/add.cpp.o: source/add/CMakeFiles/add.dir/flags.make
source/add/CMakeFiles/add.dir/add.cpp.o: ../source/add/add.cpp
source/add/CMakeFiles/add.dir/add.cpp.o: source/add/CMakeFiles/add.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/add/CMakeFiles/add.dir/add.cpp.o"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT source/add/CMakeFiles/add.dir/add.cpp.o -MF CMakeFiles/add.dir/add.cpp.o.d -o CMakeFiles/add.dir/add.cpp.o -c /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add.cpp

source/add/CMakeFiles/add.dir/add.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/add.dir/add.cpp.i"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add.cpp > CMakeFiles/add.dir/add.cpp.i

source/add/CMakeFiles/add.dir/add.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/add.dir/add.cpp.s"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add.cpp -o CMakeFiles/add.dir/add.cpp.s

source/add/CMakeFiles/add.dir/add3.cpp.o: source/add/CMakeFiles/add.dir/flags.make
source/add/CMakeFiles/add.dir/add3.cpp.o: ../source/add/add3.cpp
source/add/CMakeFiles/add.dir/add3.cpp.o: source/add/CMakeFiles/add.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object source/add/CMakeFiles/add.dir/add3.cpp.o"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT source/add/CMakeFiles/add.dir/add3.cpp.o -MF CMakeFiles/add.dir/add3.cpp.o.d -o CMakeFiles/add.dir/add3.cpp.o -c /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add3.cpp

source/add/CMakeFiles/add.dir/add3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/add.dir/add3.cpp.i"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add3.cpp > CMakeFiles/add.dir/add3.cpp.i

source/add/CMakeFiles/add.dir/add3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/add.dir/add3.cpp.s"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add/add3.cpp -o CMakeFiles/add.dir/add3.cpp.s

# Object files for target add
add_OBJECTS = \
"CMakeFiles/add.dir/add.cpp.o" \
"CMakeFiles/add.dir/add3.cpp.o"

# External object files for target add
add_EXTERNAL_OBJECTS =

../lib/libadd.a: source/add/CMakeFiles/add.dir/add.cpp.o
../lib/libadd.a: source/add/CMakeFiles/add.dir/add3.cpp.o
../lib/libadd.a: source/add/CMakeFiles/add.dir/build.make
../lib/libadd.a: source/add/CMakeFiles/add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library ../../../lib/libadd.a"
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && $(CMAKE_COMMAND) -P CMakeFiles/add.dir/cmake_clean_target.cmake
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/add/CMakeFiles/add.dir/build: ../lib/libadd.a
.PHONY : source/add/CMakeFiles/add.dir/build

source/add/CMakeFiles/add.dir/clean:
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add && $(CMAKE_COMMAND) -P CMakeFiles/add.dir/cmake_clean.cmake
.PHONY : source/add/CMakeFiles/add.dir/clean

source/add/CMakeFiles/add.dir/depend:
	cd /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /config/workspace/CMakeTutorial-master/CMakeLists-7-Test /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/source/add /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add /config/workspace/CMakeTutorial-master/CMakeLists-7-Test/build/source/add/CMakeFiles/add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/add/CMakeFiles/add.dir/depend

