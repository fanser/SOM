# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data01/home/fanzhongyue/workspace/SOM/cpp/som

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data01/home/fanzhongyue/workspace/SOM/cpp/som/build

# Include any dependencies generated for this target.
include CMakeFiles/som.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/som.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/som.dir/flags.make

CMakeFiles/som.dir/test/test.cc.o: CMakeFiles/som.dir/flags.make
CMakeFiles/som.dir/test/test.cc.o: ../test/test.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /data01/home/fanzhongyue/workspace/SOM/cpp/som/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/som.dir/test/test.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/som.dir/test/test.cc.o -c /data01/home/fanzhongyue/workspace/SOM/cpp/som/test/test.cc

CMakeFiles/som.dir/test/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/test/test.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data01/home/fanzhongyue/workspace/SOM/cpp/som/test/test.cc > CMakeFiles/som.dir/test/test.cc.i

CMakeFiles/som.dir/test/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/test/test.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data01/home/fanzhongyue/workspace/SOM/cpp/som/test/test.cc -o CMakeFiles/som.dir/test/test.cc.s

CMakeFiles/som.dir/test/test.cc.o.requires:
.PHONY : CMakeFiles/som.dir/test/test.cc.o.requires

CMakeFiles/som.dir/test/test.cc.o.provides: CMakeFiles/som.dir/test/test.cc.o.requires
	$(MAKE) -f CMakeFiles/som.dir/build.make CMakeFiles/som.dir/test/test.cc.o.provides.build
.PHONY : CMakeFiles/som.dir/test/test.cc.o.provides

CMakeFiles/som.dir/test/test.cc.o.provides.build: CMakeFiles/som.dir/test/test.cc.o

CMakeFiles/som.dir/src/base/circleTopo.cc.o: CMakeFiles/som.dir/flags.make
CMakeFiles/som.dir/src/base/circleTopo.cc.o: ../src/base/circleTopo.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /data01/home/fanzhongyue/workspace/SOM/cpp/som/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/som.dir/src/base/circleTopo.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/som.dir/src/base/circleTopo.cc.o -c /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/base/circleTopo.cc

CMakeFiles/som.dir/src/base/circleTopo.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/src/base/circleTopo.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/base/circleTopo.cc > CMakeFiles/som.dir/src/base/circleTopo.cc.i

CMakeFiles/som.dir/src/base/circleTopo.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/src/base/circleTopo.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/base/circleTopo.cc -o CMakeFiles/som.dir/src/base/circleTopo.cc.s

CMakeFiles/som.dir/src/base/circleTopo.cc.o.requires:
.PHONY : CMakeFiles/som.dir/src/base/circleTopo.cc.o.requires

CMakeFiles/som.dir/src/base/circleTopo.cc.o.provides: CMakeFiles/som.dir/src/base/circleTopo.cc.o.requires
	$(MAKE) -f CMakeFiles/som.dir/build.make CMakeFiles/som.dir/src/base/circleTopo.cc.o.provides.build
.PHONY : CMakeFiles/som.dir/src/base/circleTopo.cc.o.provides

CMakeFiles/som.dir/src/base/circleTopo.cc.o.provides.build: CMakeFiles/som.dir/src/base/circleTopo.cc.o

CMakeFiles/som.dir/src/som.cc.o: CMakeFiles/som.dir/flags.make
CMakeFiles/som.dir/src/som.cc.o: ../src/som.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /data01/home/fanzhongyue/workspace/SOM/cpp/som/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/som.dir/src/som.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/som.dir/src/som.cc.o -c /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/som.cc

CMakeFiles/som.dir/src/som.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/src/som.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/som.cc > CMakeFiles/som.dir/src/som.cc.i

CMakeFiles/som.dir/src/som.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/src/som.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /data01/home/fanzhongyue/workspace/SOM/cpp/som/src/som.cc -o CMakeFiles/som.dir/src/som.cc.s

CMakeFiles/som.dir/src/som.cc.o.requires:
.PHONY : CMakeFiles/som.dir/src/som.cc.o.requires

CMakeFiles/som.dir/src/som.cc.o.provides: CMakeFiles/som.dir/src/som.cc.o.requires
	$(MAKE) -f CMakeFiles/som.dir/build.make CMakeFiles/som.dir/src/som.cc.o.provides.build
.PHONY : CMakeFiles/som.dir/src/som.cc.o.provides

CMakeFiles/som.dir/src/som.cc.o.provides.build: CMakeFiles/som.dir/src/som.cc.o

# Object files for target som
som_OBJECTS = \
"CMakeFiles/som.dir/test/test.cc.o" \
"CMakeFiles/som.dir/src/base/circleTopo.cc.o" \
"CMakeFiles/som.dir/src/som.cc.o"

# External object files for target som
som_EXTERNAL_OBJECTS =

../lib/libsom.a: CMakeFiles/som.dir/test/test.cc.o
../lib/libsom.a: CMakeFiles/som.dir/src/base/circleTopo.cc.o
../lib/libsom.a: CMakeFiles/som.dir/src/som.cc.o
../lib/libsom.a: CMakeFiles/som.dir/build.make
../lib/libsom.a: CMakeFiles/som.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../lib/libsom.a"
	$(CMAKE_COMMAND) -P CMakeFiles/som.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/som.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/som.dir/build: ../lib/libsom.a
.PHONY : CMakeFiles/som.dir/build

CMakeFiles/som.dir/requires: CMakeFiles/som.dir/test/test.cc.o.requires
CMakeFiles/som.dir/requires: CMakeFiles/som.dir/src/base/circleTopo.cc.o.requires
CMakeFiles/som.dir/requires: CMakeFiles/som.dir/src/som.cc.o.requires
.PHONY : CMakeFiles/som.dir/requires

CMakeFiles/som.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/som.dir/cmake_clean.cmake
.PHONY : CMakeFiles/som.dir/clean

CMakeFiles/som.dir/depend:
	cd /data01/home/fanzhongyue/workspace/SOM/cpp/som/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data01/home/fanzhongyue/workspace/SOM/cpp/som /data01/home/fanzhongyue/workspace/SOM/cpp/som /data01/home/fanzhongyue/workspace/SOM/cpp/som/build /data01/home/fanzhongyue/workspace/SOM/cpp/som/build /data01/home/fanzhongyue/workspace/SOM/cpp/som/build/CMakeFiles/som.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/som.dir/depend

