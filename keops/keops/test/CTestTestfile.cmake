# CMake generated Testfile for 
# Source directory: /home/lcosmo/PROJECTS/keops/keops/test
# Build directory: /home/lcosmo/PROJECTS/keops/keops/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unit_test_compile "unit_test_compile")
set_tests_properties(unit_test_compile PROPERTIES  _BACKTRACE_TRIPLES "/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;12;add_test;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;18;add_test_helper;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;0;")
add_test(unit_test_tensordot "unit_test_tensordot")
set_tests_properties(unit_test_tensordot PROPERTIES  _BACKTRACE_TRIPLES "/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;12;add_test;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;20;add_test_helper;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;0;")
add_test(unit_test_grad1conv "unit_test_grad1conv")
set_tests_properties(unit_test_grad1conv PROPERTIES  _BACKTRACE_TRIPLES "/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;12;add_test;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;23;add_test_helper;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;0;")
add_test(unit_test_conv "unit_test_conv")
set_tests_properties(unit_test_conv PROPERTIES  _BACKTRACE_TRIPLES "/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;12;add_test;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;25;add_test_helper;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;0;")
add_test(unit_test_tensordot_cuda "unit_test_tensordot_cuda")
set_tests_properties(unit_test_tensordot_cuda PROPERTIES  _BACKTRACE_TRIPLES "/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;12;add_test;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;27;add_test_helper;/home/lcosmo/PROJECTS/keops/keops/test/CMakeLists.txt;0;")
