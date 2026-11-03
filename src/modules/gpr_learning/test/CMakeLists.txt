cmake_minimum_required(VERSION 3.10)
project(gpr_learning_regression LANGUAGES CXX)

enable_testing()
add_executable(gpr_learning_regression GPR_AlgorithmTest.cpp ../GPR_Algorithm.cpp)
target_compile_features(gpr_learning_regression PRIVATE cxx_std_14)
set_target_properties(gpr_learning_regression PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_compile_options(gpr_learning_regression PRIVATE -Wall -Wextra -Werror)
add_test(NAME gpr_learning_regression COMMAND gpr_learning_regression)
