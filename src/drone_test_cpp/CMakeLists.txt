cmake_minimum_required(VERSION 3.8)
project(drone_test_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)   

set(ONNXRUNTIME_ROOT
  /home/vasudevan/Downloads/onnxruntime-linux-x64-1.23.2
)

include_directories(
  include
  ${ONNXRUNTIME_ROOT}/include
)

link_directories(
  ${ONNXRUNTIME_ROOT}/lib
)

add_library(drone_core
  src/vehicle.cpp
  src/model.cpp
)

target_include_directories(drone_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(drone_core
  rclcpp
  px4_msgs
)


target_link_libraries(drone_core
  Eigen3::Eigen
)

target_link_libraries(drone_core
  onnxruntime
)

add_executable(drone_test
  src/test_dron_node.cpp
)

ament_target_dependencies( drone_test rclcpp std_msgs px4_msgs)

target_link_libraries(drone_test Eigen3::Eigen)
target_link_libraries(drone_test
  drone_core
)
install(
  DIRECTORY include/
  DESTINATION include
)

# install(TARGETS
#   drone_test
#   DESTINATION lib/${PROJECT_NAME})
install(
  TARGETS drone_core drone_test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to allc source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
