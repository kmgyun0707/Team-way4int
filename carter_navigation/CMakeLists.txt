cmake_minimum_required(VERSION 3.5)
project(carter_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  params
  maps
  rviz2
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/yolo_commander.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
