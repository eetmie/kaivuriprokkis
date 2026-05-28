cmake_minimum_required(VERSION 3.8)
project(kaivuri_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch robot rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
