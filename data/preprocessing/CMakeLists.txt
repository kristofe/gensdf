cmake_minimum_required(VERSION 3.1)
project(sdf_gen)


#RUN THIS: sudo apt-get install xorg-dev libglu1-mesa-dev
#sudo apt-get install m4

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libigl/cmake)

add_subdirectory("libigl")

# https://github.com/libigl/libigl
# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)

#find_package(LIBIGL REQUIRED QUIET)

# Add your project files
file(GLOB SRCFILES sdf_gen.cpp)
add_executable(${PROJECT_NAME} ${SRCFILES})
target_link_libraries(${PROJECT_NAME} igl::core igl::glfw CGAL::CGAL)
