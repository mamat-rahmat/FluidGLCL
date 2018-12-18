cmake_minimum_required(VERSION 3.0.0)

project(FluidGLCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#add_definitions("/D\"_CRT_SECURE_NO_WARNINGS\"")
#add_definitions("/D\"_SCL_SECURE_NO_WARNINGS\"")

set (CMAKE_CXX_STANDARD 11)

find_package(Qt5 REQUIRED Core Gui)
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED)

set(FluidCLGL_SRC
    main.cpp
)

set(FluidCLGL_HEADERS
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories(${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

qt5_generate_moc(main.cpp main.moc)

add_executable(FluidCLGL ${FluidCLGL_SRC} ${FluidCLGL_HEADERS} main.moc)

target_link_libraries(FluidCLGL ${Boost_LIBRARIES})
target_link_libraries(FluidCLGL Qt5::Core)
target_link_libraries(FluidCLGL Qt5::Gui)
target_link_libraries(FluidCLGL OpenCL::OpenCL)
target_link_libraries(FluidCLGL opengl32)

