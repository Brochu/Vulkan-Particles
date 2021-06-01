cmake_minimum_required(VERSION 3.20)

project(ParticlesVulkan)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE "Debug")

find_package(Vulkan REQUIRED)

find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
link_directories(${GLM_LIBRARY_DIRS})
add_definitions(${GLM_DEFINITIONS})

add_subdirectory($ENV{GLFW3_ROOT} ./bin)
include_directories($ENV{GLFW3_ROOT}/include)

file (GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/source/*.cpp
)
file (GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/headers/*.h
)

add_executable(ParticlesVulkan)

target_sources(ParticlesVulkan PRIVATE ${SRC_FILES} ${HEADER_FILES})
target_include_directories(ParticlesVulkan PRIVATE ${PROJECT_SOURCE_DIR}/headers)

target_link_libraries(ParticlesVulkan PRIVATE Vulkan::Vulkan ${GLM_LIBRARIES} glfw ${GLFW_LIBRARIES})
