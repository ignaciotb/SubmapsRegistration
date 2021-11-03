project(feature_matching)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

include_directories(include)

add_executable (submap_registration
	src/pcl_registration.cpp)
target_link_libraries (submap_registration
	${PCL_LIBRARIES})

