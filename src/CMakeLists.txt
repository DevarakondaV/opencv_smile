set(SOURCES opencv_smile.cpp)

find_package(OpenCV REQUIRED)

add_executable(SmileNN ${SOURCES})
target_link_libraries(SmileNN ${OpenCV_LIBS})
