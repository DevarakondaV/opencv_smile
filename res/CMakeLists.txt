#
set(RESOURCE_FILES nn1.yml haarcascade_frontalface_default.xml)
file(COPY ${RESOURCE_FILES} DESTINATION "${CMAKE_BINARY_DIR}/res")
