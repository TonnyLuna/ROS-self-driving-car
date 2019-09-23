execute_process(COMMAND "/home/tonny/catvehicle_ws/build/catvehicle/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/tonny/catvehicle_ws/build/catvehicle/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
