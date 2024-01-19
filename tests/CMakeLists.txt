add_definitions(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

ADD_UNIT_TEST(cpp-test-estimator test_estimator.cpp)
target_link_libraries(cpp-test-estimator ${PROJECT_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

ADD_UNIT_TEST(cpp-test-mh-estimator test_mh_estimator.cpp)
target_link_libraries(cpp-test-mh-estimator ${PROJECT_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

# ADD_UNIT_TEST(cpp-test-mh-varying-estimator test_mh_varying_estimator.cpp)
# target_link_libraries(cpp-test-mh-estimator ${PROJECT_NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})


add_test(NAME exe-fail COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAMESPACE}-${PROJECT_NAME} 3 4)

if(BUILD_PYTHON_INTERFACE)
  add_python_unit_test("py-test-estimators" "tests/test_estimators.py" src tests)
  add_python_unit_test("py-test-dam-delta-tau" "tests/test_dam_delta_tau.py" src tests)
  add_python_unit_test("py-test-contact-dam" "tests/test_contact_dam.py" src tests)
endif()
