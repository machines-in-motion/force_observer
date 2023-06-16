"""@package
Unit-tests for PID and related factories
"""

from pathlib import Path
import unittest
import yaml
import os
from package_template.pid import (
    get_config_file_pid,
    get_default_pid,
    PythonPID,
    DefaultConfiguration,
)


## set of unit-tests for PID and related factories
class PID_TESTCASE(unittest.TestCase):

    YAML_CONFIG_FILE = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "pid_config_test.yaml"
    )

    def setUp(self):
        """
        Testing the function get_config_file_pid will require a yaml config
        file creating this file here, which will be called before running all
        tests.
        """
        with open(self.YAML_CONFIG_FILE, "w+") as f:
            yaml.dump(DefaultConfiguration(), f)

    def tearDown(self):
        """Deleting the file created above when we leave the tests."""
        os.remove(self.YAML_CONFIG_FILE)

    def test_config_file_factory(self):
        """Testing creating a pid controller from file works as expected."""
        pid_ctrl = get_config_file_pid(
            config_file_path=self.YAML_CONFIG_FILE, verbose=False
        )
        gains = pid_ctrl.get_gains()
        self.assertEqual(gains["kp"], DefaultConfiguration.kp)
        self.assertEqual(gains["kd"], DefaultConfiguration.kd)
        self.assertEqual(gains["ki"], DefaultConfiguration.ki)

    def test_config_file_factory_from_default_file(self):
        """Testing creating a pid controller from default config file."""
        get_config_file_pid(verbose=False)

    def test_default_factory(self):
        """Testing creation using default config."""
        pid_ctrl = get_default_pid()
        gains = pid_ctrl.get_gains()
        self.assertEqual(gains["kp"], DefaultConfiguration.kp)
        self.assertEqual(gains["kd"], DefaultConfiguration.kd)
        self.assertEqual(gains["ki"], DefaultConfiguration.ki)

    def test_exception_on_non_existing_config_file(self):
        """
        Testing creating a pid controller from a non existing file raises
        an exception.
        """
        with self.assertRaises(Exception):
            get_config_file_pid(
                config_file_path="non_existing_path", verbose=False
            )

    def test_integral(self):
        """Testing integral integrates, except if ki is zero."""

        class Config:
            kp, kd, ki = 1, 1, 1

        config = Config()
        position = 1
        velocity = 1
        position_target = 2
        delta_time = 0.1
        pid_ctrl = PythonPID(config)
        force_1 = pid_ctrl.compute(
            position, velocity, position_target, delta_time
        )
        force_2 = pid_ctrl.compute(
            position, velocity, position_target, delta_time
        )
        self.assertNotEqual(force_1, force_2)
        config.ki = 0
        force_3 = pid_ctrl.compute(
            position, velocity, position_target, delta_time
        )
        force_4 = pid_ctrl.compute(
            position, velocity, position_target, delta_time
        )
        self.assertEqual(force_3, force_4)

        # testing force is zero if already at target
        config = Config()
        position = 1
        velocity = 0
        position_target = 1
        delta_time = 0.1
        pid_ctrl = PythonPID(config)
        force = pid_ctrl.compute(
            position, velocity, position_target, delta_time
        )
        self.assertEqual(force, 0)

        # testing the controller pushes in the right direction
        config = Config()
        position = 0
        velocity = 0
        target_position = 1
        delta_time = 0.1
        pid_ctrl = PythonPID(config)
        force = pid_ctrl.compute(
            position, velocity, target_position, delta_time
        )
        self.assertTrue(force > 0)
        target_position = -1
        force = pid_ctrl.compute(
            position, velocity, target_position, delta_time
        )
        self.assertTrue(force < 0)
        target_position = 0
        velocity = 1.0
        force = pid_ctrl.compute(
            position, velocity, target_position, delta_time
        )
        self.assertTrue(force < 0)
