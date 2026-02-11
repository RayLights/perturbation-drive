import unittest

import numpy as np

from perturbationdrive.perturbationfuncs import (
    lidar_range_noise,
    lidar_sensor_miscalibration,
)


class LidarPerturbationsTestCase(unittest.TestCase):
    def setUp(self):
        self.point_cloud = np.array(
            [
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 1.0, 0.0, 0.7],
                [0.0, 0.0, 1.0, 0.5],
                [1.0, 1.0, 1.0, 0.3],
            ],
            dtype=np.float32,
        )

    def test_lidar_range_noise_preserves_shape(self):
        rng = np.random.default_rng(0)
        output = lidar_range_noise(2, self.point_cloud, rng=rng)

        self.assertEqual(output.shape, self.point_cloud.shape)
        self.assertTrue(np.allclose(output[:, 3:], self.point_cloud[:, 3:]))
        self.assertFalse(np.allclose(output[:, :3], self.point_cloud[:, :3]))

    def test_lidar_sensor_miscalibration_preserves_intensity(self):
        rng = np.random.default_rng(1)
        output = lidar_sensor_miscalibration(3, self.point_cloud, rng=rng)

        self.assertEqual(output.shape, self.point_cloud.shape)
        self.assertTrue(np.allclose(output[:, 3:], self.point_cloud[:, 3:]))
        self.assertFalse(np.allclose(output[:, :3], self.point_cloud[:, :3]))


if __name__ == "__main__":
    unittest.main()
