import numpy as np
import inspect
from typing import Any, Optional, Callable, Dict, List

# Import specific LiDAR functions
from .perturbationfuncs import (
    lidar_inject_ghost_points,
    lidar_reduce_reflectivity,
    lidar_simulate_adverse_weather,
    pts_motion,
    transform_points,
    reduce_LiDAR_beamsV2,
    pointsreducing,
    simulate_snow_sweep, # ransac for ground truth
    simulate_snow, # needs ground truth labels
    simulate_fog,
)

# Mapping of configuration strings to actual function objects
LIDAR_FUNCTION_MAPPING = {
    "lidar_inject_ghost_points": lidar_inject_ghost_points,
    "lidar_reduce_reflectivity": lidar_reduce_reflectivity,
    "lidar_simulate_adverse_weather": lidar_simulate_adverse_weather,
    "pts_motion": pts_motion,
    "transform_points": transform_points,
    "reduce_LiDAR_beamsV2": reduce_LiDAR_beamsV2,
    "pointsreducing": pointsreducing,
    "simulate_snow_sweep": simulate_snow_sweep,
    "simulate_snow": simulate_snow,
    "simulate_fog": simulate_fog,
}

class LidarPerturbation:
    """
    Controller for LiDAR-specific perturbations.
    Manages the mapping of function names to implementations and handles random number generation.
    """

    def __init__(
        self,
        funcs: Optional[List[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize the LiDAR perturbation controller.

        :param funcs: List of perturbation names to enable. If None or empty, all are enabled.
        :param rng: Optional numpy random generator for reproducibility.
        """
        # If no specific functions requested, load all available in the mapping
        if funcs is None or len(funcs) == 0:
            self._func_map: Dict[str, Callable] = dict(LIDAR_FUNCTION_MAPPING)
        else:
            # Only load the functions requested by the user
            self._func_map = {
                name: LIDAR_FUNCTION_MAPPING[name]
                for name in funcs
                if name in LIDAR_FUNCTION_MAPPING
            }

        if len(self._func_map) == 0:
            # If the user requested functions but none matched valid keys
            print(f"Warning: No valid LiDAR perturbations found in list {funcs}. Available: {list(LIDAR_FUNCTION_MAPPING.keys())}")

        # Initialize Random Number Generator
        self._rng = rng if rng is not None else np.random.default_rng()

    def list_available(self) -> List[str]:
        """Returns a list of available perturbation names."""
        return list(self._func_map.keys())

    def perturbation(
        self,
        point_cloud: np.ndarray,
        perturbation_name: str,
        intensity: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply a specific perturbation to the point cloud.

        :param point_cloud: Numpy array of shape (N, C) (usually N, 3 or N, 4)
        :param perturbation_name: string name of the filter (e.g., "lidar_adverse_weather")
        :param intensity: int scale (0-4)
        """
        if perturbation_name == "":
            return np.asarray(point_cloud).copy()

        if perturbation_name not in self._func_map:
            raise KeyError(
                f"Unknown LiDAR perturbation '{perturbation_name}'. Available: {self.list_available()}"
            )

        func = self._func_map[perturbation_name]
        
        # Prepare arguments
        bound_kwargs = dict(kwargs)
        signature = inspect.signature(func)

        # Inject the random number generator if the function expects it
        if "rng" in signature.parameters and "rng" not in bound_kwargs:
            bound_kwargs["rng"] = self._rng

        # Execute the function
        # Note: Lidar functions generally expect (scale, point_cloud, ...)
        return func(intensity, point_cloud, **bound_kwargs)

    def __call__(
        self,
        point_cloud: np.ndarray,
        perturbation_name: str,
        intensity: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Allows calling the instance directly as a function."""
        return self.perturbation(point_cloud, perturbation_name, intensity, **kwargs)