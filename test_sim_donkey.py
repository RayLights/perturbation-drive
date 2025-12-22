import platform
import os
from pathlib import Path
from perturbationdrive import PerturbationDrive, CustomRoadGenerator
from examples.self_driving_sandbox_donkey.sdsandbox_simulator import SDSandboxSimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

def get_simulator_path():
    """Simple OS detection for simulator path"""
    base_path = "./examples/self_driving_sandbox_donkey/sim"
    system = platform.system().lower()
    
    # Platform-specific simulator paths
    paths = {
        "darwin": f"{base_path}/sdsim_macos/sdsim_macos.app",
        "linux": f"{base_path}/sdsim_linux/sdsim_binary.x86_64", 
        "windows": f"{base_path}/sdsim_windows/sdsim_binary.exe" # do we support windows ? 
    }
    
    if system not in paths:
        raise OSError(f"Unsupported platform: {system}")
    
    path = paths[system]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Simulator not found at: {path}")
    
    return path

try:
    # Automatic platform detection
    simulator_path = get_simulator_path()
    print(f"Detected platform: {platform.system()}")
    print(f"Using simulator: {simulator_path}")
    
    simulator = SDSandboxSimulator(
        simulator_exe_path=simulator_path,
        host="127.0.0.1",
        port=9091,
        show_image_cb=True
    )
    
    ads = Dave2Agent(model_path="./examples/models/checkpoints/dave_90k_v1.h5")
    road_angles = [0, -35, 0, -17, -35, 35, 6, -22]
    road_segments = [25, 25, 25, 25, 25, 25, 25, 25]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))

    benchmarking_obj = PerturbationDrive(simulator, ads)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    attention_map = {
                "map": "grad_cam",
                "model": model,
                "threshold": 0.1,
                "layer": "conv2d_5",
            }
    

    perturbations = [
        "gaussian_noise",
        "poisson_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "increase_brightness",
        "contrast",
        "elastic",
        "pixelate",
        "jpeg_filter",
        "translate_image",
        "scale_image",
        "splatter_mapping",
        "dotted_lines_mapping",
        "zigzag_mapping",
        "canny_edges_mapping",
        "speckle_noise_filter",
        "false_color_filter",
        "high_pass_filter",
        "low_pass_filter",
        "phase_scrambling",
        "histogram_equalisation",
        "reflection_filter",
        "white_balance_filter",
        "sharpen_filter",
        "grayscale_filter",
        "fog_filter",
        "frost_filter",
        "snow_filter",
        "posterize_filter",
        "cutout_filter",
        "sample_pairing_filter",
        "gaussian_blur",
        "saturation_filter",
        "saturation_decrease_filter",
        "new_rain_filter",
        "static_rain_filter",
        # "candy",
        # "la_muse",
        # "mosaic",
        # "feathers",
        # "the_scream",
        # "udnie",
        # "effects_attention_regions",
        # "effects_attention_regions_dynamic",
        # "effects_rain_dynamic",
        # "dynamic_snow_filter",
        # "dynamic_rain_filter",
        # "dynamic_object_overlay",
        # "dynamic_sun_filter",
        # "dynamic_lightning_filter",
        # "dynamic_smoke_filter",
        # "static_snow_filter",
        # "static_rain_filter",
        # "static_object_overlay",
        # "static_sun_filter",
        # "static_lightning_filter",
        # "static_smoke_filter",
        # "dynamic_raindrop_filter",
    ]

    benchmarking_obj.grid_seach(
        perturbation_functions=perturbations,
        attention_map = attention_map,
        road_generator=road_generator,
        road_angles=road_angles,
        road_segments=road_segments,
        log_dir=f"./donkey_logs_{time}.json",
        overwrite_logs=True,
        image_size=(240, 320),  # images are resized to these values
        test_model=True,  #choose to run NPC or model
        perturb=True,  
    )
    print(f"{5 * '#'} Finished Running SDSandBox Sim {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} SDSandBox Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )