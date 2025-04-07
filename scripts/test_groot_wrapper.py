import numpy as np
import torch
from eval_wrapper import GROOTWrapper
import tyro
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestConfig:
    model_ckpt_folder: str = "/mnt/8tb-drive/groot_ckpt/dpgs_sim_coffee"
    ckpt_id: int = 10000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    text_prompt: str = "put the white cup on the coffee machine"
    batch_size: int = 1
    sequence_length: int = 10
    num_cameras: int = 2
    image_height: int = 270
    image_width: int = 480
    num_joints: int = 14
    num_grippers: int = 2

def create_dummy_batch(config: TestConfig) -> dict:
    """Create a dummy batch with correct dimensions and dtypes."""
    # Create observation data (B, T, num_cameras, H, W, C)
    observation = np.random.randint(
        0, 255, 
        size=(config.batch_size, config.sequence_length, config.num_cameras, 
              config.image_height, config.image_width, 3), 
        dtype=np.uint8
    )
    
    # Create proprioceptive data (B, T, num_joints + num_grippers)
    proprio = np.random.randn(
        config.batch_size, 
        config.sequence_length, 
        config.num_joints + config.num_grippers
    ).astype(np.float64)
    
    return {
        "observation": observation,
        "proprio": proprio
    }

def main(config: TestConfig):
    # Initialize the wrapper
    wrapper = GROOTWrapper(
        model_ckpt_folder=config.model_ckpt_folder,
        ckpt_id=config.ckpt_id,
        device=config.device,
        text_prompt=config.text_prompt
    )
    
    # Create dummy batch
    nbatch = create_dummy_batch(config)
    
    # Print input shapes and dtypes
    print("\nInput shapes and dtypes:")
    print(f"observation shape: {nbatch['observation'].shape}, dtype: {nbatch['observation'].dtype}")
    print(f"proprio shape: {nbatch['proprio'].shape}, dtype: {nbatch['proprio'].dtype}")
    
    try:
        # Get action from wrapper
        target_q = wrapper(nbatch)
        
        # Print output shapes and dtypes
        print("\nOutput shapes and dtypes:")
        print(f"target_q shape: {target_q.shape}, dtype: {target_q.dtype}")
        
        # Verify output dimensions
        expected_shape = (len(wrapper.policy._modality_config["action"].delta_indices), config.num_joints + config.num_grippers)
        assert target_q.shape == expected_shape, f"Expected shape {expected_shape}, got {target_q.shape}"
        print("\nTest passed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    args = tyro.cli(TestConfig)
    main(args)