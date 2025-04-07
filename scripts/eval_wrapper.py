import torch 
import os
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import numpy as np

class DiffusionWrapper():
    def __init__(
        self, 
        model_ckpt_folder : str, 
        ckpt_id : int, 
        device : str,
        text_prompt : str = "put the white cup on the coffee machine",
    ) -> None:
        """
        Args:
            model_ckpt_folder: str, path to the model checkpoint folder
            ckpt_id: int, checkpoint id
            device: str, device to run the model on
            text_prompt: str, text prompt to use for the model
        Example:
        model_ckpt_folder = "/mnt/8tb-drive/groot_ckpt/dpgs_sim_coffee"
        ckpt_id = 10000
        device = "cuda"
        """

        # get the data config
        data_config = DATA_CONFIG_MAP["yumi"]

        # get the modality configs and transforms
        modality_config = data_config.modality_config()
        transforms = data_config.transform()
        self.device = device

        # load the model 
        # 3. Load pre-trained model
        model_path = os.path.join(model_ckpt_folder, f"checkpoint-{ckpt_id}")
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=modality_config,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.GR1,
            device=self.device
        )
        self.text_prompt = text_prompt

    def __call__(self, nbatch):
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images 
        """
        Model input expected: 
            video.exterior_image_1_left: (1, 270, 480, 3) np.uint8 (center image) 
            video.exterior_image_2_left: (1, 270, 480, 3) np.uint8 (side image)
            state.yumi_joints: (1, 14) np.float64
            state.yumi_grippers: (1, 2) np.float64
            annotation.human.action.task_description: List["text"], i.e. "put the white cup on the coffee machine"
        
        Model will output:
            "action.yumi_grippers": (1, 2) np.float64
            "action.yumi_delta_joints": (1, 14) np.float64
        """
        # update nbatch observation (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C)
        nbatch["observation"] = nbatch["observation"][:, -1]
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # we use the last two action dimensions for the grippers
        joint_positions = nbatch["proprio"][:, -1, :-2]
        gripper_positions = nbatch["proprio"][:, -1, -2:]
        batch = {
            "video.exterior_image_1_left": nbatch["observation"][:, 0],
            "video.exterior_image_2_left": nbatch["observation"][:, 1],
            "state.yumi_joints": joint_positions,
            "state.yumi_grippers": gripper_positions,
            "annotation.human.action.task_description": [self.text_prompt],
        }
        action = self.policy.get_action(batch)

        # convert to absolute action 
        target_joint_positions = action["action.yumi_delta_joints"] + joint_positions
        
        # append gripper command to the joint positions
        target_q = np.concatenate([target_joint_positions, action["action.yumi_grippers"]], axis=-1)
        return target_q