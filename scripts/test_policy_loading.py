from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

# get the data config
data_config = DATA_CONFIG_MAP["yumi"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# load the model 
# 3. Load pre-trained model
policy_path = "/mnt/8tb-drive/groot_ckpt/dpgs_sim_coffee/checkpoint-10000"
policy = Gr00tPolicy(
    model_path=policy_path,
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device="cuda"
)

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="/mnt/8tb-drive/lerobot_conversion/lerobot/mlfu7/dpgs_conversion_video_groot",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, # the embodiment to use
)

# get the first batch of data from the dataset
batch = dataset[0]

# get the action from the policy
action = policy.get_action(batch)

import pdb; pdb.set_trace()