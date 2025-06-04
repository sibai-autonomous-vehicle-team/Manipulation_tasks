import torch
from pathlib import Path
import re
from torch.utils.data import Dataset
# 1. Load the global episode-wise tensors (shape: [num_episodes, episode_len, dim])
actions     = torch.load("/Users/maxwellastafyev/Desktop/Research_project/Manipulation_tasks/Dino_MW/MS_data/actions.pth")
states      = torch.load("/Users/maxwellastafyev/Desktop/Research_project/Manipulation_tasks/Dino_MW/MS_data/states.pth")
seq_lengths = torch.load("/Users/maxwellastafyev/Desktop/Research_project/Manipulation_tasks/Dino_MW/MS_data/seq_length.pth")  # e.g., tensor([100, 100, ...])
# 2. Gather & sort the observation files
obs_dir = Path("/Users/maxwellastafyev/Desktop/Research_project/Manipulation_tasks/Dino_MW/MS_data/obses")
def episode_index(fn):
    return int(re.search(r"\d+", fn.name).group())
obs_paths = sorted(obs_dir.glob("episode_*.pth*"), key=episode_index)
assert len(obs_paths) == len(seq_lengths) == len(states), "Mismatch in number of episodes"
# 3. Define the EpisodeDataset class
class EpisodeDataset(Dataset):
    def __init__(self, states, actions, obs_paths):
        self.states = states        # shape: [N, T, state_dim]
        self.actions = actions      # shape: [N, T, action_dim]
        self.obs_paths = obs_paths  # list of Path objects
    def __len__(self):
        return len(self.obs_paths)
    def __getitem__(self, idx):
        ep_states = self.states[idx]      # shape: [T, state_dim]
        ep_actions = self.actions[idx]    # shape: [T, action_dim]
        ep_obs = torch.load(self.obs_paths[idx])  # shape: [T, H, W, C] or similar
        return {
            "states": ep_states,
            "actions": ep_actions,
            "obs": ep_obs
        }
# 4. Instantiate the dataset and test
dataset = EpisodeDataset(states, actions, obs_paths)
print(f"{len(dataset)} episodes, first episode has:")
sample = dataset[0]
print(" • states:",  sample["states"].shape)
print(" • actions:", sample["actions"].shape)
print(" • obs:",     type(sample["obs"]), getattr(sample["obs"], "shape", None))