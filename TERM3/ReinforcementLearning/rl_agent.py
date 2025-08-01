#rl_agent.py
#import gym
import subprocess
import sys
try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
import torch
import torch.nn as nn
import numpy as np
import os



# Get the current directory of submission.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the model file
model_path = os.path.join(CURRENT_DIR, "dqn_inventory.pth")

# # Load model content
# with open(model_path, 'r') as f:
#     model_data = f.read()

# # Optionally, process the model data
# print("Loaded model data:", model_data)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class RLAgent:
    def __init__(self):
        pass

    def flatten_state(self, state):
      if isinstance(state, dict):
          return np.concatenate([np.array(v, dtype=np.float32) for v in state.values()])
      return np.array(state, dtype=np.float32)

    def run_policy(self,state):
        state = self.flatten_state(state)
        STATE_SIZE = len(state)  # Example: adjust to your actual flattened state size
        ACTION_SIZE = 11 ** 3  # 3 products, 11 discrete actions each

        policy_net = QNetwork(STATE_SIZE, ACTION_SIZE)
        policy_net.load_state_dict(torch.load(model_path,map_location='cpu'))
        policy_net.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action_idx = torch.argmax(q_values).item()

        # Convert flat index → orders for 3 products
        orders = np.unravel_index(action_idx, (11, 11, 11))
        return [o * 10 for o in orders]  # since action space is {0,10,...,100}

