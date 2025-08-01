import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class DQN(nn.Module):
    """Deep Q-Network for inventory control with updated state representation"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
from typing import List, Tuple, Optional
import numpy as np
 
class InventoryEnv:
    """
    Inventory management environment for 3 products with volume constraints, lead times, and stochastic or deterministic demand.
 
    This environment simulates:
    - Warehouse inventory evolution with lead-time-based ordering.
    - Daily customer demand and fulfillment.
    - Cost computation due to holding, ordering, and stockouts.
 
    Attributes:
        volume_capacity (float): Max warehouse volume capacity.
        initial_inventory (List[int]): Initial stock for each product.
        product_volumes (List[float]): Volume per unit of each product.
        holding_cost_per_volume (float): Cost per unit volume per day for storing inventory.
        stockout_costs (List[float]): Penalty per unit of unfulfilled demand for each product.
        ordering_costs (List[float]): Fixed cost per order placed for each product.
        discard_costs (List[float]): Cost per unit discarded due to over-capacity.
        lead_times (List[int]): Days before an order arrives for each product.
        simulation_days (int): Episode length in days.
        demand_sequences (Optional[List[List[int]]]): Predefined demand for evaluation.
        demand_lambda (List[float]): Poisson mean for training demand generation.
    """
 
    def __init__(
        self,
        volume_capacity: float = 1000.0,
        initial_inventory: List[int] = [100, 100, 100],
        product_volumes: List[float] = [2.0, 3.0, 1.5],
        holding_cost_per_volume: float = 5.0,  # Updated holding cost
        stockout_costs: List[float] = [400.0, 500.0, 300.0],
        ordering_costs: List[float] = [80.0, 200.0, 120.0],
        discard_costs: List[float] = [200.0, 250.0, 150.0],  # New discard penalties
        lead_times: List[int] = [3, 2, 1],
        simulation_days: int = 50,
        demand_sequences: Optional[List[List[int]]] = None,
        demand_lambda: List[float] = [30, 25, 35],
        seed: int = 42
    ):
        self.volume_capacity = volume_capacity
        self.initial_inventory = initial_inventory[:]
        self.product_volumes = product_volumes
        self.holding_cost_per_volume = holding_cost_per_volume
        self.stockout_costs = stockout_costs
        self.ordering_costs = ordering_costs
        self.discard_costs = discard_costs
        self.lead_times = lead_times
        self.simulation_days = simulation_days
        self.demand_sequences = demand_sequences
        self.demand_lambda = demand_lambda
        self.random_state = np.random.RandomState(seed)
 
        self.reset()
 
    def reset(self) -> List[int]: 
        """
        Reset environment to initial state for a new episode.
        Returns the initial observation state.
        """
        self.day = 0
        self.inventory = self.initial_inventory[:] # Resets current inventory to initial inventory
        self.pending_orders = [[] for _ in range(len(self.initial_inventory))]  # list of orders to be delivered (day_due, quantity)
        return self._get_state() # Returns initial state of the environment
 
    def step(self, action: List[int]) -> Tuple[List[int], float, bool, dict]:
        """
        Executes one simulation step.
 
        Args:
            action (List[int]): List of order quantities for each product. Each value must be in {0, 10, ..., 100}.
 
        Returns:
            state (List[int]): Updated state after taking the action.
            reward (float): Scaled negative cost for the step.
            done (bool): True if the episode is over.
            info (dict): Additional information (cost breakdown, demand, fulfillment).
        """
        assert all(a in range(0, 101, 10) for a in action), "Actions must be in {0, 10, ..., 100}" # Invalid actions are rejected
 
        # 1. Receive due orders and add them to current inventory
        for i in range(3):
            arrivals = [qty for due, qty in self.pending_orders[i] if due == self.day]
            self.inventory[i] += sum(arrivals)
            self.pending_orders[i] = [(due, qty) for due, qty in self.pending_orders[i] if due > self.day]
 
        # 2. Place new orders and add them to pending orders
        order_cost = 0
        for i in range(3):
            if action[i] > 0:
                order_cost += self.ordering_costs[i]
                self.pending_orders[i].append((self.day + self.lead_times[i], action[i]))
 
        # 3. Generate demand if not provided
        if self.demand_sequences:
            demand = self.demand_sequences[self.day]
        else:
            demand = [self.random_state.poisson(lam) for lam in self.demand_lambda]
 
        # 4. Enforce volume capacity and compute discards
        total_volume = sum(self.inventory[i] * self.product_volumes[i] for i in range(3))
        discarded = [0, 0, 0]
        if total_volume > self.volume_capacity:
            overflow = total_volume - self.volume_capacity
            # discard from highest-volume items first
            for i in sorted(range(3), key=lambda j: self.product_volumes[j], reverse=True):
                max_remove = int(overflow // self.product_volumes[i])
                remove_qty = min(max_remove, self.inventory[i])
                discarded[i] = remove_qty
                self.inventory[i] -= remove_qty
                overflow -= remove_qty * self.product_volumes[i]
                if overflow <= 0:
                    break
 
        # 5. Fulfill demand and compute stockouts
        fulfilled = [min(self.inventory[i], demand[i]) for i in range(3)]
        unfulfilled = [demand[i] - fulfilled[i] for i in range(3)]
        self.inventory = [self.inventory[i] - fulfilled[i] for i in range(3)]
 
        # 6. Compute costs and reward
        holding_cost = sum(self.inventory[i] * self.product_volumes[i] * self.holding_cost_per_volume for i in range(3))
        stockout_cost = sum(unfulfilled[i] * self.stockout_costs[i] for i in range(3))
        discard_cost = sum(discarded[i] * self.discard_costs[i] for i in range(3))
        total_cost = holding_cost + stockout_cost + order_cost + discard_cost
        reward = - total_cost / 100.0  # scaled for stability
 
        # 7. Update state
        self.day += 1
        done = self.day >= self.simulation_days # True if episode ends
        info = {
            "day": self.day,
            "inventory": self.inventory[:],
            "fulfilled": fulfilled,
            "unfulfilled": unfulfilled,
            "order_cost": order_cost,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "discard_cost": discard_cost,
            "total_cost": total_cost
        }
 
        return self._get_state(), reward, done, info
 
    def _get_state(self) -> List[int]:
        """
        Constructs the state vector including inventory levels and outstanding orders.
 
        Returns:
            List[int]: State representation with 7 variables 
        """
        outstanding_orders = [sum(qty for _, qty in self.pending_orders[i]) for i in range(3)]
        return self.inventory + outstanding_orders + [self.day] 


class InventoryAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter()
        self.steps = 0
        self.target_update_freq = 100
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action but with valid quantities (0, 10, 20, ..., 100)
            return [random.choice(range(0, 101, 10)) for _ in range(3)]
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        
        # Convert Q-values to action (3 products, each 0-100 in steps of 10)
        action = []
        for i in range(3):
            # Get Q-values for this product's actions (0-100 in steps of 10)
            product_q = q_values[0, i*11:(i+1)*11]
            action.append(torch.argmax(product_q).item() * 10)
        
        return action
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = np.array([t[1] for t in minibatch])
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Convert actions to indices for Q-value selection
        action_indices = (actions / 10).astype(int)
        
        # Current Q-values
        current_q = self.model(states)
        current_q_selected = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            for j in range(3):
                current_q_selected[i] += current_q[i, j*11 + action_indices[i,j]]
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q
        
        loss = self.criterion(current_q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Logging
        self.writer.add_scalar('Training/loss', loss.item(), self.steps)
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def train_agent(episodes=1000):
    env = InventoryEnv()
    state_size = 7  # inventory (3) + outstanding orders (3) + day (1)
    
    # Action space is 11 options (0-100 in steps of 10) for each of 3 products
    agent = InventoryAgent(state_size, 11*3)  # We'll handle the action space differently
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for t in range(50):  # 50 days per episode
            action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
            agent.replay()
        
        # Log episode reward
        agent.writer.add_scalar('Training/episode_reward', total_reward, e)
        
        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    agent.save("inventory_dqn_updated.pth")
    return agent

def evaluate_agent(agent, eval_episodes=10):
    env = InventoryEnv(demand_sequences=None)  # Use stochastic demand for evaluation
    total_costs = []
    
    for _ in range(eval_episodes):
        state = env.reset()
        episode_cost = 0
        
        for _ in range(50):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_cost += info['total_cost']
            state = next_state
            
            if done:
                break
                
        total_costs.append(episode_cost)
    
    avg_cost = sum(total_costs) / eval_episodes
    print(f"Average cost over {eval_episodes} evaluation episodes: {avg_cost}")
    return avg_cost

def create_submission_file(agent, filename="submission.py"):
    with open(filename, 'w') as f:
        f.write("""import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Global variables to maintain state across calls
_agent = None
_prev_state = None

def init_agent():
    global _agent
    state_size = 7  # inventory (3) + outstanding orders (3) + day (1)
    
    # Load the trained model
    _agent = DQN(state_size, 33)  # 11 options * 3 products
    checkpoint = torch.load('inventory_dqn_updated.pth', map_location='cpu')
    _agent.load_state_dict(checkpoint['model_state_dict'])
    _agent.eval()

def run_policy(state):
    global _agent, _prev_state
    
    if _agent is None:
        init_agent()
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = _agent(state_tensor)
    
    # Convert Q-values to action (3 products, each 0-100 in steps of 10)
    action = []
    for i in range(3):
        # Get Q-values for this product's actions (0-100 in steps of 10)
        product_q = q_values[0, i*11:(i+1)*11]
        action.append(torch.argmax(product_q).item() * 10)
    
    return action
""")
def save_model_to_text(agent, filename="model.txt"):
    """
    Save the model's state dictionary to a text file in a human-readable format.
    """
    with open(filename, 'w') as f:
        for key, value in agent.model.state_dict().items():
            f.write(f"{key}: {value.tolist()}\n")
    print(f"Model saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent(episodes=1000)
    
    # Evaluate the agent
    evaluate_agent(trained_agent)

    # Save the model to a text file
    save_model_to_text(trained_agent, filename="model.txt")
    
    # Create submission file
    create_submission_file(trained_agent)
    print("Submission file 'submission.py' created successfully!")