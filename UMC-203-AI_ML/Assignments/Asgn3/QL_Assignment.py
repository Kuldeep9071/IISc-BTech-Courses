#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import time
import pandas as pd
import pickle
import os
import json
from IPython.display import display, clear_output
import ipywidgets as widgets
import gymnasium as gym
from gymnasium import spaces


with open("themes.json", "r", encoding="utf-8") as f:
    THEMES = json.load(f)

# ========== CONFIG ==========
maze_size = (20, 20)
participant_id = 23684  ## SR.No
enable_enemy = False
enable_trap_boost = False
save_path = f"{participant_id}_basic.pkl"

# Q-learning parameters
###################################
#      WRITE YOUR CODE BELOW      #
num_actions = 4
gamma = 0.95                # between 0 - 1
alpha = 0.4                # between 0 - 1
epsilon = 1.0              # between 0 - 1
epsilon_decay = 0.99        # between 0.1 - 1
min_epsilon = 0.05
num_episodes = 5000    
max_steps = 50  
###################################    

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

# ========== REWARDS ==========
###################################
#      WRITE YOUR CODE BELOW      #
#      WRITE YOUR CODE BELOW      #
REWARD_GOAL = 10000    # Reward for reaching goal.
REWARD_TRAP = -500      # Trap cell.
REWARD_OBSTACLE = -100     # Obstacle cell.
REWARD_REVISIT = -200      # Revisiting same cell.
REWARD_ENEMY = -2000     # Getting caught by enemy.
REWARD_STEP = -5    # Per-step time penalty.
REWARD_BOOST = 200       # Boost cell.
###################################
# =============================


# In[ ]:


# Environment
class MazeGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps):
        super().__init__()
        """
        initialize the maze_size, participant_id, enable_enemy, enable_trap_boost and max_steps variables
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        self.maze_size = maze_size
        self.participant_id = participant_id
        self.enable_enemy = enable_enemy
        self.enable_trap_boost = enable_trap_boost
        self.max_steps = max_steps
        ###################################

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(maze_size[0]),
            spaces.Discrete(maze_size[1])
        ))

        """
        generate  self.maze using the _generate_obstacles method
        make self.start as the top left cell of the maze and self.goal as the bottom right
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        self.maze = self._generate_obstacles()
        self.start = (0, 0)
        self.goal = (self.maze_size[0]-1, self.maze_size[1]-1)
        ###################################

        if self.enable_trap_boost:
            self.trap_cells, self.boost_cells = self._generate_traps_and_boosts(self.maze)
        else:
            self.trap_cells, self.boost_cells = ([], [])

        self.enemy_cells = []
        self.current_step = 0
        self.agent_pos = None

        self.reset()

    def _generate_obstacles(self):
        """
        generates the maze with random obstacles based on the SR.No.
        """
        np.random.seed(self.participant_id)
        maze = np.zeros(self.maze_size, dtype=int)
        mask = np.ones(self.maze_size, dtype=bool)
        safe_cells = [
            (0, 0), (0, 1), (1, 0),
            (self.maze_size[0]-1, self.maze_size[1]-1), (self.maze_size[0]-2, self.maze_size[1]-1),
            (self.maze_size[0]-1, self.maze_size[1]-2)
        ]
        for row, col in safe_cells:
            mask[row, col] = False
        maze[mask] = np.random.choice([0, 1], size=mask.sum(), p=[0.9, 0.1])
        return maze

    def _generate_traps_and_boosts(self, maze):
        """
        generates special cells, traps and boosts. While training our agent,
        we want to pass thru more number of boost cells and avoid trap cells 
        """
        if not self.enable_trap_boost:
            return [], []
        exclusions = {self.start, self.goal}
        empty_cells = list(zip(*np.where(maze == 0)))
        valid_cells = [cell for cell in empty_cells if cell not in exclusions]
        num_traps = self.maze_size[0] * 2
        num_boosts = self.maze_size[0] * 2
        random.seed(self.participant_id)
        trap_cells = random.sample(valid_cells, num_traps)
        trap_cells_ = trap_cells
        remaining_cells = [cell for cell in valid_cells if cell not in trap_cells]
        boost_cells = random.sample(remaining_cells, num_boosts)
        boost_cells_ = boost_cells
        return trap_cells, boost_cells

    def move_enemy(self, enemy_pos):
        possible_moves = []
        for dx, dy in actions:
            new_pos = (enemy_pos[0] + dx, enemy_pos[1] + dy)
            if (0 <= new_pos[0] < self.maze_size[0] and
                0 <= new_pos[1] < self.maze_size[1] and
                self.maze[new_pos] != 1):
                possible_moves.append(new_pos)
        return random.choice(possible_moves) if possible_moves else enemy_pos

    def update_enemies(self):
        if self.enable_enemy:
            self.enemy_cells = [self.move_enemy(enemy) for enemy in self.enemy_cells]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        empty_cells = list(zip(*np.where(self.maze == 0)))
        self.start = (0, 0)
        self.goal = (self.maze_size[0]-1, self.maze_size[1]-1)

        for pos in (self.start, self.goal):
            if pos in self.trap_cells:
                self.trap_cells.remove(pos)
            if pos in self.boost_cells:
                self.boost_cells.remove(pos)

        if self.enable_enemy:
            enemy_candidates = [cell for cell in empty_cells if cell not in {self.start, self.goal}]
            num_enemies = max(1, int((self.maze_size[0] * self.maze_size[1]) / 100))
            self.enemy_cells = random.sample(enemy_candidates, min(num_enemies, len(enemy_candidates)))
        else:
            self.enemy_cells = []

        self.current_step = 0
        self.agent_pos = self.start
        self.visited = set()


        return self.agent_pos, {}

    def get_reward(self, state):
        if state == self.goal:
            return REWARD_GOAL
        elif state in self.trap_cells:
            return REWARD_TRAP
        elif state in self.boost_cells:
            return REWARD_BOOST
        elif self.maze[state] == 1:
            return REWARD_OBSTACLE
        else:
            return REWARD_STEP

    def take_action(self, state, action):
        attempted_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        if (0 <= attempted_state[0] < self.maze_size[0] and
            0 <= attempted_state[1] < self.maze_size[1] and
            self.maze[attempted_state] != 1):
            return attempted_state, False
        else:
            return state, True

    def step(self, action):
        self.current_step += 1
        next_state, wall_collision = self.take_action(self.agent_pos, action)
        if wall_collision:
            reward = REWARD_OBSTACLE
            next_state = self.agent_pos
        else:
            if self.enable_enemy:
                self.update_enemies()
            if self.enable_enemy and next_state in self.enemy_cells:
                reward = REWARD_ENEMY
                done = True
                truncated = True
                info = {'terminated_by': 'enemy'}
                self.agent_pos = next_state
                return self.agent_pos, reward, done, truncated, info
            else:
                revisit_penalty = REWARD_REVISIT if next_state in self.visited else 0
                self.visited.add(next_state)
                reward = self.get_reward(next_state) + revisit_penalty
        self.agent_pos = next_state

        if self.agent_pos == self.goal:
            done = True
            truncated = False
            info = {'completed_by': 'goal'}
        elif self.current_step >= self.max_steps:
            done = True
            truncated = True
            info = {'terminated_by': 'timeout'}
        else:
            done = False
            truncated = False
            info = {
                'current_step': self.current_step,
                'agent_position': self.agent_pos,
                'remaining_steps': self.max_steps - self.current_step
            }

        return self.agent_pos, reward, done, truncated, info

    def render(self, path=None, theme="racing"):
        icons = THEMES.get(theme, THEMES["racing"])
        clear_output(wait=True)
        grid = np.full(self.maze_size, icons["empty"])
        grid[self.maze == 1] = icons["obstacle"]
        for cell in self.trap_cells:
            grid[cell] = icons["trap"]
        for cell in self.boost_cells:
            grid[cell] = icons["boost"]
        grid[self.start] = icons["start"]
        grid[self.goal] = icons["goal"]
        if path is not None:
            for cell in path[1:-1]:
                if grid[cell] not in (icons["goal"], icons["obstacle"], icons["trap"], icons["boost"]):
                    grid[cell] = icons["path"]
        if self.agent_pos is not None:
            if grid[self.agent_pos] not in (icons["goal"], icons["obstacle"]):
                grid[self.agent_pos] = icons["agent"]
        if self.enable_enemy:
            for enemy in self.enemy_cells:
                grid[enemy] = icons["enemy"]
        df = pd.DataFrame(grid)
        print(df.to_string(index=False, header=False))

    def print_final_message(self, success, interrupted, caught, theme):
        msgs = THEMES.get(theme, THEMES["racing"]).get("final_messages", {})
        if interrupted:
            print(f"\n{msgs.get('Interrupted', '🛑 Interrupted.')}")
        elif caught:
            print(f"\n{msgs.get('Defeat', '🚓 Caught by enemy.')}")
        elif success:
            print(f"\n{msgs.get('Triumph', '🏁 Success.')}")
        else:
            print(f"\n{msgs.get('TimeOut', '⛽ Time Out.')}")


# In[ ]:


# Agent
class QLearningAgent:
    def __init__(self, maze_size, num_actions, alpha=0.1, gamma=0.99):
        """
        initialize self.num_actions, self.alpha, self.gamma
        initialize self.q_table based on number of states and number of actions
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        # Initialize Q-table with zeros for all state-action pairs
        self.q_table = np.zeros((maze_size[0], maze_size[1], num_actions))
        ###################################


    def choose_action(self, env, state, epsilon):
        """
        returns an integer between [0,3]

        epsilon is a parameter between 0 and 1.
        It is the probability with which we choose an exploratory action (random action)
        Eg: ---
        If epsilon = 0.25, probability of choosing action from q_table = 0.75
                           probability of choosing random action = 0.25
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation: choose best action from Q-table
            return np.argmax(self.q_table[state[0], state[1]])
        ###################################


    def update(self, state, action, reward, next_state):
        """
        Use the Q-learning update equation to update the Q-Table
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        # Get the current Q-value for this state-action pair
        current_q = self.q_table[state[0], state[1], action]
        
        # Get the maximum Q-value for the next state
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        
        # Q-learning update equation: Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update the Q-table
        self.q_table[state[0], state[1], action] = new_q
        ###################################


# In[ ]:


# Training loop
env = MazeGymEnv(maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps)
agent = QLearningAgent(maze_size, num_actions, alpha=alpha, gamma=gamma)

start_episode = 0
best_reward = -np.inf
best_q_table = None

if os.path.exists(save_path):
    print("Checkpoint found. Loading...")
    with open(save_path, 'rb') as f:
        checkpoint = pickle.load(f)
        agent.q_table = checkpoint['q_table']
        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        best_q_table = checkpoint.get('best_q_table', agent.q_table.copy())
        best_reward = checkpoint.get('best_reward', -np.inf)
        best_step_counter = checkpoint.get('best_step_counter', 0)
    print(f"Resuming from episode {start_episode} with epsilon {epsilon:.4f}, best reward {best_reward:.2f} and best step {best_step_counter}")
else:
    epsilon = 1.0

try:
    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        done = False
        visited_states = set()
        episode_reward = 0
        step_counter = 0

        while not done and step_counter < max_steps:
            action = agent.choose_action(env, state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            if next_state in visited_states:
                reward += REWARD_REVISIT
            visited_states.add(next_state)

            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            step_counter += 1

            if state == env.goal:
                done = True

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_q_table = agent.q_table.copy()
            best_step_counter = step_counter
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'q_table': agent.q_table,
                    'episode': episode,
                    'epsilon': epsilon,
                    'best_q_table': best_q_table,
                    'best_reward': best_reward,
                    'best_step_counter': best_step_counter
                }, f)
            print(f"New best at episode {episode}: {step_counter} steps and Reward {best_reward:.2f}")

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} - Epsilon: {epsilon:.4f} - Total Steps: {step_counter} - Episode Reward: {episode_reward:.2f} - Best Reward: {best_reward:.2f}")

except KeyboardInterrupt:
    print("\nTraining interrupted.")
    print(f"Interrupted at episode {episode} with epsilon: {epsilon:.4f}, Total Steps: {step_counter}, Episode Reward: {episode_reward:.2f}")
else:
    print(f"\nTraining completed. Total episodes: {episode}")


# In[ ]:


def test_agent(env, agent, animated, delay, theme):

    obs, _ = env.reset()
    state = obs
    path = [state]
    visited_states = set()
    total_reward = 0
    reward_breakdown = {
        'goal':     {'count': 0, 'reward': 0.0},
        'trap':     {'count': 0, 'reward': 0.0},
        'boost':    {'count': 0, 'reward': 0.0},
        'obstacle': {'count': 0, 'reward': 0.0},
        'step':     {'count': 0, 'reward': 0.0},
        'revisit':  {'count': 0, 'reward': 0.0}
    }
    caught_by_enemy = False
    success = False
    interrupted = False

    try:
        for step in range(env.max_steps):
            visited_states.add(state)

            action = agent.choose_action(env, state, epsilon=0.0)
            next_state, reward, done, truncated, info = env.step(action)

            if info.get('terminated_by') == 'enemy':
                caught_by_enemy = True
                reward_breakdown.setdefault('enemy', {'count': 0, 'reward': 0.0})
                reward_breakdown['enemy']['count'] += 1
                reward_breakdown['enemy']['reward'] += reward
                total_reward += reward
                path.append(next_state)
                break
            else:
                if (next_state == state) and (reward == REWARD_OBSTACLE):
                    reward_breakdown['obstacle']['count'] += 1
                    reward_breakdown['obstacle']['reward'] += REWARD_OBSTACLE
                elif next_state == env.goal:
                    reward_breakdown['goal']['count'] += 1
                    reward_breakdown['goal']['reward'] += REWARD_GOAL
                elif next_state in env.trap_cells:
                    reward_breakdown['trap']['count'] += 1
                    reward_breakdown['trap']['reward'] += REWARD_TRAP
                elif next_state in env.boost_cells:
                    reward_breakdown['boost']['count'] += 1
                    reward_breakdown['boost']['reward'] += REWARD_BOOST
                elif next_state in visited_states:
                    reward += REWARD_REVISIT
                    reward_breakdown['revisit']['count'] += 1
                    reward_breakdown['revisit']['reward'] += REWARD_REVISIT
                reward_breakdown['step']['count'] += 1
                reward_breakdown['step']['reward'] += REWARD_STEP

            total_reward += reward
            state = next_state
            path.append(state)

            if animated:
                env.render(path, theme)
                print(f"\nTotal Allowed Steps: {env.max_steps}")
                print(f"Current Reward: {total_reward:.2f}")
                print("Live Reward Breakdown:")
                df = pd.DataFrame.from_dict(reward_breakdown, orient='index')
                print(df)
                time.sleep(delay)

            if done or truncated:
                break

    except KeyboardInterrupt:
        interrupted = True

    if state == env.goal:
        success = True

    return path, total_reward, reward_breakdown, success, interrupted, caught_by_enemy


# In[ ]:


env = MazeGymEnv(maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps)
agent = QLearningAgent(maze_size, num_actions)

if os.path.exists(save_path):
    print("Checkpoint found. Loading best Q-table for testing...")
    with open(save_path, 'rb') as f:
        checkpoint = pickle.load(f)
        best_q_table = checkpoint.get('best_q_table', checkpoint['q_table'])
        agent.q_table = best_q_table
    print("Best Q-table loaded successfully.")
else:
    print("No checkpoint found")


# In[ ]:


# Run test.

theme = random.choice(list(THEMES.keys()))
plot_delay = 0.1  # Adjust delay as needed

path, total_reward, reward_breakdown, success, interrupted, caught_by_enemy = test_agent(env, agent, animated=True, delay=plot_delay, theme=theme)

env.render(path, theme=theme)
env.print_final_message(success, interrupted, caught=caught_by_enemy, theme=theme)

reward_df = pd.DataFrame.from_dict(reward_breakdown, orient='index')
reward_df.index = reward_df.index.str.title()
reward_df = reward_df.rename(columns={'count': 'Count', 'reward': 'Reward'})
total_row = pd.DataFrame({
    'Count': [''],
    'Reward': [reward_df['Reward'].sum()]
}, index=['Total'])
reward_df = pd.concat([reward_df, total_row])

print(reward_df)
print(f"\nTotal Allowed Steps: {max_steps}")


# In[ ]:


# Additional code for the assignment - Train with traps and boosts enabled
enable_trap_boost = True
save_path = f"{participant_id}_trap_boost.pkl"

env = MazeGymEnv(maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps)
agent = QLearningAgent(maze_size, num_actions, alpha=alpha, gamma=gamma)

start_episode = 0
best_reward = -np.inf
best_q_table = None
epsilon = 1.0

try:
    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        done = False
        visited_states = set()
        episode_reward = 0
        step_counter = 0

        while not done and step_counter < max_steps:
            action = agent.choose_action(env, state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            if next_state in visited_states:
                reward += REWARD_REVISIT
            visited_states.add(next_state)

            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            step_counter += 1

            if state == env.goal:
                done = True

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_q_table = agent.q_table.copy()
            best_step_counter = step_counter
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'q_table': agent.q_table,
                    'episode': episode,
                    'epsilon': epsilon,
                    'best_q_table': best_q_table,
                    'best_reward': best_reward,
                    'best_step_counter': best_step_counter
                }, f)
            print(f"New best at episode {episode}: {step_counter} steps and Reward {best_reward:.2f}")

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes} - Epsilon: {epsilon:.4f} - Total Steps: {step_counter} - Episode Reward: {episode_reward:.2f} - Best Reward: {best_reward:.2f}")

except KeyboardInterrupt:
    print("\nTraining interrupted.")
    print(f"Interrupted at episode {episode} with epsilon: {epsilon:.4f}, Total Steps: {step_counter}, Episode Reward: {episode_reward:.2f}")
else:
    print(f"\nTraining completed. Total episodes: {episode}")


# In[ ]:


# Test agent with traps and boosts enabled
env = MazeGymEnv(maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps)
agent = QLearningAgent(maze_size, num_actions)

if os.path.exists(save_path):
    print("Checkpoint found. Loading best Q-table for testing...")
    with open(save_path, 'rb') as f:
        checkpoint = pickle.load(f)
        best_q_table = checkpoint.get('best_q_table', checkpoint['q_table'])
        agent.q_table = best_q_table
    print("Best Q-table loaded successfully.")
else:
    print("No checkpoint found")

theme = random.choice(list(THEMES.keys()))
plot_delay = 0.1  # Adjust delay as needed

path, total_reward, reward_breakdown, success, interrupted, caught_by_enemy = test_agent(env, agent, animated=True, delay=plot_delay, theme=theme)

env.render(path, theme=theme)
env.print_final_message(success, interrupted, caught=caught_by_enemy, theme=theme)

reward_df = pd.DataFrame.from_dict(reward_breakdown, orient='index')
reward_df.index = reward_df.index.str.title()
reward_df = reward_df.rename(columns={'count': 'Count', 'reward': 'Reward'})
total_row = pd.DataFrame({
    'Count': [''],
    'Reward': [reward_df['Reward'].sum()]
}, index=['Total'])
reward_df = pd.concat([reward_df, total_row])

print(reward_df)
print(f"\nTotal Allowed Steps: {max_steps}")


# In[ ]:


# Manual Q-value update calculation for the first 5 steps of a new episode
def manual_q_update_calculation():
    print("Manual Q-value update calculation for the first 5 steps of a new episode")
    print("=====================================================================")
    
    # Reset environment and agent
    env = MazeGymEnv(maze_size, participant_id, False, False, max_steps)
    agent = QLearningAgent(maze_size, num_actions, alpha=alpha, gamma=gamma)
    
    if os.path.exists(f"{participant_id}_basic.pkl"):
        with open(f"{participant_id}_basic.pkl", 'rb') as f:
            checkpoint = pickle.load(f)
            agent.q_table = checkpoint.get('best_q_table', checkpoint['q_table'])
    
    # Start a new episode
    state, _ = env.reset()
    visited_states = set()
    
    print(f"Starting state: {state}")
    
    for step in range(5):
        print(f"\nStep {step+1}:")
        print(f"Current state: {state}")
        
        # Current Q-values
        current_q_values = agent.q_table[state[0], state[1]]
        print(f"Current Q-values for state {state}: {current_q_values}")
        
        # Choose action with epsilon=0 (greedy)
        action = agent.choose_action(env, state, epsilon=0.0)
        print(f"Chosen action: {action} ({['Up', 'Down', 'Left', 'Right'][action]})")
        
        # Take step and get reward
        next_state, reward, done, truncated, info = env.step(action)
        
        if next_state in visited_states:
            reward += REWARD_REVISIT
            print(f"Revisit penalty applied: {REWARD_REVISIT}")
        visited_states.add(next_state)
        
        print(f"Next state: {next_state}")
        print(f"Reward received: {reward}")
        
        # Calculate expected Q-value update
        current_q = agent.q_table[state[0], state[1], action]
        max_next_q = np.max(agent.q_table[next_state[0], next_state[1]])
        
        # Manual calculation
        expected_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        
        print(f"Q-value update calculation:")
        print(f"Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]")
        print(f"Q({state}, {action}) = {current_q:.4f} + {alpha} * [{reward} + {gamma} * {max_next_q:.4f} - {current_q:.4f}]")
        print(f"Q({state}, {action}) = {current_q:.4f} + {alpha} * [{reward + gamma * max_next_q - current_q:.4f}]")
        print(f"Q({state}, {action}) = {current_q:.4f} + {alpha * (reward + gamma * max_next_q - current_q):.4f}")
        print(f"Q({state}, {action}) = {expected_q:.4f}")
        
        # Update Q-value
        agent.update(state, action, reward, next_state)
        
        # Verify updated Q-value
        updated_q = agent.q_table[state[0], state[1], action]
        print(f"Updated Q-value in agent: {updated_q:.4f}")
        
        if abs(expected_q - updated_q) < 1e-10:
            print("✓ Manual calculation matches the agent's update")
        else:
            print("✗ Discrepancy between manual calculation and agent's update")
        
        # Move to next state
        state = next_state
        
        if done:
            print("\nGoal reached!")
            break

# Run the manual calculation
manual_q_update_calculation()


# Run alternative reward configurations
def train_with_alternative_rewards():
    # Configuration 1: Higher step penalty, lower goal reward
    global REWARD_GOAL, REWARD_STEP
    old_goal = REWARD_GOAL
    old_step = REWARD_STEP
    
    REWARD_GOAL = 50    # Reduced goal reward
    REWARD_STEP = -3    # Increased step penalty
    
    print("\n===============================================")
    print("Training with Alternative Reward Configuration 1:")
    print("REWARD_GOAL =", REWARD_GOAL)
    print("REWARD_STEP =", REWARD_STEP)
    
    env = MazeGymEnv(maze_size, participant_id, False, False, max_steps)
    agent = QLearningAgent(maze_size, num_actions, alpha=alpha, gamma=gamma)
    save_path = f"{participant_id}_alt_rewards1.pkl"