# https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472

from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay
import time

import random


class Environment:
    def __init__(self, grid_size, render_on=False):
        self.grid_size = grid_size
        self.render_on = render_on
        self.grid = []
        self.agent_location = None
        self.goal_location = None

    def reset(self):
        # Initialize the empty grid as a 2d array of 0s
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Add the agent and the goal to the grid
        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()

        # Render the initial grid
        if self.render_on:
            self.render()

        # Return the initial state
        return self.get_state()

    def add_agent(self):
        # Choose a random location
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        
        # Agent is represented by a 1
        self.grid[location[0]][location[1]] = 1
        return location

    def add_goal(self):
        # Choose a random location
        location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

        # Get a random location until it is not occupied
        while self.grid[location[0]][location[1]] == 1:
            location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        
        # Goal is represented by a -1
        self.grid[location[0]][location[1]] = -1

        return location

    def move_agent(self, action):
        # Map agent action to the correct movement
        moves = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        
        previous_location = self.agent_location
        
        # Determine the new location after applying the action
        move = moves[action]
        new_location = (previous_location[0] + move[0], previous_location[1] + move[1])
        
        done = False  # The episode is not done by default
        reward = 0   # Initialize reward
        
        # Check for a valid move
        if self.is_valid_location(new_location):
            # Remove agent from old location
            self.grid[previous_location[0]][previous_location[1]] = 0
            
            # Add agent to new location
            self.grid[new_location[0]][new_location[1]] = 1
            
            # Update agent's location
            self.agent_location = new_location
            
            # Check if the new location is the reward location
            if self.agent_location == self.goal_location:
                # Reward for getting the goal
                reward = 100
                
                # Episode is complete
                done = True
            else:
                # Calculate the distance before the move
                previous_distance = np.abs(self.goal_location[0] - previous_location[0]) + \
                                    np.abs(self.goal_location[1] - previous_location[1])
                        
                # Calculate the distance after the move
                new_distance = np.abs(self.goal_location[0] - new_location[0]) + \
                               np.abs(self.goal_location[1] - new_location[1])
                
                # If new_location is closer to the goal, reward = 0.9, if further, reward = -1.1
                reward = (previous_distance - new_distance) - 0.1
        else:
            # Slightly larger punishment for an invalid move
            reward = -3

        return reward, done
            
    def is_valid_location(self, location):
        # Check if the location is within the boundaries of the grid
        if (0 <= location[0] < self.grid_size) and (0 <= location[1] < self.grid_size):
            return True
        else:
            return False
        
    def get_state(self):
        # Flatten the grid from 2d to 1d
        state = self.grid.flatten()
        return state

    def render(self):
        # Convert to a list of ints to improve formatting
        grid = self.grid.astype(int).tolist()
        for row in grid:
            print(row)
        print('') # To add some space between renders for each step

    def step(self, action):
        # Apply the action to the environment, record the observations
        reward, done = self.move_agent(action)
        next_state = self.get_state()

        # Render the grid at each step
        if self.render_on:
            self.render()

        return reward, next_state, done

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

class Agent:
    def __init__(self, grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma

    def build_model(self):
        # Create a sequential model with 3 layers
        model = Sequential([
            # Input layer expects a flattened grid, hence the input shape is grid_size squared
            Dense(128, activation='relu', input_shape=(self.grid_size**2,)),
            Dense(64, activation='relu'),
            # Output layer with 4 units for the possible actions (up, down, left, right)
            Dense(4, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse')

        return model
    
    def get_action(self, state):

        # rand() returns a random value between 0 and 1
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 4)
        else:
            # Add an extra dimension to the state to create a batch with one instance
            state = np.expand_dims(state, axis=0)

            # Use the model to predict the Q-values (action values) for the given state
            q_values = self.model.predict(state, verbose=0)

            # Select and return the action with the highest Q-value
            action = np.argmax(q_values[0]) # Take the action from the first (and only) entry
        
        # Decay the epsilon value to reduce the exploration over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action
    
    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        # Predict the Q-values (action values) for the given state batch
        current_q_values = self.model.predict(states, verbose=0)

        # Predict the Q-values for the next_state batch
        next_q_values = self.model.predict(next_states, verbose=0)

        # Initialize the target Q-values as the current Q-values
        target_q_values = current_q_values.copy()

        # Loop through each experience in the batch
        for i in range(len(experiences)):
            if dones[i]:
                # If the episode is done, there is no next Q-value
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # The updated Q-value is the reward plus the discounted max Q-value for the next state
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
    
    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)

import random
from collections import deque, namedtuple

class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        # Memory stores the experiences in a deque, so if capacity is exceeded it removes
        # the oldest item efficiently
        self.memory = deque(maxlen=capacity)

        # Batch size specifices the amount of experiences that will be sampled at once
        self.batch_size = batch_size

        # Experience is a namedtuple that stores the relevant information for training
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add_experience(self, state, action, reward, next_state, done):
        # Create a new experience and store it in memory
        experience = self.Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample_batch(self):
        # Batch will be a random sample of experiences from memory of size batch_size
        batch = random.sample(self.memory, self.batch_size)
        return batch

    def can_provide_sample(self):
        # Determines if the length of memory has exceeded batch_size
        return len(self.memory) >= self.batch_size


if __name__ == '__main__':

    grid_size = 5

    environment = Environment(grid_size=grid_size, render_on=True)
    agent = Agent(grid_size=grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01)
    # agent.load(f'models/model_{grid_size}.h5')

    experience_replay = ExperienceReplay(capacity=10000, batch_size=32)
    
    # Number of episodes to run before training stops
    episodes = 5000
    # Max number of steps in each episode
    max_steps = 200

    for episode in range(episodes):

        # Get the initial state of the environment and set done to False
        state = environment.reset()

        # Loop until the episode finishes
        for step in range(max_steps):
            print('Episode:', episode)
            print('Step:', step)
            print('Epsilon:', agent.epsilon)

            # Get the action choice from the agents policy
            action = agent.get_action(state)

            # Take a step in the environment and save the experience
            reward, next_state, done = environment.step(action)
            experience_replay.add_experience(state, action, reward, next_state, done)

            # If the experience replay has enough memory to provide a sample, train the agent
            if experience_replay.can_provide_sample():
                experiences = experience_replay.sample_batch()
                agent.learn(experiences)

            # Set the state to the next_state
            state = next_state
            
            if done:
                break
            
            # Optionally, pause for half a second to evaluate the model
            # time.sleep(0.5)

        agent.save(f'models/model_{grid_size}.h5')