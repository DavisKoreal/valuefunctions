import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to visualize value function and policy
def print_value_and_policy(value_function, policy=None):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(5, 5))

    # Create a continuous colormap from light gray to green
    cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgray', 'green'])

    # Plot the optimal value function with the custom colormap
    im = ax.imshow(value_function, cmap=cmap, interpolation='nearest')

    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value Function')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(value_function)))
    ax.set_yticks(np.arange(len(value_function)))
    ax.set_xticklabels(np.arange(0, len(value_function)))
    ax.set_yticklabels(np.arange(0, len(value_function)))
    ax.xaxis.tick_top()

    ax.set_xlabel('State')
    ax.set_ylabel('State')

    # Add text annotations (values in each cell with 2 decimal precision)
    for i in range(len(value_function)):
        for j in range(len(value_function[0])):
            ax.text(j, i, f'{value_function[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

    # Add arrows for the policy (if provided)
    if policy is not None:
        for i in range(len(policy)):
            for j in range(len(policy[0])):
                if policy[i, j] == 0:  # Up
                    ax.arrow(j, i, 0, -0.3, head_width=0.2, head_length=0.2, color='darkorange')
                elif policy[i, j] == 1:  # Down
                    ax.arrow(j, i, 0, 0.3, head_width=0.2, head_length=0.2, color='darkorange')
                elif policy[i, j] == 2:  # Right
                    ax.arrow(j, i, 0.3, 0, head_width=0.2, head_length=0.2, color='darkorange')
                elif policy[i, j] == 3:  # Left
                    ax.arrow(j, i, -0.3, 0, head_width=0.2, head_length=0.2, color='darkorange')

    plt.show()

# Gridworld setup
grid_width = 8
grid_height = 8
grid_size = grid_width * grid_height
theta = 0.0001  # Small threshold for stopping value iteration
gamma = 0.9  # Discount factor

# Initialize rewards
rewards = np.zeros((grid_width, grid_height))
rewards[0, :] = 5  # Top row reward of 5
rewards[-1, :] = -1  # Bottom row penalty of -1

# Possible actions: 0 = Up, 1 = Down, 2 = Right, 3 = Left
actions = [0, 1, 2, 3]
action_names = ["Up", "Down", "Right", "Left"]

# Value iteration function
def value_iteration(rewards, gamma, theta):
    value_function = np.zeros((grid_width, grid_height))  # Initialize value function
    policy = np.zeros((grid_width, grid_height), dtype=int)  # Initialize policy
    
    while True:
        delta = 0  # Difference between updates
        for i in range(grid_width):
            for j in range(grid_height):
                if (i == 0 or i == grid_width - 1):  # If in terminal row (reward)
                    continue
                
                v = value_function[i, j]  # Current value
                new_values = []
                
                # Calculate possible next states and update based on policy
                for action in actions:
                    if action == 0:  # Up
                        next_i, next_j = max(i - 1, 0), j
                    elif action == 1:  # Down
                        next_i, next_j = min(i + 1, grid_width - 1), j
                    elif action == 2:  # Right
                        next_i, next_j = i, min(j + 1, grid_height - 1)
                    else:  # Left
                        next_i, next_j = i, max(j - 1, 0)

                    new_value = rewards[next_i, next_j] + gamma * value_function[next_i, next_j]
                    new_values.append(new_value)

                value_function[i, j] = max(new_values)
                best_action = np.argmax(new_values)
                policy[i, j] = best_action
                delta = max(delta, abs(v - value_function[i, j]))  # Maximum change

        if delta < theta:  # Check if the value function has converged
            break

    return value_function, policy

# Run value iteration
value_function, policy = value_iteration(rewards, gamma, theta)

# Display results
print_value_and_policy(value_function)
print_value_and_policy(value_function, policy)
