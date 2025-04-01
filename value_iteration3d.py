import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Function to visualize value function in 3D
def print_3d_value_function(value_function):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid coordinates
    x = np.arange(value_function.shape[1])
    y = np.arange(value_function.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, value_function, cmap=cm.viridis, 
                          linewidth=0, antialiased=False, alpha=0.8)
    
    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Customize the view angle
    ax.view_init(elev=30, azim=45)
    
    # Add labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Value')
    ax.set_title('3D Value Function Visualization')
    
    plt.tight_layout()
    plt.show()

# Original 2D visualization function (unchanged)
def print_value_and_policy(value_function, policy=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgray', 'green'])
    im = ax.imshow(value_function, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value Function')
    ax.set_xticks(np.arange(len(value_function)))
    ax.set_yticks(np.arange(len(value_function)))
    ax.set_xticklabels(np.arange(0, len(value_function)))
    ax.set_yticklabels(np.arange(0, len(value_function)))
    ax.xaxis.tick_top()
    ax.set_xlabel('State')
    ax.set_ylabel('State')

    for i in range(len(value_function)):
        for j in range(len(value_function[0])):
            ax.text(j, i, f'{value_function[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

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

# Gridworld setup (unchanged)
grid_width = 8
grid_height = 8
grid_size = grid_width * grid_height
theta = 0.0001
gamma = 0.9

# Initialize rewards (unchanged)
rewards = np.zeros((grid_width, grid_height))
rewards[0, :] = 5  # Top row reward of 5
rewards[-1, :] = -1  # Bottom row penalty of -1

# Possible actions (unchanged)
actions = [0, 1, 2, 3]
action_names = ["Up", "Down", "Right", "Left"]

# Value iteration function (unchanged)
def value_iteration(rewards, gamma, theta):
    value_function = np.zeros((grid_width, grid_height))
    policy = np.zeros((grid_width, grid_height), dtype=int)
    
    while True:
        delta = 0
        for i in range(grid_width):
            for j in range(grid_height):
                if (i == 0 or i == grid_width - 1):
                    continue
                
                v = value_function[i, j]
                new_values = []
                
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
                delta = max(delta, abs(v - value_function[i, j]))

        if delta < theta:
            break

    return value_function, policy

# Run value iteration (unchanged)
value_function, policy = value_iteration(rewards, gamma, theta)

# Display results - both 2D and 3D
print_value_and_policy(value_function)
print_value_and_policy(value_function, policy)
print_3d_value_function(value_function)