import numpy as np
import matplotlib.pyplot as plt

# constants
G = 1.0  # <-- set G = 1 for easier units (or you can keep real G)

# masses
m1 = 4.88
m2 = 1.0

# initial positions
r1 = np.array([-0.5, 0.0])
r2 = np.array([0.5, 0.0])

# initial velocities
v1 = np.array([0.0, 0.512])
v2 = np.array([0.0, -2.5])

# time settings
dt = 0.001
steps = 10000

# record positions
r1_list = []
r2_list = []

for _ in range(steps):
    # distance vector and distance
    r = r2 - r1
    distance = np.linalg.norm(r)
    
    # force magnitude
    force_mag = G * m1 * m2 / distance**2
    
    # force direction
    force_dir = r / distance
    
    # force vectors
    F1 = force_mag * force_dir
    F2 = -F1
    
    # update velocities
    v1 += F1 / m1 * dt
    v2 += F2 / m2 * dt
    
    # update positions
    r1 += v1 * dt
    r2 += v2 * dt
    
    # save for plotting
    r1_list.append(r1.copy())
    r2_list.append(r2.copy())

# convert to array
r1_list = np.array(r1_list)
r2_list = np.array(r2_list)

# plot
plt.plot(r1_list[:,0], r1_list[:,1], label='Mass 1')
plt.plot(r2_list[:,0], r2_list[:,1], label='Mass 2')
plt.scatter(0, 0, color='black', label='Center of Mass')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
