import numpy as np
import matplotlib.pyplot as plt
import random
from rrt import RRTPlanner3D
from energy import get_energy

def building_height(arr: np.ndarray) -> np.ndarray:
    X, Z, Y = arr.shape
    heights = np.arange(Z)[None, :, None]
    masked = np.where(arr, heights, -1)
    max_h = masked.max(axis=1)
    return max_h + 1



grid = np.load('building_case_0.npy')
sdf = np.load('sdf_case_0.npy')
wind = np.load('mean_case_0.npy')[:, :, :, [2,1,0]]
# wind = wind * 3

print(grid.shape)
print(wind.shape)

# plt.imshow(grid[:, 5, :])
# plt.show()


start = [20, 1, 280]
goal = [280, 1, 20]

n_trials = 20

seed_base = 42
np.random.seed(seed_base)

best_path = None
best_loss = np.inf

planner = RRTPlanner3D(grid,sdf, step_size=15, min_step_size=5,sdf_scale=0.4,max_iters=2000, goal_sample_rate=0.1, shortcut_iters=0, shortcut_max_span=3,smooth_iters=30,el_gain=0.15,rep_gain=30,repulsive_radius=50.0)
for iter in range(n_trials):

    path = planner.plan(start, goal)
    if path is not None:
        path = np.array(path)
        # loss = np.linalg.norm((path[1:] - path[:-1]) * np.array([4, 1.5, 4]), axis=1).sum() # distance
        loss = get_energy(path, wind = wind) # wind aware energy
        # loss = get_energy(path) # zero wind energy
        if loss < best_loss:
            best_loss = loss
            best_path = path
    if iter % 10 == 0:
        print(best_loss)

# print(best_loss)
print('wind aware energy: ' ,get_energy(best_path, wind = wind))
slice_z = 10
plt.figure(figsize=(10,8))
# plt.imshow(grid[:, slice_z, :].T, origin='lower')
# plt.imshow(sdf[:, slice_z, :].T, origin='lower')
# plt.imshow(wind[:, slice_z, :, 2].T, origin='lower', cmap = 'RdBu', vmin = -5, vmax = 5)
plt.imshow(building_height(grid).T, origin='lower', cmap = 'turbo', vmin = 0, vmax = 120)
plt.scatter(best_path[:,0], best_path[:,2], c=best_path[:,1], s=15, cmap = 'turbo', vmin = 0, vmax = 120)
plt.colorbar(label='Height (z)')
plt.title(f'Top-Down View at z={slice_z}')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', 'box')
plt.show()