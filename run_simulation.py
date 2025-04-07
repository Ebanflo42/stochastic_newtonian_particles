import os
import cv2
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap
from functools import partial

from typing import *
from tqdm import tqdm
from functools import partial
from jax import jit, disable_jit

from simulation_utils import *


results_dir = "results/trial41"

# green, cyan, blue-pink, red-magenta, orange, bright yellow, blue-magenta, blue-cyan
palette = [(75, 200, 50),
           (0, 175, 127),
           (102, 51, 229),
           (204, 51, 127),
           (255, 127, 0),
           (255, 255, 0),
           (127, 0, 255),
           (0, 127, 255)]

potential_close = 2*np.array([[0.03, 0.015, 0.025, 0.025],
                            [0.015, 0.05, 0.025, 0.1],
                            [0.025, 0.025, 0.1, 0.05],
                            [0.025, 0.1, 0.05, 0.025]])
potential_close = np.tile(potential_close, (2, 2))
#potential_close[4:, :4] *= 2.0
#potential_close[:4, 4:] *= 2.0
potential_trough = -0.1*np.array([[0.5, 0.25, 0.5, 0],
                              [0.25, 0.5, 0.25, 0],
                              [0.5, 0.25, 1, 0.25],
                              [0, 0, 0.25, 0.25]])
potential_trough = np.tile(potential_trough, (2, 2))
potential_trough[4:, :4] *= 0.3
potential_trough[:4, 4:] *= 0.5
potential_trough[:4, :4] *= 2.0
potential_trough[4:, 4:] *= 3.0

potential_peak = 1.5*np.array([[2, 0.5, 0.3, 3],
                           [0.5, 2, 0.3, 3],
                           [0.3, 0.3, 3, 3],
                           [3, 3, 3, 3]])
potential_peak = np.tile(potential_peak, (2, 2))
potential_peak[4:, :4] *= 2.0
potential_peak[:4, 4:] *= 3.0
potential_peak[:4, :4] *= 0.5
potential_peak[4:, 4:] *= 0.3
potential_far = 2*np.array([[0.15, 0.1, 0.2, 0.1],
                          [0.1, 0.2, 0.1, 0.2],
                          [0.2, 0.1, 0.2, 0.1],
                          [0.1, 0.2, 0.1, 0.05]])
potential_far = np.tile(potential_far, (2, 2))
#potential_far[4:, :4] *= 2.0
#potential_far[:4, 4:] *= 2.0
potential_far = np.clip(potential_far, 0, 0.25)
masses = np.array([1e3, 1e3, 1e3, 1e3])
max_init_speed = 0
speed_limit = 5e-3
n_particles_per_type = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
seed = 25
num_steps = 1000


def run_simulation():
    print("Running simulation...")

    # Create an array indicating the type of each particle
    tot_particles = sum(n_particles_per_type)
    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(n_particles_per_type)])

    # Initialize positions and velocities
    key = jrd.PRNGKey(seed)
    sim_state = simulation_init(tot_particles, max_init_speed, key)

    # Initialize simulation history
    simulation_history = np.zeros((num_steps, tot_particles, 4))
    simulation_history[0] = sim_state

    # Compile the simulation step function to improve performance
    # disable_jit()
    step = jit(partial(simulation_step,
                       jnp.array(particle_type_table),
                       jnp.array(potential_peak),
                       jnp.array(potential_close),
                       jnp.array(potential_trough),
                       jnp.array(potential_far),
                       jnp.array(masses),
                       speed_limit))

    for t in tqdm(range(1, num_steps)):
        sim_state = step(sim_state)
        simulation_history[t] = sim_state

    return simulation_history


def save_simulation_mp4(simulation_history: np.ndarray):
    print("Rendering video...")

    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(n_particles_per_type)])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 720, 720
    out = cv2.VideoWriter(os.path.join(
        results_dir, "simulation.mp4"), fourcc, 30.0, (width, height))

    for frame in tqdm(simulation_history):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(frame.shape[0]):
            x, y = int(frame[i][0] * width), int(frame[i][1] * height)
            color = palette[particle_type_table[i]]
            cv2.circle(img, (x, y), 1, color, -1)
        out.write(img)

    out.release()


if __name__ == "__main__":
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(os.path.join(results_dir, "potential_close.npy"), potential_close)
    np.save(os.path.join(results_dir, "potential_trough.npy"), potential_trough)
    np.save(os.path.join(results_dir, "potential_peak.npy"), potential_peak)
    np.save(os.path.join(results_dir, "potential_far.npy"), potential_far)
    np.save(os.path.join(results_dir, "masses.npy"), masses)
    np.save(os.path.join(results_dir, "n_particles_per_type.npy"),
            n_particles_per_type)
    np.save(os.path.join(results_dir, "max_init_speed.npy"), max_init_speed)
    np.save(os.path.join(results_dir, "speed_limit.npy"), speed_limit)
    np.save(os.path.join(results_dir, "seed.npy"), seed)

    simulation_history = run_simulation()
    save_simulation_mp4(simulation_history)
