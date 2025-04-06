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


results_dir = "results/trial1"

palette = [(75, 175, 50),
           (0, 175, 127),
           (102, 51, 229),
           (204, 51, 127)]

potential_close = np.array([[0.05, 0.1, 0.05, 0.1],
                            [0.1, 0.1, 0.1, 0.2],
                            [0.05, 0.1, 0.05, 0.2],
                            [0.1, 0.2, 0.2, 0.1]])
potential_trough = -np.array([[0.3, 0.3, 0.1, 0.03],
                              [0.3, 1, 0.3, 0.1],
                              [0.1, 0.3, 0.3, 0.03],
                              [0.03, 0.1, 0.03, 0.1]])
potential_peak = np.array([[1, 3, 1, 3],
                           [3, 1, 1, 3],
                           [1, 1, 1, 1],
                           [3, 3, 1, 1]])
potential_far = np.array([[0.1, 0.2, 0.1, 0.2],
                          [0.2, 0.2, 0.2, 0.4],
                          [0.1, 0.2, 0.1, 0.4],
                          [0.2, 0.4, 0.4, 0.2]])
masses = np.array([1e3, 1e3, 1e3, 1e3])
max_init_speed = 1e-5
speed_limit = 3e-3
n_particles_per_type = [1024, 1024, 1024, 1024]
seed = 24
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
    height, width = 512, 512
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
    np.save(os.path.join(results_dir, "seed.npy"), seed)

    simulation_history = run_simulation()
    save_simulation_mp4(simulation_history)
