import os
import cv2
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
from jax import vmap
from functools import partial

from typing import *
from tqdm import tqdm
from jax import jit, disable_jit

from simulation_utils import *


results_dir = "results/trial1"

close = np.array([[0.05, 0.1, 0.05, 0.1],
                  [0.1, 0.1, 0.1, 0.2],
                  [0.05, 0.1, 0.05, 0.2],
                  [0.1, 0.2, 0.2, 0.1]])
attraction = -np.array([[0.3, 0.3, 0.1, 0.03],
                        [0.3, 1, 0.3, 0.1],
                        [0.1, 0.3, 0.3, 0.03],
                        [0.03, 0.1, 0.03, 0.1]])
repulsion = np.array([[1, 3, 1, 3],
                      [3, 1, 1, 3],
                      [1, 1, 1, 1],
                      [3, 3, 1, 1]])
range = np.array([[0.1, 0.2, 0.1, 0.2],
                  [0.2, 0.2, 0.2, 0.4],
                  [0.1, 0.2, 0.1, 0.4],
                  [0.2, 0.4, 0.4, 0.2]])
masses = np.array([1e3, 1e3, 1e3, 1e3])
n_particles_per_type = [1024, 1024, 1024, 1024]
seed = 24
num_steps = 1000


def run_simulation():
    tot_particles = sum(n_particles_per_type)
    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(n_particles_per_type)])

    # Initialize positions and velocities
    key = jrd.PRNGKey(seed)
    sim_state = simulation_init(key, tot_particles)

    simulation_history = np.zeros((num_steps, tot_particles, 4))
    simulation_history[0] = sim_state

    for t in tqdm(range(1, num_steps)):
        sim_state = simulation_step(particle_type_table,
                                    attraction,
                                    close,
                                    repulsion,
                                    range,
                                    masses,
                                    sim_state)
        simulation_history[t] = sim_state

    return simulation_history


def save_simulation_mp4(simulation_history: np.ndarray):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 512, 512
    out = cv2.VideoWriter(os.path.join(
        results_dir, "simulation.mp4"), fourcc, 30.0, (width, height))

    for frame in simulation_history:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(frame.shape[0]):
            x, y = int(frame[i][0] * width), int(frame[i][1] * height)
            color = (int(frame[i][3] * 255), int(frame[i]
                     [3] * 255), int(frame[i][3] * 255))
            cv2.circle(img, (x, y), 5, color, -1)
        out.write(img)

    out.release()


if __name__ == "__main__":
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(os.path.join(results_dir, "close.npy"), close)
    np.save(os.path.join(results_dir, "attraction.npy"), attraction)
    np.save(os.path.join(results_dir, "repulsion.npy"), repulsion)
    np.save(os.path.join(results_dir, "range.npy"), range)
    np.save(os.path.join(results_dir, "masses.npy"), masses)
    np.save(os.path.join(results_dir, "n_particles_per_type.npy"), n_particles_per_type)
    np.save(os.path.join(results_dir, "seed.npy"), seed)

    simulation_history = run_simulation()
    save_simulation_mp4(simulation_history)
    print("Simulation saved to:", os.path.join(results_dir, "simulation.mp4"))