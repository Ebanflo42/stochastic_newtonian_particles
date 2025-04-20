import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import *
from tqdm import tqdm


def save_simulation_mp4(results_dir: str,
                        filename: str,
                        n_particles_per_type: List[int],
                        palette: List[Tuple[int, int, int]],
                        simulation_history: np.ndarray):
    print("Rendering video...")

    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(n_particles_per_type)])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 720, 720
    out = cv2.VideoWriter(os.path.join(
        results_dir, filename), fourcc, 30.0, (width, height))

    for frame in tqdm(simulation_history):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(frame.shape[0]):
            x, y = int(frame[i][0] * width), int(frame[i][1] * height)
            color = palette[particle_type_table[i]]
            cv2.circle(img, (x, y), 1, color, -1)
        out.write(img)

    out.release()


def plot_energy_and_momentum(simulation_history: np.ndarray,
                             potential_energy: np.ndarray,
                             masses: np.ndarray,
                             n_particles_per_type: List[int],
                             filename: str):
    particle_type_table = \
        np.concatenate(
            [[i] * n for i, n in enumerate(n_particles_per_type)]
            ).astype(np.int32)
    mass_table = masses[particle_type_table]

    momentum = mass_table[np.newaxis, :, np.newaxis]*simulation_history[:, :, 2:4]
    kinetic_energy = np.sum(0.5*np.sum(momentum**2, axis=-1)/mass_table, axis=-1)
    momentum = np.sum(momentum, axis=-1)

    fig = plt.figure()
    axk = fig.add_subplot(2, 2, 1)
    axk.plot(np.arange(simulation_history.shape[0]),
             kinetic_energy)
    axk.set_title("Kinetic Energy")
    axp = fig.add_subplot(2, 2, 2)
    axp.plot(np.arange(simulation_history.shape[0]),
             np.sum(potential_energy, axis=-1))
    axp.set_title("Potential Energy")
    axt = fig.add_subplot(2, 2, 3)
    axt.plot(np.arange(simulation_history.shape[0]),
             np.sum(potential_energy, axis=-1) + \
                np.sum(potential_energy, axis=-1))
    axt.set_title("Total Energy")
    axm = fig.add_subplot(2, 2, 4)
    axm.plot(np.arange(simulation_history.shape[0]),
            momentum[:, 0],
            label='Momentum X')
    axm.plot(np.arange(simulation_history.shape[0]),
            momentum[:, 1],
            label='Momentum Y')
    axm.legend()
    plt.tight_layout()
    plt.savefig(filename)