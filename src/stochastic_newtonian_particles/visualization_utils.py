import os
import cv2
import numpy as np

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

