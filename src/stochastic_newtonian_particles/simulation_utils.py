import numpy as np
import jax.numpy as jnp
import jax.random as jrd

from typing import *
from jax import vmap


def potential(particle_type_table: jnp.ndarray,
              potential_peak: jnp.ndarray,
              potential_close: jnp.ndarray,
              potential_trough: jnp.ndarray,
              potential_far: jnp.ndarray,
              distances: jnp.ndarray) -> jnp.ndarray:
    # don't compute a self-interaction potential,
    # as that would be meaningless
    mask = 1 - jnp.eye(particle_type_table.shape[0],
                       particle_type_table.shape[0],
                       dtype=jnp.float32)

    tiled_particle_type_table = jnp.tile(
        particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
    particle_indices = jnp.stack(
        (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

    peak = vmap(vmap(lambda i: potential_peak[i[0], i[1]]))(particle_indices)
    close = vmap(vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)
    trough = vmap(vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
    far = vmap(vmap(lambda i: potential_far[i[0], i[1]]))(particle_indices)

    close_t = distances/close
    far_t = (distances - close)/(far - close)
    close_potential = (1 - close_t)*peak + close_t*trough
    far_potential = (1 - far_t)*trough

    tot_potential = close_potential*jnp.heaviside(1 - close_t, 0) + \
        far_potential*jnp.heaviside(far_t, 0)*jnp.heaviside(1 - far_t, 0)

    return mask*tot_potential


def simulation_step(particle_type_table: jnp.ndarray,
                    potential_peak: jnp.ndarray,
                    potential_close: jnp.ndarray,
                    potential_trough: jnp.ndarray,
                    potential_far: jnp.ndarray,
                    masses: jnp.ndarray,
                    speed_limit: float,
                    dt: float,
                    sim_state: jnp.ndarray) -> jnp.ndarray:
    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    # compute difference vectors and distance matrix
    position_tiled = jnp.tile(position[jnp.newaxis], (position.shape[0], 1, 1))
    dir = position - jnp.transpose(position_tiled, (1, 0, 2))
    dir = jnp.mod(dir + 0.5, 1) - 0.5
    dist_mat = jnp.sqrt(jnp.sum(dir**2, axis=-1))

    potentials = potential(particle_type_table,
                           potential_peak,
                           potential_close,
                           potential_trough,
                           potential_far,
                           dist_mat)

    # changing the axis of the summation here will reverse the sign of the force
    accelerations = jnp.sum(potentials[..., jnp.newaxis]*dir, axis=0)/masses[particle_type_table, jnp.newaxis]

    new_velocity = velocity + dt*accelerations
    speed = jnp.sqrt(jnp.sum(new_velocity**2, axis=-1))[..., jnp.newaxis]
    new_velocity = jnp.where(speed > speed_limit, speed_limit*new_velocity/speed, new_velocity)
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1)


def simulation_init(max_init_speed: float,
                    n_particles_per_type: List[int],
                    core_size: float,
                    seed: jrd.PRNGKey,
                    initialization_mode: str) -> jnp.ndarray:
    seed1, seed2 = jrd.split(seed, num=2)
    tot_particles = sum(n_particles_per_type)

    if initialization_mode == "uniform":
        position = jrd.uniform(seed1, (tot_particles, 2), minval=0, maxval=1)

    elif initialization_mode == "core":
        len_type0 = int(np.sqrt(n_particles_per_type[0]))
        position_type0 = np.stack(np.meshgrid(
            np.linspace(0, 1, len_type0, endpoint=False),
            np.linspace(0, 1, len_type0, endpoint=False),
            indexing='ij'), axis=-1).reshape(-1, 2)

        n_other_particles = sum(n_particles_per_type[1:])
        position_other = core_size*(jrd.uniform(seed1, (n_other_particles, 2)) - 0.5) + 0.5

        position = np.concatenate((position_type0, position_other), axis=0)

    else:
        raise ValueError(
            f"Unknown initialization mode: {initialization_mode}. Supported modes are 'uniform' and 'core'.")

    velocity = max_init_speed * \
        jrd.uniform(seed2, (tot_particles, 2), minval=-1, maxval=1)

    return jnp.concatenate([position, velocity], axis=-1)
