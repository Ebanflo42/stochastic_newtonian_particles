import numpy as np
import jax.numpy as jnp
import jax.random as jrd

from typing import *
from jax import vmap


def matrix_exponetial(m: jnp.ndarray) -> jnp.ndarray:
    square = m@m
    cube = square@m
    fourth = cube@m
    return jnp.eye(m.shape[0]) + m + square/2 + cube/6 + fourth/24


def gram_schmidt(vectors: jnp.ndarray) -> jnp.ndarray:
    """Perform Gram-Schmidt orthogonalization on a set of vectors."""
    n_vectors = vectors.shape[0]
    orthogonal_vectors = jnp.zeros_like(vectors)

    for i in range(n_vectors):
        v = vectors[i]
        for j in range(i):
            v -= jnp.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        orthogonal_vectors.at[i].set(v/jnp.linalg.norm(v))

    return orthogonal_vectors


def rotation_constraint(n: int) -> jnp.ndarray:
    constraint = jnp.eye(n)
    constraint.at[0].set(jnp.ones((n,)))
    return gram_schmidt(constraint).T


def potential(peak_gathered: jnp.ndarray,
              close_gathered: jnp.ndarray,
              trough_gathered: jnp.ndarray,
              far_gathered: jnp.ndarray,
              distances: jnp.ndarray) -> jnp.ndarray:
    # don't compute a self-interaction potential,
    # as that would be meaningless
    mask = 1 - jnp.eye(distances.shape[0])

    # linear interpolation on two different pieces
    close_t = distances/close_gathered
    far_t = (distances - close_gathered)/(far_gathered - close_gathered)
    close_potential = (1 - close_t)*peak_gathered + close_t*trough_gathered
    far_potential = (1 - far_t)*trough_gathered

    # switch between the two pieces using step function
    tot_potential = close_potential*jnp.heaviside(1 - close_t, 0) + \
        far_potential*jnp.heaviside(far_t, 0)*jnp.heaviside(1 - far_t, 0)

    return mask*tot_potential


def random_momentum_transfer(key: jrd.PRNGKey,
                             rotation_constraint: jnp.ndarray,
                             noise_strength: float,
                             potential_far: jnp.ndarray,
                             distances: jnp.ndarray,
                             masses: jnp.ndarray,
                             velocity: jnp.ndarray) -> jnp.ndarray:
    # randomly generate antisymmetric matrix
    gaussian = noise_strength*jrd.normal(key, shape=distances.shape)
    generator = 0.5*(gaussian - gaussian.T)

    # only particles capable of "observing" each other will trade momentum
    generator *= (distances < potential_far).astype(jnp.float32)

    # the constraint should ensure that the sum of all momenta is conserved
    generator = generator@rotation_constraint.T
    generator[:1] = 0
    generator[:, :1] = 0
    generator = rotation_constraint@generator
    rotation = matrix_exponetial(generator)

    momenta = masses*velocity
    transferred_momenta = rotation@momenta
    transferred_velocity = transferred_momenta/masses

    return transferred_velocity


def simulation_step_deterministic(particle_type_table: jnp.ndarray,
                                  potential_peak: jnp.ndarray,
                                  potential_close: jnp.ndarray,
                                  potential_trough: jnp.ndarray,
                                  potential_far: jnp.ndarray,
                                  masses: jnp.ndarray,
                                  speed_limit: float,
                                  dt: float,
                                  sim_state: jnp.ndarray) -> jnp.ndarray:
    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    tiled_particle_type_table = jnp.tile(
        particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
    particle_indices = jnp.stack(
        (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

    peak_gathered = vmap(
        vmap(lambda i: potential_peak[i[0], i[1]]))(particle_indices)
    close_gathered = vmap(
        vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)
    trough_gathered = vmap(
        vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
    far_gathered = vmap(vmap(lambda i: potential_far[i[0], i[1]]))(
        particle_indices)

    # compute difference vectors and distance matrix
    position_tiled = jnp.tile(position[jnp.newaxis], (position.shape[0], 1, 1))
    dir = position - jnp.transpose(position_tiled, (1, 0, 2))
    dir = jnp.mod(dir + 0.5, 1) - 0.5
    dist_mat = jnp.sqrt(jnp.sum(dir**2, axis=-1))

    potentials = potential(particle_type_table,
                           peak_gathered,
                           close_gathered,
                           trough_gathered,
                           far_gathered,
                           dist_mat)

    # changing the axis of the summation here will reverse the sign of the force
    accelerations = jnp.sum(
        potentials[..., jnp.newaxis]*dir, axis=0)/masses[particle_type_table, jnp.newaxis]

    new_velocity = velocity + dt*accelerations
    speed = jnp.sqrt(jnp.sum(new_velocity**2, axis=-1))[..., jnp.newaxis]
    new_velocity = jnp.where(
        speed > speed_limit, speed_limit*new_velocity/speed, new_velocity)
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1)


def simulation_step_stochastic(particle_type_table: jnp.ndarray,
                               potential_peak: jnp.ndarray,
                               potential_close: jnp.ndarray,
                               potential_trough: jnp.ndarray,
                               potential_far: jnp.ndarray,
                               masses: jnp.ndarray,
                               speed_limit: float,
                               dt: float,
                               key: jrd.PRNGKey,
                               noise_strength: float,
                               rotation_constraint: jnp.ndarray,
                               sim_state: jnp.ndarray) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
    this_key, next_key = jrd.split(key, num=2)

    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    tiled_particle_type_table = jnp.tile(
        particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
    particle_indices = jnp.stack(
        (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

    peak_gathered = vmap(
        vmap(lambda i: potential_peak[i[0], i[1]]))(particle_indices)
    close_gathered = vmap(
        vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)
    trough_gathered = vmap(
        vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
    far_gathered = vmap(
        vmap(lambda i: potential_far[i[0], i[1]]))(particle_indices)

    # compute difference vectors and distance matrix
    position_tiled = jnp.tile(position[jnp.newaxis], (position.shape[0], 1, 1))
    dir = position - jnp.transpose(position_tiled, (1, 0, 2))
    dir = jnp.mod(dir + 0.5, 1) - 0.5
    dist_mat = jnp.sqrt(jnp.sum(dir**2, axis=-1))

    potentials = potential(particle_type_table,
                           peak_gathered,
                           close_gathered,
                           trough_gathered,
                           far_gathered,
                           dist_mat)

    # changing the axis of the summation here will reverse the sign of the force
    masses_gathered = masses[particle_type_table, jnp.newaxis]
    accelerations = jnp.sum(
        potentials[..., jnp.newaxis]*dir, axis=0)/masses_gathered

    new_velocity = velocity + dt*accelerations
    speed = jnp.sqrt(jnp.sum(new_velocity**2, axis=-1))[..., jnp.newaxis]
    new_velocity = jnp.where(
        speed > speed_limit, speed_limit*new_velocity/speed, new_velocity)
    new_velocity = random_momentum_transfer(this_key,
                                            rotation_constraint,
                                            noise_strength,
                                            far_gathered,
                                            dist_mat,
                                            masses_gathered,
                                            new_velocity)

    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1), next_key


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
        position_other = core_size * \
            (jrd.uniform(seed1, (n_other_particles, 2)) - 0.5) + 0.5

        position = np.concatenate((position_type0, position_other), axis=0)

    else:
        raise ValueError(
            f"Unknown initialization mode: {initialization_mode}. Supported modes are 'uniform' and 'core'.")

    velocity = max_init_speed * \
        jrd.uniform(seed2, (tot_particles, 2), minval=-1, maxval=1)

    return jnp.concatenate([position, velocity], axis=-1)
