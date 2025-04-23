import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrd

from typing import *
from jax import vmap, jit
from jax.lax import fori_loop
from functools import partial
from jax.experimental import sparse
from joblib import Parallel, delayed


def get_diff_wrapped(p1, p2):
    """Get the direction vector between two points, wrapped around the unit square."""
    diff = p1 - p2
    diff = jnp.mod(diff + 0.5, 1) - 0.5
    return diff


def lennard_jones(trough_gathered: jnp.ndarray,
                  close_gathered: jnp.ndarray,
                  positions: jnp.ndarray) -> float:
    positions_tiled = jnp.tile(
        positions[jnp.newaxis], (positions.shape[0], 1, 1))
    diffs = get_diff_wrapped(
        positions_tiled, jnp.transpose(positions_tiled, (1, 0, 2)))

    sq_dist_mat = jnp.sum(diffs**2, axis=-1)
    six_dist_mat = sq_dist_mat**3
    twlv_dist_mat = six_dist_mat**2

    potentials = 4*trough_gathered*(close_gathered**12/(1e-12 + twlv_dist_mat) +
                                    close_gathered**6/(1e-12 + six_dist_mat))

    potentials_masked = jnp.tril(potentials, k=-1)

    return jnp.sum(potentials_masked)


def build_rotation_matrix(pairs: jnp.ndarray,
                          unsampled_indices: jnp.ndarray,
                          angles: jnp.ndarray,
                          n_particles: int) -> sparse.BCOO:
    s = jnp.sin(angles)
    c = jnp.cos(angles)
    z = jnp.zeros_like(angles)
    indices = jnp.concatenate([jnp.stack([2*pairs[:, 0],
                                          2*pairs[:, 0]], axis=-1),
                               jnp.stack([2*pairs[:, 0],
                                          2*pairs[:, 0] + 1], axis=-1),
                               jnp.stack([2*pairs[:, 0],
                                          2*pairs[:, 1]], axis=-1),
                               jnp.stack([2*pairs[:, 0],
                                          2*pairs[:, 1] + 1], axis=-1),
                               # second row starts
                               jnp.stack([2*pairs[:, 0] + 1,
                                          2*pairs[:, 0]], axis=-1),
                               jnp.stack([2*pairs[:, 0] + 1,
                                          2*pairs[:, 0] + 1], axis=-1),
                               jnp.stack([2*pairs[:, 0] + 1,
                                          2*pairs[:, 1]], axis=-1),
                               jnp.stack([2*pairs[:, 0] + 1,
                                          2*pairs[:, 1] + 1], axis=-1),
                               # third row starts
                               jnp.stack([2*pairs[:, 1],
                                          2*pairs[:, 0]], axis=-1),
                               jnp.stack([2*pairs[:, 1],
                                          2*pairs[:, 0] + 1], axis=-1),
                               jnp.stack([2*pairs[:, 1],
                                          2*pairs[:, 1]], axis=-1),
                               jnp.stack([2*pairs[:, 1],
                                          2*pairs[:, 1] + 1], axis=-1),
                               # fourth row starts
                               jnp.stack([2*pairs[:, 1] + 1,
                                          2*pairs[:, 0]], axis=-1),
                               jnp.stack([2*pairs[:, 1] + 1,
                                          2*pairs[:, 0] + 1], axis=-1),
                               jnp.stack([2*pairs[:, 1] + 1,
                                          2*pairs[:, 1]], axis=-1),
                               jnp.stack([2*pairs[:, 1] + 1,
                                          2*pairs[:, 1] + 1], axis=-1),
                               # identity elsewhere
                               jnp.stack([2*unsampled_indices,
                                          2*unsampled_indices], axis=-1),
                               # identity elsewhere
                               jnp.stack([2*unsampled_indices + 1,
                                          2*unsampled_indices + 1], axis=-1)], axis=0)
    values = jnp.concatenate([1 + c,  z, 1 - c,    -s,
                             z,  1 + c,     s, 1 - c,
                            1 - c, -s, 1 + c,     z,
                            s,  1 - c,     z, 1 + c,
                            2*jnp.ones((2*len(unsampled_indices),))], axis=0)
    #values = jnp.concatenate([z,  1 + c, s, c - 1,
    #                          1 + c, z,  1 - c, s,
    #                          s, c - 1, z, 1 + c,
    #                          1 - c,  s, 1 + c, z,
    #                          2*jnp.ones((2*len(unsampled_indices),))], axis=0)
    return sparse.BCOO((0.5*values, indices), shape=(2*n_particles, 2*n_particles))


def potential(peak_gathered: jnp.ndarray,
              close_gathered: jnp.ndarray,
              trough_gathered: jnp.ndarray,
              far_gathered: jnp.ndarray,
              distances: jnp.ndarray) -> jnp.ndarray:
    # don't compute a self-interaction potential,
    # as that would be meaningless
    mask = jnp.tril(jnp.ones_like(distances), k=-1)

    # linear interpolation on two different pieces
    close_t = distances/close_gathered
    far_t = (distances - close_gathered)/(far_gathered - close_gathered)
    close_potential = (1 - close_t)*peak_gathered + close_t*trough_gathered
    far_potential = (1 - far_t)*trough_gathered

    # switch between the two pieces using step function
    tot_potential = close_potential*jnp.heaviside(1 - close_t, 0) + \
        far_potential*jnp.heaviside(far_t, 0)*jnp.heaviside(1 - far_t, 0)

    return jnp.sum(mask*tot_potential)


def compute_forces(max_gathered: jnp.ndarray,
                   close_gathered: jnp.ndarray,
                   min_gathered: jnp.ndarray,
                   far_gathered: jnp.ndarray,
                   distances: jnp.ndarray,
                   diffs: jnp.ndarray) -> jnp.ndarray:
    # don't compute a self-interaction potential,
    # as that would be meaningless
    mask = 1 - jnp.eye(distances.shape[0])

    close_t = distances/close_gathered
    magnitude = max_gathered*jnp.heaviside(1 - close_t, 0) + \
        min_gathered*jnp.heaviside(close_t, 0) * \
        jnp.heaviside(1 - distances/far_gathered, 0)
    magnitude *= mask

    return jnp.sum(magnitude[..., jnp.newaxis]*diffs, axis=0)


def random_momentum_transfer(transfer_intensity: float,
                             distances: jnp.ndarray,
                             pairs_to_transfer: jnp.ndarray,
                             unsampled_indices: jnp.ndarray,
                             masses: jnp.ndarray,
                             velocity: jnp.ndarray) -> jnp.ndarray:
    angles = jnp.pi*jnp.exp(-distances[pairs_to_transfer[:, 0],
                                       pairs_to_transfer[:, 1]]**2/transfer_intensity)
    rotation = build_rotation_matrix(pairs_to_transfer,
                                     unsampled_indices,
                                     angles,
                                     masses.shape[0])
    # print(rotation.data.shape, jnp.linalg.det(rotation.todense()))
    momenta = jnp.reshape(masses*velocity, (-1,))
    transferred_momenta = momenta@rotation
    transferred_velocity = jnp.reshape(transferred_momenta, (-1, 2))/masses

    return transferred_velocity


def simulation_step_deterministic(particle_type_table: jnp.ndarray,
                                  potential_trough: jnp.ndarray,
                                  potential_close: jnp.ndarray,
                                  masses: jnp.ndarray,
                                  dt: float,
                                  sim_state: jnp.ndarray
                                  ) -> jnp.ndarray:
    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    tiled_particle_type_table = jnp.tile(
        particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
    particle_indices = jnp.stack(
        (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

    trough_gathered = vmap(
        vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
    close_gathered = vmap(
        vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)

    lj = partial(lennard_jones, trough_gathered, close_gathered)
    force_fn = jax.grad(lj)
    forces = -force_fn(position)

    # compute forces
    # forces = -force_fn(position)
    accelerations = forces/masses[particle_type_table, jnp.newaxis]

    new_velocity = velocity + dt*accelerations
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1)


def simulation_step_deterministic_energy_log(particle_type_table: jnp.ndarray,
                                             potential_trough: jnp.ndarray,
                                             potential_close: jnp.ndarray,
                                             masses: jnp.ndarray,
                                             dt: float,
                                             sim_state: jnp.ndarray
                                             ) -> Tuple[jnp.ndarray, float]:
    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    tiled_particle_type_table = jnp.tile(
        particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
    particle_indices = jnp.stack(
        (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

    trough_gathered = vmap(
        vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
    close_gathered = vmap(
        vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)

    lj = partial(lennard_jones, trough_gathered, close_gathered)
    force_fn = jax.value_and_grad(lj)

    # compute forces
    potential, forces = force_fn(position)
    accelerations = -forces/masses[particle_type_table, jnp.newaxis]

    new_velocity = velocity + dt*accelerations
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1), potential


def make_simulation_step_stochastic(particle_type_table: jnp.ndarray,
                                    potential_trough: jnp.ndarray,
                                    potential_close: jnp.ndarray,
                                    masses: jnp.ndarray,
                                    dt: float,
                                    n_transfers_per_step: int,
                                    transfer_intensity: float
                                    ) -> Callable:

    def sim_step_jittable(sim_state: jnp.ndarray,
                          sampled_neighbors: jnp.ndarray,
                          unsampled_indices: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
        position, velocity = sim_state[:, :2], sim_state[:, 2:]

        position_tiled = jnp.tile(
            position[jnp.newaxis], (position.shape[0], 1, 1))
        diffs = get_diff_wrapped(
            position_tiled, jnp.transpose(position_tiled, (1, 0, 2)))
        distances = jnp.linalg.norm(diffs, axis=-1)

        tiled_particle_type_table = jnp.tile(
            particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
        particle_indices = jnp.stack(
            (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

        trough_gathered = vmap(
            vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
        close_gathered = vmap(
            vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)

        lj = partial(lennard_jones, trough_gathered, close_gathered)
        force_fn = jax.grad(lj)

        # compute forces
        forces = -force_fn(position)
        masses_gathered = masses[particle_type_table, jnp.newaxis]
        accelerations = forces/masses_gathered

        new_velocity = velocity + dt*accelerations
        new_velocity = random_momentum_transfer(transfer_intensity,
                                                distances,
                                                sampled_neighbors,
                                                unsampled_indices,
                                                masses_gathered,
                                                new_velocity)
        new_position = position + dt*new_velocity
        new_position = jnp.mod(new_position, 1)

        return jnp.concatenate((new_position, new_velocity), axis=-1)

    sim_step_jitted = jit(sim_step_jittable)

    def sim_step(key: jrd.PRNGKey,
                 sim_state: jnp.ndarray) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
        this_key, next_key = jrd.split(key, num=2)

        sampled_neighbors = jrd.choice(this_key,
                                       sim_state.shape[0],
                                       shape=(n_transfers_per_step, 2),
                                       replace=False)
        # in order to construct the sparse rotation matrix we have to keep track
        # of which indices are not sampled so that we can apply the identity matrix
        # to them
        unsampled_indices = list(range(sim_state.shape[0]))
        sampled_indices = set()
        for edge in np.array(sampled_neighbors):
            sampled_indices.add(edge[0])
            sampled_indices.add(edge[1])
        unsampled_indices = [i for i in unsampled_indices
                             if i not in sampled_indices]
        unsampled_indices = jnp.array(unsampled_indices)

        next_state = sim_step_jitted(sim_state,
                                     sampled_neighbors,
                                     unsampled_indices)

        return next_key, next_state

    return sim_step


def make_simulation_step_stochastic_energy_log(particle_type_table: jnp.ndarray,
                                               potential_trough: jnp.ndarray,
                                               potential_close: jnp.ndarray,
                                               masses: jnp.ndarray,
                                               dt: float,
                                               n_transfers_per_step: int,
                                               transfer_intensity: float) -> Callable:

    def sim_step_jittable(sim_state: jnp.ndarray,
                          sampled_neighbors: jnp.ndarray,
                          unsampled_indices: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
        position, velocity = sim_state[:, :2], sim_state[:, 2:]

        position_tiled = jnp.tile(
            position[jnp.newaxis], (position.shape[0], 1, 1))
        diffs = get_diff_wrapped(
            position_tiled, jnp.transpose(position_tiled, (1, 0, 2)))
        distances = jnp.linalg.norm(diffs, axis=-1)

        tiled_particle_type_table = jnp.tile(
            particle_type_table[jnp.newaxis], (particle_type_table.shape[0], 1))
        particle_indices = jnp.stack(
            (tiled_particle_type_table, tiled_particle_type_table.T), axis=-1)

        trough_gathered = vmap(
            vmap(lambda i: potential_trough[i[0], i[1]]))(particle_indices)
        close_gathered = vmap(
            vmap(lambda i: potential_close[i[0], i[1]]))(particle_indices)

        lj = partial(lennard_jones, trough_gathered, close_gathered)
        force_fn = jax.value_and_grad(lj)

        # compute forces
        potential, forces = force_fn(position)
        masses_gathered = masses[particle_type_table, jnp.newaxis]
        accelerations = -forces/masses_gathered

        new_velocity = velocity + dt*accelerations
        new_velocity = random_momentum_transfer(transfer_intensity,
                                                distances,
                                                sampled_neighbors,
                                                unsampled_indices,
                                                masses_gathered,
                                                new_velocity)
        new_position = position + dt*new_velocity
        new_position = jnp.mod(new_position, 1)

        return jnp.concatenate((new_position, new_velocity), axis=-1), potential

    sim_step_jitted = jit(sim_step_jittable)

    def sim_step(key: jrd.PRNGKey,
                 sim_state: jnp.ndarray) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
        this_key, next_key = jrd.split(key, num=2)

        sampled_neighbors = jrd.choice(this_key,
                                       sim_state.shape[0],
                                       shape=(n_transfers_per_step, 2),
                                       replace=False)
        # in order to construct the sparse rotation matrix we have to keep track
        # of which indices are not sampled so that we can apply the identity matrix
        # to them
        unsampled_indices = list(range(sim_state.shape[0]))
        sampled_indices = set()
        for edge in np.array(sampled_neighbors):
            sampled_indices.add(edge[0])
            sampled_indices.add(edge[1])
        unsampled_indices = [i for i in unsampled_indices
                             if i not in sampled_indices]
        unsampled_indices = jnp.array(unsampled_indices)

        next_state, potential = sim_step_jitted(sim_state,
                                                sampled_neighbors,
                                                unsampled_indices)

        return next_key, next_state, potential

    return sim_step


def simulation_init(max_init_speed: float,
                    n_particles_per_type: List[int],
                    core_size: float,
                    seed: jrd.PRNGKey,
                    initialization_mode: str) -> jnp.ndarray:
    seed1, seed2 = jrd.split(seed, num=2)
    tot_particles = sum(n_particles_per_type)

    if initialization_mode == "uniform":
        position = jrd.uniform(seed1, (tot_particles, 2), minval=0, maxval=1)
        velocity = max_init_speed * \
            jrd.uniform(seed2, (tot_particles, 2), minval=-1, maxval=1)

    elif initialization_mode == "core":
        len_type01 = int(np.sqrt(sum(n_particles_per_type[:2])))
        position_type01 = np.stack(np.meshgrid(
            np.linspace(0, 1, 1 + len_type01, endpoint=True)[1:],
            np.linspace(0, 1, 1 + len_type01, endpoint=True)[1:],
            indexing='ij'), axis=-1).reshape(-1, 2)

        position_type0 = position_type01[0:len(position_type01):2]
        position_type1 = position_type01[1:len(position_type01):2]

        n_other_particles = sum(n_particles_per_type[2:])
        len_other = int(np.sqrt(n_other_particles))
        # position_other = core_size * \
        #    (jrd.uniform(seed1, (n_other_particles, 2)) - 0.5) + 0.5
        position_other = np.stack(np.meshgrid(
            np.linspace(0.5 - 0.5*core_size, 0.5 + 0.5 *
                        core_size, len_other, endpoint=False),
            np.linspace(0.5 - 0.5*core_size, 0.5 + 0.5 *
                        core_size, len_other, endpoint=False),
            indexing='ij'), axis=-1).reshape(-1, 2)
        position_other = np.concatenate((position_other[0:len(position_other):2],
                                         position_other[1:len(position_other):2]),
                                        axis=0)

        position = np.concatenate(
            (position_type0, position_type1, position_other), axis=0)

        velocity_type01 = jnp.zeros_like(position_type01)
        velocity_other = max_init_speed * \
            jrd.uniform(seed2, (n_other_particles, 2), minval=-1, maxval=1)
        velocity = np.concatenate((velocity_type01, velocity_other), axis=0)

    else:
        raise ValueError(
            f"Unknown initialization mode: {initialization_mode}. Supported modes are 'uniform' and 'core'.")

    return jnp.concatenate([position, velocity], axis=-1)
