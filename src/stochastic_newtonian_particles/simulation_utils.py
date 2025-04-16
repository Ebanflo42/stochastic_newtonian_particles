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


def matrix_exponetial(m: jnp.ndarray) -> jnp.ndarray:
    square = m@m
    cube = square@m
    fourth = cube@m
    return jnp.eye(m.shape[0]) + m + square/2 + cube/6 + fourth/24


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Perform Gram-Schmidt orthogonalization on a set of vectors."""
    n_vectors = vectors.shape[0]
    orthogonal_vectors = np.zeros_like(vectors)

    for i in range(n_vectors):
        v = vectors[i]
        for j in range(i):
            v -= np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        orthogonal_vectors[i] = v/jnp.linalg.norm(v)

    return orthogonal_vectors


def rotation_constraint(n: int) -> np.ndarray:
    constraint = np.eye(n)
    constraint[0] = np.ones((n,))
    return gram_schmidt(constraint).T


def rotation_matrix_4d(theta: float) -> jnp.ndarray:
    """A rotation in the plane orthogonal to (1, 0, 1, 0) and (0, 1, 0, 1)"""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    m = jnp.array([[1 + c, 0, 1 - c, -s],
                   [0, 1 + c, s, 1 - c],
                   [1 - c, -s, 1 + c, 0],
                   [s, 1 - c, 0, 1 + c]])
    return m


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
    values = jnp.concatenate([1 + c,     z, 1 - c,    -s,
                              z, 1 + c,     s, 1 - c,
                              1 - c,    -s, 1 + c,     z,
                              s, 1 - c,     z, 1 + c,
                              2*jnp.ones((2*len(unsampled_indices),))], axis=0)
    return sparse.BCOO((0.5*values, indices), shape=(2*n_particles, 2*n_particles))
    # return sparse.BCOO((jnp.ones_like(values), indices), shape=(2*n_particles, 2*n_particles))


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


def compute_force_sparse(n_particles: int,
                         peak: jnp.ndarray,
                         close: jnp.ndarray,
                         trough: jnp.ndarray,
                         far: jnp.ndarray,
                         edges: jnp.ndarray,
                         type_edges: jnp.ndarray,
                         distances: jnp.ndarray,
                         diffs: jnp.ndarray,
                         ) -> jnp.ndarray:
    # don't compute a self-interaction potential,
    # as that would be meaningless

    # linear interpolation on two different pieces
    close_t = distances/close[type_edges[:, 0], type_edges[:, 1]]
    far_t = (distances - close[type_edges[:, 0], type_edges[:, 1]]) / \
        (far[type_edges[:, 0], type_edges[:, 1]] -
         close[type_edges[:, 0], type_edges[:, 1]])
    close_potential = \
        (1 - close_t)*peak[type_edges[:, 0], type_edges[:, 1]] + \
        close_t*trough[type_edges[:, 0], type_edges[:, 1]]
    far_potential = (1 - far_t)*trough[type_edges[:, 0], type_edges[:, 1]]

    # switch between the two pieces using step function
    tot_potential = close_potential*jnp.heaviside(1 - close_t, 0) + \
        far_potential*jnp.heaviside(far_t, 0)*jnp.heaviside(1 - far_t, 0)

    # check sign here!
    potential_matrix = sparse.BCOO((tot_potential, edges), shape=(
        n_particles, n_particles))
    diff_x_matrix = sparse.BCOO((diffs[:, 0], edges), shape=(
        n_particles, n_particles))
    diff_y_matrix = sparse.BCOO((diffs[:, 1], edges), shape=(
        n_particles, n_particles))
    force_vectors = jnp.stack(((diff_x_matrix*potential_matrix).sum(axis=1),
                               (diff_y_matrix*potential_matrix).sum(axis=1)), axis=-1)

    return force_vectors


def potential_serial(peak: jnp.ndarray,
                     close: jnp.ndarray,
                     trough: jnp.ndarray,
                     far: jnp.ndarray,
                     particle_type_table: jnp.ndarray,
                     masses: jnp.ndarray,
                     positions: jnp.ndarray,
                     current_particle_id: int) -> jnp.ndarray:
    """Compute the acceleration for a single particle."""
    mass = masses[current_particle_id]
    position = positions[current_particle_id]
    particle_type = particle_type_table[current_particle_id]
    acceleration = jnp.array([0.0, 0.0])

    def accumulate_acceleration(i: int, accum: float) -> float:
        other_position = positions[i]
        diff = position - other_position
        diff = jnp.mod(diff + 0.5, 1) - 0.5
        dist = jnp.sqrt(jnp.sum(diff**2))

        other_type = particle_type_table[i]

        # linear interpolation on two different pieces
        close_t = dist/close[particle_type, other_type]
        far_t = (dist - close[particle_type, other_type])\
            / (far[particle_type, other_type] - close[particle_type, other_type])
        close_potential = (
            1 - close_t)*peak[particle_type, other_type] + close_t*trough[particle_type, other_type]
        far_potential = (1 - far_t)*trough[particle_type, other_type]

        # switch between the two pieces using step function
        p = close_potential*jnp.heaviside(1 - close_t, 0) + \
            far_potential*jnp.heaviside(far_t, 0)*jnp.heaviside(1 - far_t, 0)
        p *= jnp.astype(i != current_particle_id, jnp.float32)

        # check sign here!
        accel = p*diff/mass

        return accum + accel

    return fori_loop(0, positions.shape[0], accumulate_acceleration, acceleration)


def random_momentum_transfer(key: jrd.PRNGKey,
                             transfer_intensity: float,
                             distances: jnp.ndarray,
                             pairs_to_transfer: jnp.ndarray,
                             unsampled_indices: jnp.ndarray,
                             masses: jnp.ndarray,
                             velocity: jnp.ndarray) -> jnp.ndarray:
    # angles = 2*jnp.pi*jrd.beta(key, transfer_intensity,
    #                           transfer_intensity, shape=(pairs_to_transfer.shape[0],))
    # angles = 2*jnp.arctan(transfer_intensity*distances[pairs_to_transfer[:, 0],
    #                                                   pairs_to_transfer[:, 1]])
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
    diff = position - jnp.transpose(position_tiled, (1, 0, 2))
    diff = jnp.mod(diff + 0.5, 1) - 0.5
    dist_mat = jnp.sqrt(jnp.sum(diff**2, axis=-1))

    potentials = potential(peak_gathered,
                           close_gathered,
                           trough_gathered,
                           far_gathered,
                           dist_mat)

    # accelerations = vmap(partial(potential_serial,
    #                             potential_peak,
    #                             potential_close,
    #                             potential_trough,
    #                             potential_far,
    #                             particle_type_table,
    #                             masses,
    #                             position))(jnp.arange(position.shape[0]))

    # changing the axis of the summation here will reverse the sign of the force
    accelerations = jnp.sum(
        potentials[..., jnp.newaxis]*diff, axis=0)/masses[particle_type_table, jnp.newaxis]

    new_velocity = velocity + dt*accelerations
    speed = jnp.sqrt(jnp.sum(new_velocity**2, axis=-1))[..., jnp.newaxis]
    new_velocity = jnp.where(
        speed > speed_limit, speed_limit*new_velocity/speed, new_velocity)
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    return jnp.concatenate((new_position, new_velocity), axis=-1)


def make_simulation_step_stochastic(particle_type_table: jnp.ndarray,
                                    potential_peak: jnp.ndarray,
                                    potential_close: jnp.ndarray,
                                    potential_trough: jnp.ndarray,
                                    potential_far: jnp.ndarray,
                                    masses: jnp.ndarray,
                                    dt: float,
                                    n_transfers_per_step: int,
                                    transfer_intensity: float) -> Callable:

    def get_diff_matrix(position: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        position_tiled = jnp.tile(
            position[jnp.newaxis], (position.shape[0], 1, 1))
        diffs = get_diff_wrapped_jax(
            position_tiled, jnp.transpose(position_tiled, (1, 0, 2)))
        distances = jnp.linalg.norm(diffs, axis=-1)
        return distances, diffs

    get_diff_matrix_jitted = jit(get_diff_matrix)

    def sim_step_jittable(key: jrd.PRNGKey,
                          position: jnp.ndarray,
                          velocity: jnp.ndarray,
                          diffs: jnp.ndarray,
                          distances: jnp.ndarray,
                          sampled_neighbors: jnp.ndarray,
                          unsampled_indices: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jrd.PRNGKey]:

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

        potentials = potential(peak_gathered,
                               close_gathered,
                               trough_gathered,
                               far_gathered,
                               distances)

        masses_gathered = masses[particle_type_table, jnp.newaxis]

        # changing the axis of the summation here will reverse the sign of the force
        accelerations = jnp.sum(
            potentials[..., jnp.newaxis]*diffs, axis=0)/masses_gathered

        new_velocity = velocity + dt*accelerations
        mean_vel1 = jnp.max(jnp.linalg.norm(new_velocity, axis=-1))
        new_velocity = random_momentum_transfer(key,
                                                transfer_intensity,
                                                distances,
                                                sampled_neighbors,
                                                unsampled_indices,
                                                masses_gathered,
                                                new_velocity)
        mean_vel2 = jnp.max(jnp.linalg.norm(new_velocity, axis=-1))
        new_position = position + dt*new_velocity
        new_position = jnp.mod(new_position, 1)

        return jnp.concatenate((new_position, new_velocity), axis=-1), mean_vel1, mean_vel2

    sim_step_jitted = jit(sim_step_jittable)

    def sim_step(key: jrd.PRNGKey,
                 sim_state: jnp.ndarray) -> Tuple[jnp.ndarray, jrd.PRNGKey]:
        choice_key, beta_key, next_key = jrd.split(key, num=3)

        position, velocity = sim_state[:, :2], sim_state[:, 2:]
        distances, diffs = get_diff_matrix_jitted(position)

        sampled_neighbors = jrd.choice(choice_key,
                                       distances.shape[0],
                                       shape=(n_transfers_per_step, 2),
                                       replace=False)
        # in order to construct the sparse rotation matrix we have to keep track
        # of which indices are not sampled so that we can apply the identity matrix
        # to them
        unsampled_indices = list(range(position.shape[0]))
        sampled_indices = set()
        for edge in np.array(sampled_neighbors):
            sampled_indices.add(edge[0])
            sampled_indices.add(edge[1])
        unsampled_indices = [i for i in unsampled_indices
                             if i not in sampled_indices]
        unsampled_indices = jnp.array(unsampled_indices)

        next_state, v1, v2 = sim_step_jitted(beta_key,
                                             position,
                                             velocity,
                                             diffs,
                                             distances,
                                             sampled_neighbors,
                                             unsampled_indices)

        return next_key, next_state, v1, v2

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
        len_type0 = int(np.sqrt(n_particles_per_type[0]))
        position_type0 = np.stack(np.meshgrid(
            np.linspace(0, 1, len_type0, endpoint=False),
            np.linspace(0, 1, len_type0, endpoint=False),
            indexing='ij'), axis=-1).reshape(-1, 2)

        n_other_particles = sum(n_particles_per_type[1:])
        position_other = core_size * \
            (jrd.uniform(seed1, (n_other_particles, 2)) - 0.5) + 0.5

        position = np.concatenate((position_type0, position_other), axis=0)

        velocity_type0 = jnp.zeros_like(position_type0)
        velocity_other = max_init_speed * \
            jrd.uniform(seed2, (n_other_particles, 2), minval=-1, maxval=1)
        velocity = np.concatenate((velocity_type0, velocity_other), axis=0)

    else:
        raise ValueError(
            f"Unknown initialization mode: {initialization_mode}. Supported modes are 'uniform' and 'core'.")

    return jnp.concatenate([position, velocity], axis=-1)


def get_diff_wrapped(p1, p2):
    """Get the direction vector between two points, wrapped around the unit square."""
    diff = p1 - p2
    diff = np.mod(diff + 0.5, 1) - 0.5
    return diff


def get_diff_wrapped_jax(p1, p2):
    """Get the direction vector between two points, wrapped around the unit square."""
    diff = p1 - p2
    diff = jnp.mod(diff + 0.5, 1) - 0.5
    return diff


def simulation_init_sparse(max_init_speed: float,
                           n_particles_per_type: List[int],
                           core_size: float,
                           potential_far: jnp.ndarray,
                           particle_type_table: jnp.ndarray,
                           seed: jrd.PRNGKey,
                           initialization_mode: str) -> Tuple[jnp.ndarray, sparse.BCOO, jnp.ndarray]:
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

    def get_ixs_diffs_dist(i):
        # compute the direction vectors of all neighbors close enough together
        diff = get_diff_wrapped(position[i], position)
        dist = np.linalg.norm(diff, axis=1)
        # store the indices and distances in a COO sparse matrix
        ix = np.where(np.logical_and(dist < potential_far[particle_type_table[i]][particle_type_table],
                                     dist > 0))[0]
        diff = diff[ix]
        dist = dist[ix]
        ix = np.concatenate(
            (np.repeat(i, ix.shape[0])[:, np.newaxis], ix[:, np.newaxis]), axis=-1)
        return ix, diff, dist

    # use cpu to find initial neighbor indices
    # neighbor_data = Parallel(n_jobs=3)(delayed(get_ixs_diffs_dist)(i) for i in range(tot_particles))
    neighbor_data = [get_ixs_diffs_dist(i) for i in range(tot_particles)]
    neighbor_data = [x for x in neighbor_data if len(x[0]) > 0]
    # store the indices and distances in a COO sparse matrix
    neighbor_matrix = sparse.BCOO((np.concatenate([x[2] for x in neighbor_data], axis=0),
                                   np.concatenate([x[0] for x in neighbor_data], axis=0)),
                                  shape=(tot_particles, tot_particles))
    # and keep the direction vectors separate
    diffs = np.concatenate([x[1] for x in neighbor_data], axis=0)

    return jnp.concatenate([position, velocity], axis=-1), neighbor_matrix, diffs


def simulation_step_sparse(particle_type_table: jnp.ndarray,
                           potential_peak: jnp.ndarray,
                           potential_close: jnp.ndarray,
                           potential_trough: jnp.ndarray,
                           potential_far: jnp.ndarray,
                           masses: jnp.ndarray,
                           dt: float,
                           depth: int,
                           sim_state: jnp.ndarray,
                           neighbor_matrix: sparse.BCOO,
                           diffs: jnp.ndarray) -> Tuple[jnp.ndarray, sparse.BCOO, jnp.ndarray]:
    position, velocity = sim_state[:, :2], sim_state[:, 2:]

    type_edges = particle_type_table[neighbor_matrix.indices]

    force_vectors = compute_force_sparse(len(particle_type_table),
                                         potential_peak,
                                         potential_close,
                                         potential_trough,
                                         potential_far,
                                         neighbor_matrix.indices,
                                         type_edges,
                                         neighbor_matrix.data,
                                         diffs)

    acceleration = force_vectors/masses[particle_type_table, jnp.newaxis]
    new_velocity = velocity + dt*acceleration
    new_position = position + dt*new_velocity
    new_position = jnp.mod(new_position, 1)

    new_neighbor_matrix = jnp.copy(neighbor_matrix)
    for i in range(depth):
        new_neighbor_matrix = neighbor_matrix@new_neighbor_matrix
    new_dists = jnp.linalg.norm(get_diff_wrapped(position[new_neighbor_matrix.indices[:, 0]],
                                                 position[new_neighbor_matrix.indices[:, 1]]), axis=1)
    indices = jnp.where(jnp.logical_and(new_dists < potential_far[particle_type_table[new_neighbor_matrix.indices[:, 0]]]
                                                                 [particle_type_table[new_neighbor_matrix.indices[:, 1]]],
                                        new_dists > 0))
    new_diffs = get_diff_wrapped_jax(position[new_neighbor_matrix.indices[indices, 0]],
                                     position[new_neighbor_matrix.indices[indices, 1]])
    new_neighbor_matrix = sparse.BCOO((new_dists[indices],
                                       new_neighbor_matrix.indices[indices]),
                                      shape=(position.shape[0], position.shape[0]))

    return jnp.concatenate((new_position, new_velocity), axis=-1), new_neighbor_matrix, new_diffs
