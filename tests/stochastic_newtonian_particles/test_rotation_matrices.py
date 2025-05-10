import jax.numpy as jnp
import jax.random as jrd

from typing import *
from jax.experimental import sparse

from stochastic_newtonian_particles.simulation_utils import *


def test_matrix(pairs: jnp.ndarray,
                unsampled_indices: jnp.ndarray,
                angles: jnp.ndarray,
                n_particles: int,
                test_momentum_vectors: jnp.ndarray) -> Tuple[float, float]:

    print(unsampled_indices.shape, n_particles)
    matrix = build_rotation_matrix(pairs, unsampled_indices, angles, n_particles)

    identity_test: sparse.BCOO = matrix.T@matrix
    identity_test_frobenius = jnp.linalg.norm(identity_test.todense() -jnp.eye(n_particles))

    sum_x = jnp.sum(test_momentum_vectors[:, 0])
    sum_y = jnp.sum(test_momentum_vectors[:, 1])
    flattened = jnp.reshape(test_momentum_vectors, (-1,))
    transformed = jnp.reshape(matrix@flattened, (-1, 2))
    transformed_sum_x = jnp.sum(transformed[:, 0])
    transformed_sum_y = jnp.sum(transformed[:, 1])

    max_diff = jnp.maximum(jnp.abs(sum_x - transformed_sum_x),
                           jnp.abs(sum_y - transformed_sum_y))

    return identity_test_frobenius, max_diff


if __name__ == '__main__':

    key = jrd.PRNGKey(24)
    k1, k2, k3 = jrd.split(key, num=3)

    N = 10
    pairs = jrd.choice(k1, jnp.arange(N), (2, 2), replace=False)
    unsampled_indices = jnp.array([i for i in range(N) if i not in np.array(pairs)])
    angles = jrd.uniform(k1, (5,))
    momenta = jrd.uniform(k1, (N, 2), minval=-1, maxval=1)
    print(test_matrix(pairs, unsampled_indices, angles, N, momenta))

    N = 100
    pairs = jrd.choice(k1, jnp.arange(N), (32, 2), replace=False)
    unsampled_indices = jnp.array([i for i in range(N) if i not in np.array(pairs)])
    angles = jrd.uniform(k1, (5,))
    momenta = jrd.uniform(k1, (N, 2), minval=-1, maxval=1)
    print(test_matrix(pairs, unsampled_indices, angles, N, momenta))

    N = 1000
    pairs = jrd.choice(k1, jnp.arange(N), (128, 2), replace=False)
    unsampled_indices = jnp.array([i for i in range(N) if i not in np.array(pairs)])
    angles = jrd.uniform(k1, (5,))
    momenta = jrd.uniform(k1, (N, 2), minval=-1, maxval=1)
    print(test_matrix(pairs, unsampled_indices, angles, N, momenta))