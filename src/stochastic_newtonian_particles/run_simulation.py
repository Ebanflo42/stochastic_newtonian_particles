import os
import yaml
import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jrd

from typing import *
from tqdm import tqdm
from functools import partial
from easydict import EasyDict
from jax import jit, disable_jit

from stochastic_newtonian_particles.simulation_utils import *
from stochastic_newtonian_particles.visualization_utils import *


def run_simulation_loop(config: EasyDict) -> np.ndarray:
    print("Running simulation...")

    # Create an array indicating the type of each particle
    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(config.n_particles_per_type)])

    # Initialize positions and velocities
    key = jrd.PRNGKey(config.seed)
    init_key, sim_key = jrd.split(key, num=2)
    sim_state = simulation_init(config.max_init_speed,
                                config.n_particles_per_type,
                                config.core_size,
                                init_key,
                                config.initialization_mode)

    # Initialize simulation history
    simulation_history = np.zeros(
        (config.num_steps, sum(config.n_particles_per_type), 4))
    simulation_history[0] = sim_state

    # Compile the simulation step function to improve performance
    # disable_jit()
    if config.stochastic:
        step = partial(simulation_step_stochastic,
                       jnp.array(particle_type_table),
                       jnp.array(config.potential_peak),
                       jnp.array(config.potential_close),
                       jnp.array(config.potential_trough),
                       jnp.array(config.potential_far),
                       jnp.array(config.masses),
                       config.speed_limit,
                       config.dt,
                       config.n_transfers_per_step,
                       config.transfer_intensity)
    else:
        step = jit(partial(simulation_step_deterministic,
                           jnp.array(particle_type_table),
                           jnp.array(config.potential_peak),
                           jnp.array(config.potential_close),
                           jnp.array(config.potential_trough),
                           jnp.array(config.potential_far),
                           jnp.array(config.masses),
                           config.speed_limit,
                           config.dt))

    for t in tqdm(range(1, config.num_steps)):
        if config.stochastic:
            sim_state, sim_key = step(sim_key, sim_state)
        else:
            sim_state = step(sim_state)
        simulation_history[t] = sim_state

    return simulation_history


def run_simulation_main(config: EasyDict):
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
        with open(os.path.join(config.results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    elif not config.extend and \
            os.path.exists(os.path.join(config.results_dir, 'simulation_hisory.npy')):
        raise FileExistsError(
            f"Results directory {config.results_dir} already exists with simulation_history.npy. Use --extend to continue.")

    simulation_history = run_simulation_loop(config)
    save_simulation_mp4(config.results_dir,
                        'simulation.mp4',
                        config.n_particles_per_type,
                        config.palette,
                        simulation_history)


def run_simulation_entry():
    parser = argparse.ArgumentParser(description="Run simulation")

    # command line arguments will override yaml configs
    parser.add_argument("--config", type=str, default="sim_config.yaml")
    parser.add_argument("--results_dir", type=str, default="", required=True)
    parser.add_argument("--extend", type=bool, default=False)

    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--speed_limit", type=float, default=None)
    parser.add_argument("--max_init_speed", type=float, default=None)
    parser.add_argument("--initialization_mode", type=str, default=None)
    parser.add_argument("--stochastic", type=bool, default=None)
    parser.add_argument("--n_transfers_per_step", type=int, default=None)
    parser.add_argument("--transfer_intensity", type=float, default=None)

    args = parser.parse_args()

    with open(args.__dict__['config'], 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = EasyDict(yaml_config)
    config.extend = args.extend
    if not config.extend:
        for key, value in args.__dict__.items():
            if value is not None:
                config[key] = value

    run_simulation_main(config)


if __name__ == "__main__":
    with open("sim_config.yaml", 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = EasyDict(yaml_config)
    config.results_dir = "results/debug"
    config.extend = False
    run_simulation_main(config)
