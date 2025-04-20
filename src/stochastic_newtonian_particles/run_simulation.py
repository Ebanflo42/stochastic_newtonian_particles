import os
import jax
import yaml
import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jrd
# jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_bcoo_cusparse_lowering", True)
# jax.config.update("jax_array_garbage_collection_guard", "allow")

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
        [[i] * n for i, n in enumerate(config.n_particles_per_type)]).astype(np.int32)

    key = jrd.PRNGKey(config.seed)

    # Initialize simulation history
    n_particles = sum(config.n_particles_per_type)
    if config.extend:
        simulation_history = np.load(os.path.join(
            config.results_dir, 'simulation_history.npy'))
        init_t = len(simulation_history)
        # Extend the simulation history
        if simulation_history.shape[0] < config.num_steps:
            sim_state = simulation_history[-1]
            simulation_history = np.concatenate(
                [simulation_history, np.zeros((config.num_steps - simulation_history.shape[0], n_particles, 4))])
        else:
            raise ValueError(
                f"Simulation history already has {simulation_history.shape[0]} steps, cannot extend to {config.num_steps}.")
    else:
        init_t = 1
        simulation_history = np.zeros(
            (config.num_steps, n_particles, 4))
        sim_state = simulation_init(config.max_init_speed,
                                    config.n_particles_per_type,
                                    config.core_size,
                                    key,
                                    config.initialization_mode)
        simulation_history[0] = sim_state

    force_max = (jnp.array(config.potential_peak) -
                 jnp.array(config.potential_trough)) / jnp.array(config.potential_close)
    force_min = jnp.array(config.potential_trough) / \
        (jnp.array(config.potential_far) - jnp.array(config.potential_close))

    if config.stochastic:
        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        # disable_jit()
        step = \
            make_simulation_step_stochastic(jnp.array(particle_type_table),
                                            force_max,
                                            jnp.array(config.potential_close),
                                            force_min,
                                            jnp.array(config.potential_far),
                                            jnp.array(config.masses),
                                            config.dt,
                                            config.n_transfers_per_step,
                                            config.transfer_intensity)
        for t in tqdm(range(init_t, config.num_steps)):
            key, sim_state = step(key, sim_state)
            simulation_history[t] = sim_state
        return simulation_history
    else:
        if config.log_energy:
            potential_energy = np.zeros((config.num_steps, n_particles))
            step = jit(partial(simulation_step_deterministic_energy_log,
                               jnp.array(particle_type_table),
                               force_max,
                               jnp.array(config.potential_close),
                               force_min,
                               jnp.array(config.potential_far),
                               jnp.array(config.masses),
                               config.speed_limit,
                               config.dt))
            for t in tqdm(range(init_t, config.num_steps)):
                sim_state, potential = step(sim_state)
                simulation_history[t] = sim_state
                potential_energy[t] = potential
            return simulation_history, potential_energy
        else:
            step = jit(partial(simulation_step_deterministic,
                               jnp.array(particle_type_table),
                               force_max,
                               jnp.array(config.potential_close),
                               force_min,
                               jnp.array(config.potential_far),
                               jnp.array(config.masses),
                               config.speed_limit,
                               config.dt))
            for t in tqdm(range(init_t, config.num_steps)):
                sim_state = step(sim_state)
                simulation_history[t] = sim_state
            return simulation_history


def run_simulation_main(config: EasyDict):
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
        with open(os.path.join(config.results_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config.__dict__, f)
    elif not config.extend and \
            os.path.exists(os.path.join(config.results_dir, 'simulation_history.npy')):
        raise FileExistsError(
            f"Results directory {config.results_dir} already exists with simulation_history.npy. Use --extend to continue.")

    simulation_history = run_simulation_loop(config)

    if config.log_energy:
        potential_energy = simulation_history[1]
        simulation_history = simulation_history[0]
        np.save(os.path.join(config.results_dir, 'potential_energy.npy'),
                potential_energy)
        plot_energy_and_momentum(np.array(simulation_history),
                                 np.array(potential_energy),
                                 np.array(config.masses),
                                 config.n_particles_per_type,
                                 os.path.join(config.results_dir,
                                              'energy_and_momentum.png'))

    np.save(os.path.join(config.results_dir, 'simulation_history.npy'),
            simulation_history)
    save_simulation_mp4(config.results_dir,
                        'simulation.mp4',
                        config.n_particles_per_type,
                        config.palette,
                        simulation_history)


def run_simulation_entry():
    parser = argparse.ArgumentParser(description="Run simulation")

    # command line arguments will override yaml configs
    parser.add_argument("--config", type=str, default=None,)
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

    if args.__dict__['config'] is not None:
        with open(args.__dict__['config'], 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = EasyDict(yaml_config)
        for key, value in args.__dict__.items():
            if value is not None:
                config[key] = value
    elif args.__dict__['extend']:
        results_dir = args.__dict__['results_dir']
        if not os.path.exists(results_dir):
            raise FileNotFoundError(
                f"Results directory {results_dir} does not exist.")
        with open(os.path.join(results_dir, 'config.yaml'), 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = EasyDict(yaml_config)
        for key, value in args.__dict__.items():
            if value is not None:
                config[key] = value
    else:
        raise ValueError(
            "Either config file or results directory must be provided.")

    run_simulation_main(config)


if __name__ == "__main__":
    with open("sim_config.yaml", 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = EasyDict(yaml_config)
    config.results_dir = "results/debug"
    config.extend = False
    run_simulation_main(config)
