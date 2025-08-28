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
from jax import jit
#jax.config.update('jax_disable_jit', True)

from stochastic_newtonian_particles.simulation_utils import *
from stochastic_newtonian_particles.visualization_utils import *


def run_simulation_loop(config: EasyDict) -> np.ndarray:
    print("Running simulation...")

    # Create an array indicating the type of each particle
    particle_type_table = np.concatenate(
        [[i] * n for i, n in enumerate(config.n_particles_per_type)]).astype(np.int32)

    key = jrd.PRNGKey(config.seed)

    if config.potential_name == "diff_gaussians":
        potential_fun = diff_gaussians
    elif config.potential_name == "lennard_jones":
        potential_fun = lennard_jones
    elif config.potential_name == "quadratic":
        potential_fun = quadratic
    else:
        raise ValueError(
            f"Unknown potential name {config.potential_name}. Use 'diff_gaussians' or 'lennard_jones'.")

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

    # the simulation step will have a different type signature
    # depending on whether the simulation is deterministic or stochastic
    # and whether or not we log the potential energy/momentum
    if config.stochastic:
        if config.log_energy:
            sim_step = \
                make_simulation_step_stochastic_energy_log(
                    jnp.array(particle_type_table),
                    jnp.array(config.potential_peak),
                    jnp.array(config.potential_trough, dtype=jnp.float32),
                    jnp.array(config.potential_close),
                    jnp.array(config.potential_far),
                    jnp.array(config.masses),
                    config.dt,
                    config.transfer_intensity,
                    config.n_transfers_per_step,
                    potential_fun)
        else:
            sim_step = \
                make_simulation_step_stochastic(
                    jnp.array(particle_type_table),
                    jnp.array(config.potential_peak),
                    jnp.array(config.potential_trough, dtype=jnp.float32),
                    jnp.array(config.potential_close),
                    jnp.array(config.potential_far),
                    jnp.array(config.masses),
                    config.dt,
                    config.transfer_intensity,
                    config.n_transfers_per_step,
                    potential_fun)
    else:
        if config.log_energy:
            sim_step = \
                jit(partial(simulation_step_deterministic_energy_log,
                            jnp.array(particle_type_table),
                            jnp.array(config.potential_peak),
                            jnp.array(config.potential_trough,
                                      dtype=jnp.float32),
                            jnp.array(config.potential_close),
                            jnp.array(config.potential_far),
                            jnp.array(config.masses),
                            config.dt,
                            config.speed_limit,
                            potential_fun))
        else:
            sim_step = \
                jit(partial(simulation_step_deterministic,
                            jnp.array(particle_type_table),
                            jnp.array(config.potential_peak),
                            jnp.array(config.potential_trough,
                                      dtype=jnp.float32),
                            jnp.array(config.potential_close),
                            jnp.array(config.potential_far),
                            jnp.array(config.masses),
                            config.dt,
                            config.speed_limit,
                            potential_fun))

    if config.stochastic:
        if config.log_energy:
            tot_potential_energy = np.zeros((config.num_steps,))
            for t in tqdm(range(init_t, config.num_steps)):
                key, sim_state, potential = sim_step(key, sim_state)
                simulation_history[t] = sim_state
                tot_potential_energy[t] = potential
            return simulation_history, tot_potential_energy
        else:
            for t in tqdm(range(init_t, config.num_steps)):
                key, sim_state = sim_step(key, sim_state)
                simulation_history[t] = sim_state
            return simulation_history
    else:
        if config.log_energy:
            tot_potential_energy = np.zeros((config.num_steps,))
            for t in tqdm(range(init_t, config.num_steps)):
                sim_state, potential = sim_step(sim_state)
                simulation_history[t] = sim_state
                tot_potential_energy[t] = potential
            return simulation_history, tot_potential_energy
        else:
            for t in tqdm(range(init_t, config.num_steps)):
                sim_state = sim_step(sim_state)
                simulation_history[t] = sim_state
                #print(np.all(sim_state == sim_state))
            return simulation_history


def run_simulation_main(config: EasyDict):
    # I am actually too lazy to write out 8x8 matrices of potential paramters
    # so I do this instead
    #config.potential_far = np.array(config.potential_far)
    #config.potential_far = \
    #    np.concatenate([np.concatenate([config.potential_far, 2*config.potential_far], axis=-1),
    #                    np.concatenate([2*config.potential_far.T, config.potential_far], axis=-1)], axis=0)
    #config.potential_trough = np.array(config.potential_trough)
    #config.potential_trough = \
    #    np.concatenate([np.concatenate([config.potential_trough, 2*config.potential_trough], axis=-1),
    #                    np.concatenate([2*config.potential_trough.T, config.potential_trough], axis=-1)], axis=0)
    #config.potential_peak = np.array(config.potential_peak)
    #config.potential_peak = \
    #    np.concatenate([np.concatenate([config.potential_peak, config.potential_peak], axis=-1),
    #                    np.concatenate([config.potential_peak.T, config.potential_peak], axis=-1)], axis=0)
    #config.potential_close = np.array(config.potential_close)
    #config.potential_close = \
    #    np.concatenate([np.concatenate([0.25*config.potential_far[:4, :4], config.potential_close], axis=-1),
    #                    np.concatenate([config.potential_close.T, 0.25*config.potential_far[4:, 4:]], axis=-1)], axis=0)
    #config.potential_close = 0.5*config.potential_far

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
        tot_potential_energy = simulation_history[1]
        simulation_history = simulation_history[0]
        np.save(os.path.join(config.results_dir, 'tot_potential_energy.npy'),
                tot_potential_energy)
        plot_energy_and_momentum(np.array(simulation_history),
                                 np.array(tot_potential_energy),
                                 np.array(config.masses),
                                 config.n_particles_per_type,
                                 os.path.join(config.results_dir,
                                              'energy_and_momentum.png'))
        plot_velocity_histograms(np.array(simulation_history),
                                 os.path.join(config.results_dir,
                                              'velocity_histograms.png'))

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
    with open("sim_config_4types.yaml", 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = EasyDict(yaml_config)
    config.results_dir = "results/debug"
    config.extend = False
    # config.num_steps = 10000
    run_simulation_main(config)
