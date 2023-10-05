"""
MIT License

Copyright (c) 2023 Nicholas Pritchard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import os

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from coolname import generate_slug
from tabascal.dask.interferometry import time_avg
from tabascal.dask.observation import Observation
from tabascal.utils.sky import generate_random_sky
from tabascal.utils.tools import load_antennas
from tqdm import tqdm

from utils import load_config_file


def get_default_config() -> dict:
    """
    Returns the default config values
    """
    return {
        "t_0": 440.0,
        "dT": 1.17,
        "N_t": 512,
        "N_freq": 512,
        "freq_start": 1000e6,
        "freq_end": 10e9,
        "latitude": -30.0,
        "longitude": 21.0,
        "elevation": 1050.0,
        "ra": 27.0,
        "dec": 15.0,
        "num_antenna": 64,
        "SEFD": 420.0,
        "n_int_samples": 4,
        "max_chunk_MB": 100.0,
        "n_src": 1000,
        "max_I": 1.0,
        "num_satellites": 2,
        "satellite_rfi_amp": 1.0,
        "num_ground_sources": 3,
        "ground_rfi_amp": 1.0,
        "G0_mean": 1.0,
        "G0_std": 0.05,
        "Gt_std_amp": 1e-5,
        "Gt_std_phase": np.deg2rad(1e-3),
        "num_sample_baseline": 512,
    }


def generate_obs_name(
    obs: Observation, num_sample_baselines: int, with_slug=False
) -> str:
    """
    Generates an observation name
    """
    observation_name = (
        f"obs_{obs.n_ast:0>3}"
        + f"AST_{obs.n_rfi_satellite}"
        + f"SAT_{obs.n_rfi_stationary}"
        + f"GRD_{num_sample_baselines}"
        + f"BSL_{obs.n_ant:0>2}"
        + f"A_{obs.n_time:0>3}"
        + f"T-{int(obs.times[0]):0>4}"
        + f"-{int(obs.times[-1]):0>4}"
        + f"_{obs.n_int_samples:0>3}"
        + f"I_{obs.n_freq:0>3}"
        + f"F-{float(obs.freqs[0]):.3e}"
        + f"-{float(obs.freqs[-1]):.3e}"
    )
    if with_slug:
        observation_name += f"_{generate_slug(2)}"
    return observation_name


def setup_environment(
    t_0: float,
    integration_time: float,
    num_timesteps: int,
    num_frequencies: int,
    freq_start: float,
    freq_end: float,
    output_dir: str,
) -> tuple:
    """
    Sets up the environment for the simulation. Returns the times, frequencies
    and random number generator.
    """
    print("Setting up environment")
    rng = np.random.default_rng(12345)
    times = np.arange(t_0, t_0 + num_timesteps * integration_time, integration_time)
    freqs = np.linspace(freq_start, freq_end, num_frequencies)
    os.makedirs(output_dir, exist_ok=True)
    return times, freqs, rng


def setup_observation(
    lat: float,
    lon: float,
    elevation: float,
    ra: float,
    dec: float,
    num_antenna: int,
    system_flux: float,
    times: np.ndarray,
    frequencies: np.ndarray,
    n_int_samples: int,
    max_chunk_mb: float,
    rng: np.random.Generator,
) -> Observation:
    """
    Sets up the observation. Returns the Tabascal observation object.
    """
    print("Setting up observation")
    ants_enu = rng.permutation(load_antennas("MeerKAT"))[:num_antenna]
    obs = Observation(
        latitude=lat,
        longitude=lon,
        elevation=elevation,
        ra=ra,
        dec=dec,
        times=times,
        freqs=frequencies,
        SEFD=system_flux,
        ENU_array=ants_enu,
        n_int_samples=n_int_samples,
        max_chunk_MB=max_chunk_mb,
    )
    return obs


def setup_sky_model(
    obs: Observation, n_src: int, max_intensity: float, beam_width: float, seed: int
) -> tuple:
    """
    Sets up the sky model. Returns the sky model, and the RA and DEC offsets.
    """
    print("Setting up sky model")
    intensity, d_ra, d_dec = generate_random_sky(
        n_src=n_src,
        min_I=np.mean(obs.noise_std) / 5.0,
        max_I=max_intensity,
        freqs=obs.freqs,
        fov=obs.fov,
        beam_width=beam_width,
        random_seed=seed,
    )
    return intensity, d_ra, d_dec


def add_astro_sources(
    obs: Observation, intensity: np.ndarray, d_ra: np.ndarray, d_dec: np.ndarray
):
    """
    Adds the astro sources to the observation.
    """
    print("Adding astro sources")
    obs.addAstro(
        I=intensity[:, None, :] * np.ones((1, obs.n_time_fine, 1)),
        ra=obs.ra + d_ra,
        dec=obs.dec + d_dec,
    )


def add_satellite_rfi(
    obs: Observation, freqs: np.ndarray, num_satellites: int, rfi_amplitude: float
):
    """
    Adds satellite rfi to the observation.
    WARNING: Does not bound check the number of satellites.
    """
    # TODO: Make satellite rfi configurable
    print("Adding satellite RFI")

    rfi_position = np.array(
        [
            rfi_amplitude * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
            rfi_amplitude * 2 * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
        ]
    )

    elevation = [20200e3, 19140e3]
    inclination = [55.0, 64.8]
    lon_asc_node = [21.0, 17.0]
    periapsis = [7.0, 1.0]

    obs.addSatelliteRFI(
        Pv=rfi_position[:num_satellites, None, :]
        * np.ones((num_satellites, obs.n_time_fine, obs.n_freq)),
        elevation=elevation[:num_satellites],
        inclination=inclination[:num_satellites],
        lon_asc_node=lon_asc_node[:num_satellites],
        periapsis=periapsis[:num_satellites],
    )


def add_stationary_rfi(
    obs: Observation, freqs: np.ndarray, num_ground_sources: int, rfi_amplitude: float
):
    """
    Adds stationary rfi to the observation.
    WARNING: Does not bound check paremeters.
    """
    # TODO: Make stationary rfi configurable
    print("Adding stationary RFI")

    rfi_positions = np.array(
        [
            rfi_amplitude * 6e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            rfi_amplitude * 1.5e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            rfi_amplitude * 0.4e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
        ]
    )
    latitude = [-20.0, -20.0, -25.0]
    longitude = [30.0, 20.0, 20.0]
    elevation = [obs.elevation, obs.elevation, obs.elevation]

    obs.addStationaryRFI(
        Pv=rfi_positions[:num_ground_sources, None, :]
        * np.ones((num_ground_sources, obs.n_time_fine, obs.n_freq)),
        latitude=latitude[:num_ground_sources],
        longitude=longitude[:num_ground_sources],
        elevation=elevation[:num_ground_sources],
    )


def add_gains(
    obs: Observation,
    gain_0_mean: float,
    gain_0_std: float,
    gain_t_std_amp: float,
    gain_t_std_phase: float,
):
    """
    Adds gains to the observation.
    """
    print("Adding gains")
    obs.addGains(gain_0_mean, gain_0_std, gain_t_std_amp, gain_t_std_phase)


def calculate_visibilities(obs: Observation):
    """
    Calculates the visibilities for the observation.
    """
    print("Calculating visibilities")
    obs.calculate_vis()


def convert_to_ml_format(
    obs: Observation,
    num_baselines: int,
    rng: np.random.Generator,
    dataset_name,
    output_dir,
):
    """
    Converts the observation to the ML format. Saves the data to HDF5.
    """
    print("Converting data to TF format")
    baselines = sorted(rng.choice(obs.n_bl, size=num_baselines, replace=False))

    vis = obs.vis_obs[:, baselines, :].astype("float32")
    vis = np.abs(vis).astype("float32")
    vis = np.expand_dims(vis, axis=-1)
    masks = time_avg(obs.vis_rfi, obs.n_int_samples)
    masks = masks[:, baselines, :]
    masks = np.abs(masks).astype("float32")
    masks = np.expand_dims(masks, axis=-1)
    # masks = np.isclose(masks > 0, 1, 0).astype('bool')
    masks_16 = np.greater(masks, 16.0).astype("bool")
    masks_0 = np.greater(masks, 0.0).astype("bool")
    masks_1 = np.greater(masks, 1.0).astype("bool")
    masks_2 = np.greater(masks, 2.0).astype("bool")
    masks_4 = np.greater(masks, 4.0).astype("bool")
    masks_8 = np.greater(masks, 8.0).astype("bool")
    # masks = np.invert(np.isclose(masks, 0.0, rtol=1e-1).astype('bool'))

    print("Saving data to HDF5")
    output_filename = os.path.join(output_dir, f"{dataset_name}.hdf5")
    da.to_hdf5(
        output_filename,
        {
            "vis": vis,
            "masks_orig": masks,
            "masks_16": masks_16,
            "masks_0": masks_0,
            "masks_1": masks_1,
            "masks_2": masks_2,
            "masks_4": masks_4,
            "masks_8": masks_8,
        },
    )


def plot_examples(data, masks, output_dir, num_examples=5):
    """
    Plots examples of the data and masks.
    """
    print("Plotting examples")
    indicies = np.random.choice(data.shape[0], size=num_examples, replace=False)
    for i in tqdm(indicies):
        plt.clf()
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(data[i, :, :, 0], aspect="auto")
        plt.subplot(122)
        plt.imshow(masks[i, :, :, 0], aspect="auto")
        plt.savefig(os.path.join(output_dir, f"example_{i}.png"))


def run_simulation(config_dict: dict):
    """
    Runs the simulation.
    """
    output_dir = "./outputs"
    times, freqs, rng = setup_environment(
        config_dict["t_0"],
        config_dict["dT"],
        config_dict["N_t"],
        config_dict["N_freq"],
        config_dict["freq_start"],
        config_dict["freq_end"],
        output_dir,
    )
    obs = setup_observation(
        config_dict["latitude"],
        config_dict["longitude"],
        config_dict["elevation"],
        config_dict["ra"],
        config_dict["dec"],
        config_dict["num_antenna"],
        config_dict["SEFD"],
        times,
        freqs,
        config_dict["n_int_samples"],
        config_dict["max_chunk_MB"],
        rng,
    )
    intensity, d_ra, d_dec = setup_sky_model(
        obs,
        config_dict["n_src"],
        config_dict["max_I"],
        obs.syn_bw if obs.syn_bw < 1e-2 else 1e-2,
        12345,
    )
    add_astro_sources(obs, intensity, d_ra, d_dec)
    if config_dict["num_satellites"] > 0:
        add_satellite_rfi(
            obs, freqs, config_dict["num_satellites"], config_dict["satellite_rfi_amp"]
        )
    if config_dict["num_ground_sources"] > 0:
        add_stationary_rfi(
            obs, freqs, config_dict["num_ground_sources"], config_dict["ground_rfi_amp"]
        )
    add_gains(
        obs,
        config_dict["G0_mean"],
        config_dict["G0_std"],
        config_dict["Gt_std_amp"],
        config_dict["Gt_std_amp"],
    )
    calculate_visibilities(obs)
    dataset_name = generate_obs_name(obs, config_dict["num_sample_baseline"])
    config_dict["dataset_name"] = dataset_name
    print(f"Dataset name: {dataset_name}")
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    convert_to_ml_format(
        obs, config_dict["num_sample_baseline"], rng, dataset_name, dataset_output_dir
    )
    with open(
        os.path.join(dataset_output_dir, f"{dataset_name}_config.json"),
        "w",
        encoding="utf-8",
    ) as ofile:
        json.dump(config_dict, ofile)


if __name__ == "__main__":
    CONFIG_FILES = [
        "config_0SAT_0GRD.json",
        "config_1SAT_0GRD.json",
        "config_1SAT_3GRD.json",
        "config_2SAT_0GRD.json",
        "config_2SAT_3GRD.json",
    ]
    for config_file in CONFIG_FILES:
        print(config_file)
        config = load_config_file("./configs", config_file)
        run_simulation(config)
