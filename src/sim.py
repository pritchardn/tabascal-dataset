import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from coolname import generate_slug
from tabascal.dask.observation import Observation
from tabascal.utils.sky import generate_random_sky
from tabascal.utils.tools import load_antennas
from tqdm import tqdm


def get_default_config():
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
        "n_src": 10000,
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


def generate_obs_name(obs: Observation):
    return (
        f"{generate_slug(2)}_obs_{obs.n_ant:0>2}A_{obs.n_time:0>3}T-{int(obs.times[0]):0>4}-{int(obs.times[-1]):0>4}"
        + f"_{obs.n_int_samples:0>3}I_{obs.n_freq:0>3}F-{float(obs.freqs[0]):.3e}-{float(obs.freqs[-1]):.3e}"
        + f"_{obs.n_ast:0>3}AST_{obs.n_rfi_satellite}SAT_{obs.n_rfi_stationary}GRD"
    )


def setup_environment(
    t_0: float,
    dT: float,
    N_t: int,
    N_freq: int,
    freq_start: float,
    freq_end: float,
    output_dir: str,
):
    print("Setting up environment")
    rng = np.random.default_rng(12345)
    times = np.arange(t_0, t_0 + N_t * dT, dT)
    freqs = np.linspace(freq_start, freq_end, N_freq)
    os.makedirs(output_dir, exist_ok=True)
    return times, freqs, rng


def setup_observation(
    lat: float,
    lon: float,
    elevation: float,
    ra: float,
    dec: float,
    num_antenna: int,
    SEFD: float,
    times: np.ndarray,
    freqs: np.ndarray,
    n_int_samples: int,
    max_chunk_MB: float,
    rng: np.random.Generator,
):
    print("Setting up observation")
    ants_enu = rng.permutation(load_antennas("MeerKAT"))[:num_antenna]
    obs = Observation(
        latitude=lat,
        longitude=lon,
        elevation=elevation,
        ra=ra,
        dec=dec,
        times=times,
        freqs=freqs,
        SEFD=SEFD,
        ENU_array=ants_enu,
        n_int_samples=n_int_samples,
        max_chunk_MB=max_chunk_MB,
    )
    return obs


def setup_sky_model(
    obs: Observation, n_src: int, max_I: float, beam_width: float, seed: int
):
    print("Setting up sky model")
    I, d_ra, d_dec = generate_random_sky(
        n_src=n_src,
        min_I=np.mean(obs.noise_std) / 5.0,
        max_I=max_I,
        freqs=obs.freqs,
        fov=obs.fov,
        beam_width=beam_width,
        random_seed=seed,
    )
    return I, d_ra, d_dec


def add_astro_sources(
    obs: Observation, I: np.ndarray, d_ra: np.ndarray, d_dec: np.ndarray
):
    print("Adding astro sources")
    obs.addAstro(
        I=I[:, None, :] * np.ones((1, obs.n_time_fine, 1)),
        ra=obs.ra + d_ra,
        dec=obs.dec + d_dec,
    )


def add_satellite_rfi(obs: Observation, freqs: np.ndarray, N_sat: int, RFI_amp: float):
    # TODO: Make satellite rfi configurable
    print("Adding satellite RFI")

    rfi_P = np.array(
        [
            RFI_amp * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
            RFI_amp * 2 * 0.6e-4 * np.exp(-0.5 * ((freqs - 1.227e9) / 5e6) ** 2),
        ]
    )

    elevation = [20200e3, 19140e3]
    inclination = [55.0, 64.8]
    lon_asc_node = [21.0, 17.0]
    periapsis = [7.0, 1.0]

    obs.addSatelliteRFI(
        Pv=rfi_P[:N_sat, None, :] * np.ones((N_sat, obs.n_time_fine, obs.n_freq)),
        elevation=elevation[:N_sat],
        inclination=inclination[:N_sat],
        lon_asc_node=lon_asc_node[:N_sat],
        periapsis=periapsis[:N_sat],
    )


def add_stationary_rfi(obs: Observation, freqs: np.ndarray, N_grd: int, RFI_amp: float):
    # TODO: Make stationary rfi configurable
    print("Adding stationary RFI")

    rfi_P = np.array(
        [
            RFI_amp * 6e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            RFI_amp * 1.5e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
            RFI_amp * 0.4e-4 * np.exp(-0.5 * ((freqs - 1.22e9) / 3e6) ** 2),
        ]
    )
    latitude = [-20.0, -20.0, -25.0]
    longitude = [30.0, 20.0, 20.0]
    elevation = [obs.elevation, obs.elevation, obs.elevation]

    obs.addStationaryRFI(
        Pv=rfi_P[:N_grd, None, :] * np.ones((N_grd, obs.n_time_fine, obs.n_freq)),
        latitude=latitude[:N_grd],
        longitude=longitude[:N_grd],
        elevation=elevation[:N_grd],
    )


def add_gains(
    obs: Observation,
    G0_mean: float,
    G0_std: float,
    Gt_std_amp: float,
    Gt_std_phase: float,
):
    print("Adding gains")
    obs.addGains(G0_mean, G0_std, Gt_std_amp, Gt_std_phase)


def calculate_visibilities(obs: Observation):
    print("Calculating visibilities")
    obs.calculate_vis()


def convert_to_ml_dataset(
    obs: Observation, num_baselines: int, rng: np.random.Generator
):
    print("Converting to TF format")
    baselines = sorted(rng.choice(obs.n_bl, size=num_baselines, replace=False))
    vis = obs.vis_obs[:, baselines, :].compute()
    rfi = obs.vis_rfi[:, baselines, :].compute()

    data, masks = [], []

    for i in range(num_baselines):
        data.append(np.abs(vis[i]))
        masks.append(np.abs(rfi[i]) > 0)

    data = np.expand_dims(np.array(data), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)
    return data, masks


def save_observation(data, masks, filename, output_dir):
    print("Saving observation as TF dataset")
    save_path = os.path.join(output_dir, filename)
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump((data, masks), f, protocol=4)


def plot_examples(data, masks, output_dir, num_examples=5):
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


def main(config: dict):
    output_dir = "./outputs"
    times, freqs, rng = setup_environment(
        config["t_0"],
        config["dT"],
        config["N_t"],
        config["N_freq"],
        config["freq_start"],
        config["freq_end"],
        output_dir,
    )
    obs = setup_observation(
        config["latitude"],
        config["longitude"],
        config["elevation"],
        config["ra"],
        config["dec"],
        config["num_antenna"],
        config["SEFD"],
        times,
        freqs,
        config["n_int_samples"],
        config["max_chunk_MB"],
        rng,
    )
    I, d_ra, d_dec = setup_sky_model(
        obs,
        config["n_src"],
        config["max_I"],
        obs.syn_bw if obs.syn_bw < 1e-2 else 1e-2,
        12345,
    )
    add_astro_sources(obs, I, d_ra, d_dec)
    add_satellite_rfi(obs, freqs, config["num_satellites"], config["satellite_rfi_amp"])
    add_stationary_rfi(
        obs, freqs, config["num_ground_sources"], config["ground_rfi_amp"]
    )
    add_gains(
        obs,
        config["G0_mean"],
        config["G0_std"],
        config["Gt_std_amp"],
        config["Gt_std_amp"],
    )
    calculate_visibilities(obs)
    dataset_name = generate_obs_name(obs)
    config["dataset_name"] = dataset_name
    print(f"Dataset name: {dataset_name}")
    data, masks = convert_to_ml_dataset(obs, config["num_sample_baseline"], rng)
    save_observation(data, masks, dataset_name, output_dir)
    plot_examples(data, masks, output_dir)


if __name__ == "__main__":
    config = get_default_config()
    main(config)
