#!/usr/bin/env python3

import sys
import polars as pl
from pyter.infer import Inference
from pyter.data import HalfLifeData, TiterData
from pyter.models import (
    MultiphaseHalfLifeModel,
    HalfLifeModel,
    TiterModel)
import numpyro.distributions as dist
import numpy as np
import pickle


def read_data(data_path):
    return pl.read_csv(
        data_path,
        separator="\t")


def get_experiment_index(data):
    return data.unique(
        subset=["condition"]
    ).select(
        ["condition",
         "strain",
         "surface",
         "temperature",
         "relative_humidity_pct",
         "medium",
         "medium_state",
         "concentration_pct",
         "chlorine_concentration_ppm"
         ]
    ).with_columns(
        [
            pl.when(
                pl.col("temperature") < 5
            ).then(
                5
            ).when(
                pl.col("medium_state") == "liquid"
            ).then(
                1000
            ).otherwise(
                3
            ).alias("drying_low"),

            pl.when(
                pl.col("temperature") < 5
            ).then(
                10
            ).when(
                pl.col("medium_state") == "liquid"
            ).then(
                1010
            ).otherwise(
                5
            ).alias("drying_high"),
        ]
    ).sort(["condition"])


def model_factory(
        output_path,
        data,
        verbose=False):

    if output_path.endswith("titers.pickle"):
        seed = 87223
        m_data = TiterData(
            well_status=data["plaque_count"].to_numpy(),
            well_dilution=data["log10_dilution"].to_numpy(),
            well_titer_id=data["titer_id"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=0,
            log_base=10)
        model = TiterModel(
            log_titer_prior=dist.Normal(loc=3, scale=3),
            assay="pfu")
    elif output_path.endswith("halflives-surface.pickle"):
        seed = 5043
        data = data.filter(
            pl.col("medium_state") == "surface")
        experiment_index = get_experiment_index(
            data)
        m_data = HalfLifeData(
            well_status=data["plaque_count"].to_numpy(),
            well_dilution=data["log10_dilution"].to_numpy(),
            well_titer_id=data["titer_id"].to_numpy(),
            well_titer_error_scale_id=data["condition"].to_numpy(),
            well_halflife_id=data["condition"].to_numpy(),
            well_halflife_loc_id=data["condition"].to_numpy(),
            well_halflife_scale_id=data["condition"].to_numpy(),
            well_intercept_id=data["timeseries_id"].to_numpy(),
            well_intercept_loc_id=data["condition"].to_numpy(),
            well_intercept_scale_id=data["condition"].to_numpy(),
            well_time=data["timepoint"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=1e-20,
            log_base=10)

        model = MultiphaseHalfLifeModel(
            log_halflife_distribution=dist.Normal(
                loc=np.log(0.25),
                scale=np.log(20)),
            log_halflife_loc_prior=dist.Normal(
                loc=np.log(0.25),
                scale=np.log(20)),
            log_halflife_scale_prior=dist.TruncatedNormal(
                loc=0,
                scale=0.25,
                low=0),

            log_intercept_distribution=dist.Normal,
            log_intercept_loc_prior=dist.Normal(4, 2),
            log_intercept_scale_prior=dist.TruncatedNormal(
                low=0.0,
                loc=1.0,
                scale=0.25),

            log_titer_error_distribution=dist.Normal,

            log_titer_error_scale_prior=dist.TruncatedNormal(
                loc=0,
                scale=1,
                low=0.0),

            log_halflife_offset_prior=dist.TruncatedNormal(
                loc=0,
                scale=1.5,
                high=0),

            breakpoint_delta_prior=dist.TruncatedNormal(
                low=experiment_index["drying_low"].to_numpy(),
                high=experiment_index["drying_high"].to_numpy(),
                loc=(experiment_index["drying_high"].to_numpy() +
                     experiment_index["drying_low"].to_numpy())/2,
                scale=2),
            intercepts_hier=True,
            halflives_hier=False,
            titers_overdispersed=True,
            assay="pfu",
            n_phases=2)
        pass
    elif output_path.endswith("halflives-liquid.pickle"):
        seed = 50039
        data = data.filter(
            pl.col("medium_state") == "liquid")
        m_data = HalfLifeData(
            well_status=data["plaque_count"].to_numpy(),
            well_dilution=data["log10_dilution"].to_numpy(),
            well_titer_id=data["titer_id"].to_numpy(),
            well_titer_error_scale_id=data[
                "intercept_condition"].to_numpy(),
            well_halflife_id=data["condition"].to_numpy(),
            well_intercept_id=data["timeseries_id"].to_numpy(),
            well_intercept_loc_id=data[
                "intercept_condition"].to_numpy(),
            well_intercept_scale_id=data[
                "intercept_condition"].to_numpy(),
            well_time=data["timepoint"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=1e-20,
            log_base=10)
        model = HalfLifeModel(
            log_halflife_distribution=dist.Normal(
                loc=np.log(0.1),
                scale=np.log(20)),

            log_intercept_distribution=dist.Normal,
            log_intercept_loc_prior=dist.Normal(3, 2),
            log_intercept_scale_prior=dist.TruncatedNormal(
                low=0.0,
                loc=1.0,
                scale=0.25),

            log_titer_error_distribution=dist.Normal,
            log_titer_error_scale_prior=dist.TruncatedNormal(
                loc=0,
                scale=0.5,
                low=0.0),

            assay="pfu",
            intercepts_hier=True,
            halflives_hier=False,
            titers_overdispersed=True)

    elif output_path.endswith("halflives-liquid-hier.pickle"):
        seed = 253235
        data = data.filter(
            pl.col("medium_state") == "liquid")
        m_data = HalfLifeData(
            well_status=data["plaque_count"].to_numpy(),
            well_dilution=data["log10_dilution"].to_numpy(),
            well_titer_id=data["titer_id"].to_numpy(),
            well_titer_error_scale_id=data["condition"].to_numpy(),
            well_halflife_id=data["timeseries_id"].to_numpy(),
            well_halflife_loc_id=data["condition"].to_numpy(),
            well_halflife_scale_id=data["condition"].to_numpy(),
            well_intercept_id=data["timeseries_id"].to_numpy(),
            well_intercept_loc_id=data["condition"].to_numpy(),
            well_intercept_scale_id=data["condition"].to_numpy(),
            well_time=data["timepoint"].to_numpy(),
            well_volume=data["well_volume_ml"].to_numpy(),
            false_hit_rate=1e-20,
            log_base=10)
        model = HalfLifeModel(
            log_halflife_distribution=dist.Normal,
            log_halflife_loc_prior=dist.Normal(
                loc=np.log(0.25),
                scale=np.log(20)),
            log_halflife_scale_prior=dist.TruncatedNormal(
                loc=0,
                scale=0.25,
                low=0),

            log_intercept_distribution=dist.Normal,
            log_intercept_loc_prior=dist.Normal(4, 2),
            log_intercept_scale_prior=dist.TruncatedNormal(
                low=0.0,
                loc=1.0,
                scale=0.25),

            log_titer_error_distribution=dist.Normal,
            log_titer_error_scale_prior=dist.TruncatedNormal(
                loc=0,
                scale=0.5,
                low=0.0),

            assay="pfu",
            intercepts_hier=True,
            halflives_hier=True,
            titers_overdispersed=True)
    else:
        raise ValueError("Unknown model to infer from")

    return seed, m_data, model


def main(data_path,
         output_path,
         strict=True):
    data = read_data(data_path)

    seed, m_data, model = model_factory(
        output_path,
        data,
        verbose=True)

    infer = Inference(
        target_accept_prob=0.97,
        max_tree_depth=12,
        forward_mode_differentiation=False)

    infer.infer(data=m_data,
                model=model,
                random_seed=seed)
    infer.mcmc_runner.print_summary()

    if strict:
        print("Checking for MCMC convergence problems...")
        if np.any(infer.mcmc_runner.get_extra_fields()['diverging']):
            raise ValueError("At least one divergent transition after "
                             "warmup. Exiting without saving results "
                             "because `strict` was set to `True`. "
                             "If you want to save results anyway for "
                             "diagnosis, set `strict = False`")
        else:
            print("No divergent transitions.\n")

    with open(output_path, "wb") as file:
        pickle.dump(infer, file)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: ./{} <data file> <output path>"
              "".format(sys.argv[0]))
    else:
        main(sys.argv[1], sys.argv[-1])
