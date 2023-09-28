#!/usr/bin/env python3

#############
# filename: analyze.py
# description: helper functions
# for analysis of results

import polars as pl
import numpy as np
import pickle
import constants as const

from pyter.infer import Inference


@pl.api.register_expr_namespace("exstr")
class ExtraStringMethods:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def capitalize(self):
        return (
            self._expr.str.slice(
                0, 1
            ).str.to_uppercase() +
            self._expr.str.slice(
                1))


def convert_time_to(value, unit_from, unit_to):
    """
    Simple time unit conversion
    """

    ratios = {
        "s": 1./60.,
        "m": 1.,
        "h": 60.,
        "d": 60. * 24.}

    if unit_from not in ratios.keys():
        raise ValueError(
            "unknown unit to convert from "
            "{}".format(unit_from))
    if unit_to not in ratios.keys():
        raise ValueError(
            "unknown unit to convert to "
            "{}".format(unit_to))

    val_min = value * ratios.get(unit_from)

    return val_min / ratios.get(unit_to)


def spread_draws(posteriors, variable_names):
    """
    Given a dictionary of posteriors,
    return a long-form polars dataframe
    indexed by draw, with variable
    values (equivalent of tidybayes
    spread_draws() function).

    :param posteriors: a dictionary of posteriors
    with variable names as keys and numpy ndarrays
    as values (with the first axis corresponding
    to the posterior draw number.
    :param variable_names: list of strings or
    of tuples identifying which variables to
    retrieve.
    """

    for i_var, v in enumerate(variable_names):
        if isinstance(v, str):
            v_dims = None
        else:
            v_dims = v[1:]
            v = v[0]

        post = posteriors.get(v)
        long_post = post.flatten()[..., np.newaxis]

        indices = np.array(list(np.ndindex(post.shape)))
        n_dims = indices.shape[1] - 1
        if v_dims is None:
            dim_names = [("{}_dim_{}_index".format(v, k),
                          pl.Int64)
                         for k in range(n_dims)]
        elif len(v_dims) != n_dims:
            raise ValueError("incorrect number of "
                             "dimension names "
                             "provided for variable "
                             "{}".format(v))
        else:
            dim_names = [(v_dim, pl.Int64)
                         for v_dim in v_dims]

        p_df = pl.DataFrame(
            np.concatenate(
                [indices, long_post],
                axis=1
            ),
            schema=(
                [("draw", pl.Int64)] +
                dim_names +
                [(v, pl.Float64)])
        )

        if i_var == 0:
            df = p_df
        else:
            df = df.join(p_df,
                         on=[col for col in df.columns
                             if col in p_df.columns])
        pass

    return df


def spread_and_recover_ids(
        posteriors: dict,
        variable_names: list | tuple | str,
        id_mappers: dict = {},
        id_datatype: str = "str",
        keep_internal: bool = False):

    temp_spread = spread_draws(
        posteriors,
        variable_names)

    new_cols = []

    for dim_name, mapper in id_mappers.items():
        if dim_name in temp_spread.columns:
            map_vals = temp_spread.get_column(dim_name).to_numpy()
            new_cols.append(
                pl.lit(
                    mapper[map_vals].astype(id_datatype)
                ).alias(
                    dim_name
                )
            )

            if keep_internal:
                new_cols.append(
                    temp_spread.get_column(
                        dim_name
                    ).alias(
                        dim_name + "_internal"
                    ))
    return temp_spread.with_columns(
        new_cols)


def load_mcmc(path: str):
    with open(path, "rb") as file:
        infer = pickle.load(file)

    return infer


def load_data(path: str,
              sep: str = "\t",
              sort_cols: list = [
                  "condition",
                  "timepoint",
                  "replicate",
                  "titer_id"]):
    data = pl.read_csv(
        path, separator=sep
    ).sort(
        sort_cols)
    return data


def index_conditions(data: pl.DataFrame):
    condition_index = data.unique(
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
    )

    return condition_index


def index_titers(data: pl.DataFrame):
    titer_index = data.groupby(
        "titer_id"
    ).agg(
        pl.col("plaque_count").sum().alias("total_plaques")
    ).join(
        data,
        on="titer_id"
    ).with_columns(
        [(pl.col("total_plaques") > 0).alias("detected"),

         (np.log10(pl.lit(const.LOD_single_ml)) -
          np.log10(pl.col("well_volume_ml"))
          ).alias("titer_log_LOD")
         ]
    )

    return titer_index


def index_timeseries(data: pl.DataFrame):
    return data.unique(
        subset="timeseries_id"
    )


def spread_titer_draws(inference_object: Inference,
                       titer_index: pl.DataFrame):
    titers = spread_and_recover_ids(
        inference_object.mcmc_runner.get_samples(),
        [("log_titer", "titer_id")],
        {"titer_id": inference_object.run_data[
            "well_internal_id_values"]["titer"]},
        keep_internal=False
    ).with_columns(
        pl.col("titer_id").cast(pl.Int64)
    ).join(
        titer_index,
        on="titer_id"
    ).with_columns(
        pl.when(
            pl.col("detected")
        ).then(
            pl.col("log_titer")
        ).otherwise(
            pl.col("titer_log_LOD")
        ).alias(
            "log_display_titer"
        )
    ).with_columns(
        (10 ** (pl.col("log_display_titer"))
         ).alias("display_titer")
    )

    return titers


def spread_multiphase_halflives(inference_object: Inference,
                                condition_index: pl.DataFrame):
    hls = spread_and_recover_ids(
        inference_object.mcmc_runner.get_samples(),
        [
            ("log_halflife_first", "condition"),
            ("log_halflife_offsets", "phase_no", "condition"),
            ("breakpoint_times", "phase_no_break", "condition")
        ],

        {"condition":
         inference_object.run_data[
             "unique_external_ids"]["halflife"]}
    ).drop(
        ["phase_no", "phase_no_break"]
    ).join(
        condition_index,
        on="condition")

    return hls


def spread_halflives(inference_object: Inference,
                     condition_index: pl.DataFrame):
    hls = spread_and_recover_ids(
        inference_object.mcmc_runner.get_samples(),
        [("log_halflife", "condition")],
        {"condition": inference_object.run_data[
            "unique_external_ids"]["halflife"]}
    ).join(
        condition_index,
        on="condition")

    return hls


def spread_intercepts(inference_object: Inference,
                      timeseries_index: pl.DataFrame):
    intercepts = spread_and_recover_ids(
        inference_object.mcmc_runner.get_samples(),
        [("log_titer_intercept", "timeseries_id")],
        {"timeseries_id": inference_object.run_data[
            "unique_external_ids"]["intercept"]}
    ).join(
        timeseries_index,
        on="timeseries_id"
    )
    return intercepts


def concat_liquid_surface_hls(
        liquid_hls: pl.DataFrame = None,
        surface_hls: pl.DataFrame = None):

    liquid_hls = liquid_hls.rename(
        {"log_halflife": "log_halflife_first"}
    ).with_columns(
        [
            pl.lit(0.0).alias("log_halflife_offsets"),
            pl.lit(100000.0).alias("breakpoint_times")
        ]
    )

    concat_hls = pl.concat([liquid_hls,
                            surface_hls.select(
                                liquid_hls.columns
                            )])
    return concat_hls


def concat_liquid_surface_intercepts(
        liquid_intercepts: pl.DataFrame = None,
        surface_intercepts: pl.DataFrame = None):
    return pl.concat(
        [surface_intercepts,
         liquid_intercepts])


def with_halflife_derived_quantities(
        halflife_df: pl.DataFrame):

    df = halflife_df.with_columns([

        (
            pl.col("log_halflife_first")
        ).exp().alias("halflife_wet"),

        (
            pl.col("log_halflife_first") +
            pl.col("log_halflife_offsets")
        ).exp().alias("halflife_dry")

    ]).with_columns([

        (pl.lit(np.log10(2)) /
         pl.col("halflife_wet")
         ).alias("decay_rate_wet"),

        (pl.lit(np.log10(2)) /
         pl.col("halflife_dry")
         ).alias("decay_rate_dry")

    ]).with_columns([

        (-1.0 * pl.col("decay_rate_wet")
         ).alias("exp_rate_wet"),

        (-1.0 * pl.col("decay_rate_dry")
         ).alias("exp_rate_dry")

    ])

    return df


def with_regression_plot_columns(
        df: pl.DataFrame):

    result_df = df.with_columns([
        (pl.col("log_titer_intercept") -
         pl.col("decay_rate_wet") *
         pl.col("breakpoint_times")
         ).alias("breakpoint_log_titer")
    ]).with_columns(
        (pl.col("breakpoint_log_titer") +
         pl.col("decay_rate_dry") *
         pl.col("breakpoint_times")
         ).alias("log_intercept_dry")
    ).with_columns([

        (10 ** (pl.col("log_titer_intercept"))).alias("initial_titer"),
        (10 ** (pl.col("log_intercept_dry"))).alias("intercept_dry")

    ])

    return result_df


def random_sample_draws_by_timeseries(
        df: pl.DataFrame = None,
        n_lines_per_titer: int = 0,
        n_titers_per_timeseries_liquid: int = 8,
        n_titers_per_timeseries_surface: int = 1,
        random_seed: int = None):

    if random_seed is not None:
        np.random.seed(random_seed)

    to_sample = df.select(
        ["timeseries_id",
         "medium_state"]
    ).unique(
        subset="timeseries_id"
    ).with_columns(
        [pl.lit(
            n_lines_per_titer
        ).alias(
            "n_lines_per_titer"
        ),

         pl.when(
             pl.col("medium_state") == "surface"
         ).then(
             n_titers_per_timeseries_surface
         ).when(
             pl.col("medium_state") == "liquid"
         ).then(
             n_titers_per_timeseries_liquid
         ).otherwise(
             None
         ).alias(
             "n_titers_per_timeseries"
         )
         ]
    ).with_columns(
        (pl.col("n_lines_per_titer") *
         pl.col("n_titers_per_timeseries")
         ).alias("n_reps")
    ).select(
        pl.col(
            "timeseries_id"
        ).repeat_by(pl.col("n_reps"))
    ).explode(
        "timeseries_id"
    )

    sampled_draws = to_sample.with_columns(
        pl.lit(
            np.random.randint(
                df.select("draw").min(),
                df.select("draw").max(),
                size=to_sample.shape[0])
        ).alias(
            "draw"
        )
    )

    return sampled_draws


def get_tidy_titers(
        titer_infer: Inference = None,
        data: pl.DataFrame = None):
    titer_index = index_titers(
        data)
    spread_titers = spread_titer_draws(
        titer_infer,
        titer_index)

    return spread_titers


def get_tidy_hls(
        hl_liquid_infer: Inference = None,
        hl_surface_infer: Inference = None,
        data: pl.DataFrame = None):

    condition_index = index_conditions(
        data)

    _tidy_liquid_hls = spread_halflives(
        hl_liquid_infer,
        condition_index)
    _tidy_surface_hls = spread_multiphase_halflives(
        hl_surface_infer,
        condition_index)

    tidy_hls = concat_liquid_surface_hls(
        liquid_hls=_tidy_liquid_hls,
        surface_hls=_tidy_surface_hls
    ).pipe(
        with_halflife_derived_quantities)

    return tidy_hls


def get_tidy_intercepts(
        hl_liquid_infer: Inference = None,
        hl_surface_infer: Inference = None,
        data: pl.DataFrame = None):

    timeseries_index = index_timeseries(
        data)

    _tidy_liquid_intercepts = spread_intercepts(
        hl_liquid_infer,
        timeseries_index)
    _tidy_surface_intercepts = spread_intercepts(
        hl_surface_infer,
        timeseries_index)
    tidy_intercepts = concat_liquid_surface_intercepts(
        liquid_intercepts=_tidy_liquid_intercepts,
        surface_intercepts=_tidy_surface_intercepts)

    return tidy_intercepts


def get_tidy_hls_with_intercepts(
        spread_hls: pl.DataFrame = None,
        spread_intercepts: pl.DataFrame = None):

    hls_with_intercepts = spread_hls.join(
        spread_intercepts.select(
            ["draw",
             "condition",
             "timeseries_id",
             "log_titer_intercept"]),
        on=["draw", "condition"]
    ).pipe(
        with_regression_plot_columns)

    return hls_with_intercepts


def get_default_tidy_results(
        data_path: str = None,
        titer_infer_path: str = None,
        hl_liquid_infer_path: str = None,
        hl_surface_infer_path: str = None):

    data = load_data(
        data_path)
    titer_infer = load_mcmc(
        titer_infer_path)
    hl_liquid_infer = load_mcmc(
        hl_liquid_infer_path)
    hl_surface_infer = load_mcmc(
        hl_surface_infer_path)

    tidy_titers = get_tidy_titers(
        titer_infer=titer_infer,
        data=data)

    tidy_hls = get_tidy_hls(
        hl_liquid_infer=hl_liquid_infer,
        hl_surface_infer=hl_surface_infer,
        data=data)

    tidy_intercepts = get_tidy_intercepts(
        hl_liquid_infer=hl_liquid_infer,
        hl_surface_infer=hl_surface_infer,
        data=data)

    tidy_hls_with_intercepts = get_tidy_hls_with_intercepts(
        tidy_hls,
        tidy_intercepts)

    result = {
        "data": data,
        "titers": tidy_titers,
        "hls": tidy_hls,
        "intercepts": tidy_intercepts,
        "hls_with_intercepts": tidy_hls_with_intercepts}

    return result


def expression_format_point_interval(
    point_estimate_column: str,
    left_endpoint_column: str,
    right_endpoint_column: str,
    format_string: str = "{point:.2f} [{left:.2f}, {right:.2f}]"
) -> pl.Expr:
    """
    Get a Polars expression formatting
    posterior estimates in the form
    "point_estimate [interval_left,
    interval_right]" for use in
    written results sections.

    Parameters
    ----------
    point_estimate_column: str
        Name of the column containing the point estimate(s)
    left_endpoint_column: str
        Name of the column containing the left interval endpoint(s)
    right_endpoint_column: str
        Name of the column containing the right interval endpoint(s)
    format_string: str
        Format string to format with the point estimate and endpoints,
        when these are passed as a dict of the form
            ```{"point": x_1, "left": x_2, "right": x_3}```
        Default: "{point:.2f} [{left:.2f}, {right:.2f}]",
        which would yield something like "2.52 [1.88, 3.51]"

    Returns
    -------
    format_expr: pl.Expr
        A polars expression that will yield
        appropriately formatted strings when
        evaluated.
    """
    return (
        pl.struct(
            [pl.col(point_estimate_column).alias("point"),
             pl.col(left_endpoint_column).alias("left"),
             pl.col(right_endpoint_column).alias("right")]
        ).apply(
            lambda x: format_string.format(**x)
        ))


def median_qi_table(
        df: pl.DataFrame,
        columns: list,
        group_columns: list = None,
        rename={}
) -> pl.DataFrame:
    """
    Given a tidy polars DataFrame,
    a table of formatted medians
    and quantile intervals for the given
    columns, optionally grouped.

    Parameters
    ----------

    df : pl.DataFrame
        The dataframe on which to perform the summary operation
    columns : list
        The columns to get estimates for
    group_columns: list, optional
        Columns to group by, if any
    rename : dict, optional
        Optional dictionary to rename
        the estimate columns in the output
        table, e.g. {"col1": "foo"}
        would lead estimates based on column
        "col1" to be called "foo_median"
        "foo_q025", "foo_q795", and "
        "foo_formatted" in the output table.

    Returns
    -------

    summary_table: pl.DataFrame
        A polars data frame with the summary estimates
    """
    if group_columns is None:
        df = df.with_columns(pl.lit(1).alias("group_id"))
        group_columns = ["group_id"]
    tab = df.groupby(group_columns)
    estimates = ["median", "q025", "q975", "formatted",
                 "median_minutes", "q025_minutes",
                 "q975_minutes", "formatted_minutes"]

    summary_table = tab.agg(
        [
            col for x in columns for col in
            [pl.col(x).median().alias(x + "_median"),
             pl.col(x).quantile(0.025).alias(x + "_q025"),
             pl.col(x).quantile(0.975).alias(x + "_q975"),

             (pl.col(x) * 24. * 60.).median().alias(
                 x + "_median_minutes"),
             (pl.col(x) * 24. * 60.).quantile(0.025).alias(
                 x + "_q025_minutes"),
             (pl.col(x) * 24. * 60.).quantile(0.975).alias(
                 x + "_q975_minutes")]
        ]
    ).with_columns(
        [expression_format_point_interval(
            x + "_median",
            x + "_q025",
            x + "_q975").alias(
                x + "_formatted")
         for x in columns]
    ).with_columns(
        [expression_format_point_interval(
            x + "_median_minutes",
            x + "_q025_minutes",
            x + "_q975_minutes").alias(
                x + "_formatted_minutes")
         for x in columns
         ]
    ).select(
        group_columns +
        [
            col for x in columns for col in
            [pl.col(x + "_" + est).alias(
                rename.get(x, x) + "_" + est)
             for est in estimates]
        ]
    ).sort(group_columns)

    return summary_table
