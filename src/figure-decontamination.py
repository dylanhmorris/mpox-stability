#!/usr/bin/env python3

#############
# filename: figure-decontamination.py
# description: figure showing
# halflives in fluids decontaminated
# with various concentrations of chlorine

import sys
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plotting as plot
import analyze

import grizzlyplot.scales as scales


def get_decontamination_draws(df):
    """
    Filter tidy dataframe of MCMC posteriors
    to get only decontamination experiment
    row, and format those rows for plotting

    Parameters
    ----------
    df : polars.DataFrame
        Tidy polars DataFrame of MCMC draws.

    Returns
    -------

    df_plot: polars.DataFrame
        Filtered DataFrame with desired rows
        and new columns
    """

    decon_filter = (
        (pl.col("medium") == "di_water") |
        (pl.col("medium") == "wastewater")
    )

    # convert time from days to minutes
    convert_time_to_minutes = (
        (pl.col("timepoint") * 24. * 60.
         ).alias("timepoint")
    )

    convert_halflife_to_minutes = (
        (pl.col("halflife_wet") * 24. * 60.
         ).alias("halflife_wet")
    )

    convert_rate_to_per_minute = (
        (pl.col("exp_rate_wet") / 24. / 60.
         ).alias("exp_rate_wet")
    )

    # format chlorine PPM as ints for display
    chlorine_concentration_display = (
        pl.col(
            "chlorine_concentration_ppm"
        ).cast(
            pl.Int64
        ).alias(
            "chlorine_concentration_display"
        )
    )
    # format medium names for display
    medium_display = (
        pl.when(
            pl.col("medium") == "di_water"
        ).then(
            pl.lit("DI water")
        ).when(
            pl.col("medium") == "wastewater"
        ).then(
            pl.lit("Wastewater")
        ).otherwise(
            None
        ).alias(
            "medium"
        )
    )

    df_plot = df.filter(
        decon_filter
    ).with_columns([
        chlorine_concentration_display,
        medium_display
    ])

    # convert times, durations
    # and rates only if the
    # dataframe actually contains
    # relevant columns
    if "timepoint" in df_plot.columns:
        df_plot = df_plot.with_columns(
            convert_time_to_minutes)
    if "exp_rate_wet" in df_plot.columns:
        df_plot = df_plot.with_columns(
            convert_rate_to_per_minute)
    if "halflife_wet" in df_plot.columns:
        df_plot = df_plot.with_columns(
            convert_halflife_to_minutes)

    return df_plot


def render_and_style_figure(
        reg_panel,
        hl_panel):
    """
    Take in Grizzlyplot objects,
    do customization and styling,
    and return matplotlib figure
    object.

    Parameters
    ----------
    reg_panel : grizzlyplot.GrizzlyPlot
        GrizzlyPlot of titer regression

    hl_panel : grizzlyplot.GrizzlyPlot
        GrizzlyPlot of halflives

    Returns
    --------
    fig: matplotlib.figure.Figure
        The rendered, styled figure
    """

    fig = plt.figure(figsize=plot.get_figsize(1.6))
    fig_reg, fig_hl = fig.subfigures(
        2, 1, height_ratios=[1, 1])
    fig_reg, ax_reg = reg_panel.render(
        fig=fig_reg)
    fig_hl, ax_hl = hl_panel.render(
        fig=fig_hl)

    ax_reg[0].set_yticks(plot.default_yticks)
    ax_reg[0].set_xticks([0, 30, 60, 90, 120])
    ax_reg[0].set_xlim([-5, 125])
    ax_reg[0].set_ylim(plot.default_ylim)

    ytickvals, yticklabs = plot.multiunit_time_ticks(
        values=[1, 1, 1, 1, 10, 100],
        units=["s", "m", "h", "d", "d", "d"],
        axis_unit="m")
    ax_hl[0].set_ylim([1e-3, 1e7])
    ax_hl[0].set_yticks(ytickvals)
    ax_hl[0].set_yticklabels(yticklabs)
    plot.left_align_yticklabels(
        ax_hl[0])
    ax_hl[0].set_xticks([0, 1, 5, 10])

    leg_elements = [
        Patch(facecolor=plot.medium_scale(state)[0],
              label=state,
              alpha=0.75)
        for state in ["DI water", "Wastewater"]]
    fig_hl.legend(
        handles=leg_elements,
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=[0.15, 1.033],
        bbox_transform=ax_hl[0].transAxes)

    return fig


def render_and_style_control_figure(
        reg_panel):
    """
    Take in Grizzlyplot object,
    do rendering,
    customization,
    and styling,
    and return matplotlib figure
    object.

    Parameters
    ----------
    reg_panel : grizzlyplot.GrizzlyPlot
        GrizzlyPlot of titer regression

    Returns
    --------
    fig: matplotlib.figure.Figure
        The rendered, styled figure
    """

    fig = plt.figure(figsize=plot.get_figsize(1))
    fig, ax_reg = reg_panel.render(
        fig=fig)

    ax_reg[0].set_yticks(plot.default_yticks)
    ax_reg[0].set_ylim(plot.default_ylim)

    xtickvals, xticklabs = plot.multiunit_time_ticks(
        values=[0, 5, 10, 15, 20],
        units=["d", "d", "d", "d", "d"],
        axis_unit="m")
    ax_reg[0].set_xlim([24 * 60 * -1, 24 * 60 * 21])
    ax_reg[0].set_xticks(xtickvals)
    ax_reg[0].set_xticklabels(xticklabs)

    return fig


def main(data_path: str = None,
         titer_infer_path: str = None,
         hl_liquid_infer_path: str = None,
         hl_surface_infer_path: str = None,
         outpath: str = None):
    ###################
    # Data processing
    ###################
    print("Reading and processing data...")
    results = analyze.get_default_tidy_results(
        data_path=data_path,
        titer_infer_path=titer_infer_path,
        hl_liquid_infer_path=hl_liquid_infer_path,
        hl_surface_infer_path=hl_surface_infer_path)
    random_sample = analyze.random_sample_draws_by_timeseries(
        df=results["hls_with_intercepts"],
        n_lines_per_titer=3)
    hl_regression_lines = results["hls_with_intercepts"].join(
        random_sample,
        on=["timeseries_id", "draw"])

    ###################
    # Plot definition
    ###################
    print("Defining panels...")

    control_fig = "controls" in outpath

    xmax = 22 * 24 * 60 if control_fig else 130
    reg_xlabel = "Time" if control_fig else "Time (min)"

    reg_titers = results["titers"].pipe(get_decontamination_draws)
    reg_lines = hl_regression_lines.pipe(get_decontamination_draws)
    reg_facet = dict(
        row="medium",
        label_kwargs=dict(
            fontsize="medium"))
    if control_fig:
        reg_titers = reg_titers.filter(
            pl.col("chlorine_concentration_ppm") < 1)
        reg_lines = reg_lines.filter(
            pl.col("chlorine_concentration_ppm") < 1)
    else:
        reg_facet["col"] = "chlorine_concentration_display"

    reg_panel = plot.regression_liquid(
        reg_titers,
        reg_lines,
        facet=reg_facet,
        xmin=0,
        xmax=xmax,
        xlabel=reg_xlabel,
        ylabel="Virus titer (PFU/mL)")

    # change default color scale for regression
    # lines to medium (possibly make function more
    # generic instead)
    reg_panel.geoms[1].mapping["color"] = "medium"
    reg_panel.scales["color"] = plot.medium_scale

    hl_panel = plot.halflife_violins(
        results["hls"].pipe(get_decontamination_draws),
        x_column="chlorine_concentration_ppm",
        fill_column="medium",
        xlabel="Chlorine concentration (PPM)",
        ylabel="Virus half-life"
    )

    # tweak violins due to wide range
    hl_panel.scales["y"] = scales.ScaleY("log")
    hl_panel.geoms[0].params["norm"] = "max"
    hl_panel.geoms[0].params["violinwidth"] = 2

    ######################################
    # Rendering, styling, and output
    ######################################
    print("Rendering and styling figure...")
    if control_fig:
        fig = render_and_style_control_figure(
            reg_panel)
    else:
        fig = render_and_style_figure(
            reg_panel, hl_panel)

    print("Saving figure to {}...".format(outpath))
    fig.savefig(outpath)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("\nUSAGE: {} <data file> "
              "<titer mcmc file> "
              "<liquid halflife mcmc file> "
              "<surface halflife mcmc file> "
              "<output path>\n"
              "".format(sys.argv[0]))
    else:
        main(data_path=sys.argv[1],
             titer_infer_path=sys.argv[2],
             hl_liquid_infer_path=sys.argv[3],
             hl_surface_infer_path=sys.argv[4],
             outpath=sys.argv[-1])
