#!/usr/bin/env python3

#############
# filename: figure-serum.py
# description: figure showing
# halflives in serum for
# different concentrations

import sys
import polars as pl
import matplotlib.pyplot as plt

import plotting as plot
import analyze


def get_serum_draws(df):
    """
    Filter tidy dataframe of MCMC posteriors
    to get only serum experiment rows,
    and format those rows for plotting

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
    pl.Series(
        ["0\\%", "40\\%", "80\\%", "100\\%"]
    ).cast(pl.Categorical)

    serum_filter = (
        (pl.col("medium") == "Serum") &
        (pl.col("medium_state") == "liquid")
    )

    serum_percentage_float = (
        pl.col(
            "concentration_pct"
        ).cast(pl.Float64).alias("Percentage")
    )
    serum_percentage_factor = (
        pl.format("{}\\%",
                  pl.col(
                      "concentration_pct"
                  ).cast(pl.Float64).cast(pl.Int64)
                  ).cast(
            pl.Categorical
        ).alias("PercentageFactor")
    )

    return df.filter(
        serum_filter
    ).with_columns(
        [serum_percentage_float,
         serum_percentage_factor]
    )


def render_and_style_figure(
        reg_panel,
        hl_panel):
    """
    Take in Grizzlyplot objects,
    do customization and styling,
    and return matplotlib figure
    object.
    """

    fig = plt.figure(figsize=plot.get_figsize(2))
    fig_reg, fig_hl = fig.subfigures(
        2, 1)
    fig_reg, ax_reg = reg_panel.render(
        fig=fig_reg)
    fig_hl, ax_hl = hl_panel.render(
        fig=fig_hl)

    ax_reg[0].set_yticks(plot.default_yticks)
    ax_reg[0].set_xticks(plot.default_xticks)
    ax_reg[0].set_xlim(plot.default_xlim)
    ax_reg[0].set_ylim(plot.default_ylim)

    ax_hl[0].set_yticks([0, 1, 2, 3])
    ax_hl[0].set_xticks([0, 20, 40, 60, 80, 100])
    ax_hl[0].set_xlim([-20, 120])
    ax_hl[0].set_ylim([-0.25, 2.5])

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
    reg_panel = plot.regression_liquid(
        results["titers"].pipe(get_serum_draws),
        hl_regression_lines.pipe(get_serum_draws),
        facet=dict(
            col="PercentageFactor",
            label_kwargs=dict(
                fontsize="xx-large")),
        xlabel="Time (d)",
        ylabel="Virus titer (PFU/mL)")

    hl_panel = plot.halflife_violins(
        results["hls"].pipe(get_serum_draws),
        x_column="Percentage",
        xlabel="Serum percentage",
        ylabel="Virus half-life (d)"
    )

    ######################################
    # Rendering, styling, and output
    ######################################
    print("Rendering and styling figure...")
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
        with pl.StringCache():
            main(data_path=sys.argv[1],
                 titer_infer_path=sys.argv[2],
                 hl_liquid_infer_path=sys.argv[3],
                 hl_surface_infer_path=sys.argv[4],
                 outpath=sys.argv[-1])
