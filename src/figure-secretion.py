#!/usr/bin/env python3

# filename: figure-secretions.py
# description: figure of halflives and
# fits for human secretions

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotting as plot
import polars as pl

import grizzlyplot as gp
import grizzlyplot.geoms as geoms
import grizzlyplot.scales as scales
from grizzlyplot.transforms import dynamic_xspan_transform

import analyze

from grizzlyplot.defaults import plot_defaults

plot_defaults["facet_label_pad_x"] = 0.15


def prep_secretion_hl_data(data):
    pl.Series(["Liquid",
               "Surface wet",
               "Surface dry"]).cast(pl.Categorical)

    melted = data.melt(
        id_vars=["medium", "medium_state", "draw"],
        value_vars=["halflife_wet", "halflife_dry"],
        value_name="halflife",
        variable_name="drying_state"
    ).filter(
        ~(
            (pl.col("medium_state") == "liquid") &
            (pl.col("drying_state") == "halflife_dry")
        )
    ).with_columns(
        pl.col("drying_state").str.extract(
            "halflife_(wet|dry)"
        ).alias("drying_state")
    ).with_columns(
        pl.when(
            pl.col("medium_state") == "surface"
        ).then(
            "Surface " +
            pl.col("drying_state")
        ).otherwise(
            pl.col("medium_state").exstr.capitalize()
        ).alias(
            "condition_and_state"
        ).cast(
            pl.Categorical)
    )
    return melted


def secretion_halflife_plot(data):

    return gp.GrizzlyPlot(
        data=data,

        mapping=dict(
            x="condition_and_state",
            y="halflife",
            fillcolor="condition_and_state",
            markerfacecolor="condition_and_state"),

        geoms=[
            geoms.GeomViolin(
                name="halflife violin",
                violinwidth=0.5,
                linecolor="none",
                norm="area",
                trimtails=0.01),
            geoms.GeomPointIntervalY(
                name="halflife pointinterval",
                markersize=7,
                lw=3,
                color="k")
        ],

        scales={
            "fillcolor": plot.medium_state_scale,
            "markerfacecolor": plot.medium_state_scale,
            "x": scales.ScaleAxisCategorical(axis="x"),
            "y": scales.ScaleY("log")
        },

        facet=dict(
            col="medium",
            sharex=False,
            label_cols=False),

        alpha=0.75,
        fillalpha=0.5,
        lw=1,
        xlabel="Condition",
        ylabel="Virus half-life"
    )


def render_and_style_figure(
        reg_panel,
        halflife_panel):

    ylab_pad = 23
    fig_col_space = 0.225
    fig = plt.figure(figsize=plot.get_figsize(
        width=8.5,
        aspect=2.5))
    fig_reg, fig_hl = plot.ordered_subfigures(
        fig,
        2, 1,
        order="tblr",
        height_ratios=[1, 1],
        hspace=0)
    _, reg_ax = reg_panel.render(fig=fig_reg)
    reg_ax[0].get_gridspec().wspace = fig_col_space

    reg_ax[0].set_ylim(plot.default_ylim)
    reg_ax[0].set_yticks(plot.default_yticks)
    reg_ax[0].set_xlim(plot.default_xlim)
    reg_ax[0].set_xticks(plot.default_xticks)
    plot.left_align_yticklabels(
        reg_ax[0], ylab_pad)
    plot.left_align_yticklabels(
        reg_ax[6], ylab_pad)

    _, hl_ax = halflife_panel.render(fig=fig_hl)

    ytickvals, yticklabs = plot.multiunit_time_ticks(
        values=[5, 1, 1, 10, 100],
        units=["m", "h", "d", "d", "d"],
        axis_unit="d")
    hl_ax[0].set_yticks(ytickvals)
    hl_ax[0].set_yticklabels(yticklabs)
    plot.left_align_yticklabels(
        hl_ax[0])

    hl_xspan = 2
    hl_xdiff = hl_xspan
    hl_ax[0].get_gridspec().wspace = fig_col_space
    for i_ax, axis in enumerate(hl_ax):
        axis.set_title("")
        plot.rotate_xticklabels(axis, 35, fontsize="x-small")
        axis.set_xlim([-hl_xdiff,
                       2 + hl_xdiff])
        axis.set_ylim([0.2e-2, 1e4])
    plot.left_align_yticklabels(
        hl_ax[0], ylab_pad)
    leg_elements = [
        Patch(facecolor=plot.medium_state_scale(state)[0],
              label=state)
        for state in ["Liquid", "Surface wet", "Surface dry"]]
    leg = fig_hl.legend(
        title_fontsize="large",
        frameon=False,
        handles=leg_elements,
        ncol=3,
        loc="lower center",
        fontsize="x-small")
    for leg_h in leg.legend_handles:
        leg_h.set_alpha(0.75)

    def dynamic_bbox():
        leg.set_bbox_to_anchor(
            [0.25, -0.01],
            transform=dynamic_xspan_transform(fig_reg))
        return leg._bbox_to_anchor

    leg.get_bbox_to_anchor = dynamic_bbox

    return fig, fig_reg, fig_hl


def main(data_path: str = None,
         titer_infer_path: str = None,
         hl_liquid_infer_path: str = None,
         hl_surface_infer_path: str = None,
         outpath: str = None):
    print("Registering ordered categoricals with string cache")
    pl.Series(plot.secretion_order).cast(pl.Categorical)
    print("Reading and processing data...")
    tidy_results = analyze.get_default_tidy_results(
        data_path=data_path,
        titer_infer_path=titer_infer_path,
        hl_liquid_infer_path=hl_liquid_infer_path,
        hl_surface_infer_path=hl_surface_infer_path)

    is_secretion = pl.col("medium").is_in(plot.secretion_order)
    serum_exclude = ~pl.col("condition").str.contains(
        "Serum_(0%|40%|80%)_liquid"
    )
    secretion_filter = is_secretion & serum_exclude

    processed = {key:
                 val.filter(
                     secretion_filter
                 ).with_columns(
                     pl.col("medium").cast(
                         pl.Categorical)
                 )
                 for key, val in tidy_results.items()}

    random_sample = analyze.random_sample_draws_by_timeseries(
        df=processed["hls_with_intercepts"],
        n_lines_per_titer=3)

    regression_lines = processed["hls_with_intercepts"].join(
        random_sample,
        on=["timeseries_id", "draw"])

    hl_panel_data = prep_secretion_hl_data(
        processed["hls"])

    print("Defining panels...")

    regression_panel = plot.regression_surface(
        processed["titers"],
        regression_lines,
        facet=dict(
            col="medium",
            row="medium_state",
            label_kwargs=dict(
                fontsize="x-large")
        ),
        xlabel="Time (d)",
        ylabel="Titer (PFU/mL)"
    )

    halflife_panel = secretion_halflife_plot(
        hl_panel_data)

    print("Rendering and styling figure...")
    fig, fig_reg, fig_hl = render_and_style_figure(
        regression_panel,
        halflife_panel)

    print("Saving figure to {}...".format(outpath))
    fig.savefig(outpath,
                bbox_inches="tight",
                pad_inches=0.25)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("\nUSAGE: {} <data file> "
              "<titer mcmc file> "
              "<liquid halflife mcmc file> "
              "<surface halflife mcmc file> "
              "<output path>\n"
              "".format(sys.argv[0]))
    else:
        # ordered categoricals
        with pl.StringCache():
            main(data_path=sys.argv[1],
                 titer_infer_path=sys.argv[2],
                 hl_liquid_infer_path=sys.argv[3],
                 hl_surface_infer_path=sys.argv[4],
                 outpath=sys.argv[-1])
