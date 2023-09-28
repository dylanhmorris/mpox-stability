#!/usr/bin/env python3

# filename: figure-surface.py
# description: figure of halflives and
# fits for surfaces

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotting as plot
import polars as pl

import grizzlyplot as gp
import grizzlyplot.geoms as geoms
import grizzlyplot.scales as scales

import analyze


def halflife_plot(data):
    return gp.GrizzlyPlot(

        data=data,

        mapping=dict(
            x="temperature",
            y="halflife",
            fillcolor="drying_state",
            markerfacecolor="drying_state"
        ),

        facet=dict(
            col="surface_display",
            label_cols=False
        ),

        geoms=[

            geoms.GeomViolin(name="halflives",
                             violinwidth=2,
                             linecolor="none",
                             norm="area",
                             trimtails=0.01),

            geoms.GeomPointIntervalY(
                markersize=10,
                lw=2,
                color="k")
        ],

        scales={
            "y": scales.ScaleY("log"),
            "fillcolor": plot.dmem_color_scale,
            "markerfacecolor": plot.dmem_color_scale
        },

        alpha=0.75,
        fillalpha=0.5,
        ylabel="Virus half-life",
        xlabel="Temperature (°C)"
    )


def add_surface_display_columns(df):
    pl.Series(plot.surface_order + [None]).cast(
        pl.Categorical)
    pl.Series(["4°C", "21°C", "28°C", None]).cast(
        pl.Categorical)

    return df.with_columns(
        [
            pl.col(
                "surface"
            ).str.replace(
                "_", " "
            ).exstr.capitalize(
            ).cast(
                pl.Categorical
            ).alias(
                "surface_display"
            ),

            (
                pl.col(
                    "temperature"
                ).cast(
                    pl.Int64
                ).cast(
                    pl.Utf8
                ) + "°C"
            ).cast(
                pl.Categorical
            ).alias(
                "temperature_display"
            )
        ])


def read_and_process_data(data_path: str = None,
                          titer_infer_path: str = None,
                          hl_liquid_infer_path: str = None,
                          hl_surface_infer_path: str = None):
    pl.Series(["Wet", "Dry"]).cast(pl.Categorical)

    tidy_results = analyze.get_default_tidy_results(
        data_path=data_path,
        titer_infer_path=titer_infer_path,
        hl_liquid_infer_path=hl_liquid_infer_path,
        hl_surface_infer_path=hl_surface_infer_path)

    surface_filter = (
        (pl.col("medium_state") == "surface") &
        pl.col("surface").is_not_null() &
        (pl.col("strain") == 2022) &
        pl.col("temperature").is_not_null() &
        pl.col("relative_humidity_pct").is_not_null() &
        (pl.col("medium") == "DMEM"))

    processed = {key: val.filter(
        surface_filter).pipe(
            add_surface_display_columns
        ) for key, val in tidy_results.items()}

    random_sample = analyze.random_sample_draws_by_timeseries(
        df=processed["hls_with_intercepts"],
        n_lines_per_titer=3)

    titers_plot = processed["titers"]

    lines_plot = processed["hls_with_intercepts"].join(
        random_sample,
        on=["timeseries_id", "draw"])

    hls_plot = processed["hls"].melt(
        id_vars=["surface_display",
                 "temperature",
                 "relative_humidity_pct",
                 "draw"],
        value_vars=["halflife_wet",
                    "halflife_dry"],
        value_name="halflife",
        variable_name="drying_state"
    ).with_columns(
        pl.col("drying_state").str.extract(
            "halflife_(wet|dry)"
        ).exstr.capitalize(
        ).cast(
            pl.Categorical
        ).alias(
            "drying_state"
        ))

    return titers_plot, lines_plot, hls_plot


def render_and_style_figure(
        reg_panel,
        hl_panel):
    fig = plt.figure(figsize=plot.get_figsize(
        width=8.5,
        aspect=2))
    fig_reg, fig_hl = plot.ordered_subfigures(
        fig,
        2, 1,
        order="tblr",
        height_ratios=[1.75, 1],
        hspace=0)

    ylab_pad = 23

    _, reg_axes = reg_panel.render(fig=fig_reg)

    reg_axes[0].get_gridspec().wspace = 0.2
    reg_axes[0].set_ylim(plot.default_ylim)
    reg_axes[0].set_yticks(plot.default_yticks)
    reg_axes[0].set_xlim([-1, 21])
    reg_axes[0].set_xticks(plot.default_xticks)
    plot.left_align_yticklabels(
        reg_axes[0], ylab_pad)
    plot.left_align_yticklabels(
        reg_axes[3], ylab_pad)
    plot.left_align_yticklabels(
        reg_axes[6], ylab_pad)

    _, hl_axes = hl_panel.render(fig=fig_hl)
    hl_axes[0].get_gridspec().wspace = 0.2
    hl_axes[0].set_xticks([0, 4, 21, 28])
    hl_axes[0].set_xlim([-1.5, 31.5])

    ytickvals, yticklabs = plot.multiunit_time_ticks(
        values=[5, 1, 1, 10, 100],
        units=["m", "h", "d", "d", "d"],
        axis_unit="d")
    hl_axes[0].set_yticks(ytickvals)
    hl_axes[0].set_yticklabels(yticklabs)
    plot.left_align_yticklabels(
        hl_axes[0], ylab_pad)
    hl_axes[0].set_ylim([5e-4, 2e2])

    leg_elements = [
        Patch(facecolor=plot.dmem_color_scale(state)[0],
              label=state,
              alpha=0.75)
        for state in ["Wet", "Dry"]]
    leg = fig_hl.legend(
        title="Sample state",
        handles=leg_elements,
        ncol=3,
        frameon=False,
        loc="lower center")

    def dynamic_bbox():
        leg.set_bbox_to_anchor(
            [0.5, 0.9],
            transform=hl_axes[0].transAxes)
        return leg._bbox_to_anchor

    leg.get_bbox_to_anchor = dynamic_bbox

    fig.align_labels()
    return fig


def main(data_path: str = None,
         titer_infer_path: str = None,
         hl_liquid_infer_path: str = None,
         hl_surface_infer_path: str = None,
         outpath: str = None):

    print("Reading and processing data...")
    (titers_plot,
     lines_plot,
     hls_plot) = read_and_process_data(
         data_path,
         titer_infer_path,
         hl_liquid_infer_path,
         hl_surface_infer_path)

    print("Defining panels...")
    reg_panel = plot.regression_surface(
        titers_plot,
        lines_plot,
        facet=dict(
            row="temperature_display",
            col="surface_display",
            label_kwargs=dict(
                fontsize="x-large")
        ),
        xlabel="Time (d)",
        ylabel="Virus titer (PFU/mL)"
    )

    hl_panel = halflife_plot(hls_plot)

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
