#!/usr/bin/env python3

# filename: table-halflives.py
# description: autogenerate table of halflives
# from the MCMC results

import sys
import polars as pl
import pandas as pd

import analyze


def column_name_latex_formatter(col_name):
    if col_name == "relative_humidity_pct":
        result = "RH (\\si{\\%})"
    elif col_name == "temperature":
        result = "Temp. (\\si{\\celsius})"
    elif col_name == "halflife_wet_formatted":
        result = "Liquid-phase h.l. (\\si{\\day})"
    elif col_name == "halflife_dry_formatted":
        result = "Dried-phase h.l. (\\si{\\day})"
    else:
        result = col_name.capitalize()
    return result


def main(data_path: str = None,
         titer_infer_path: str = None,
         hl_liquid_infer_path: str = None,
         hl_surface_infer_path: str = None,
         outpath: str = None):

    print("Reading and processing data...")
    tidy_results = analyze.get_default_tidy_results(
        data_path=data_path,
        titer_infer_path=titer_infer_path,
        hl_liquid_infer_path=hl_liquid_infer_path,
        hl_surface_infer_path=hl_surface_infer_path)

    surface = "surface" in outpath
    liquid = "liquid" in outpath

    if surface:
        state = "surface"
        columns = ["halflife_wet",
                   "halflife_dry"]
        rename = {}
    elif liquid:
        state = "liquid"
        columns = ["halflife_wet"]
        rename = {"halflife_wet": "halflife"}
    else:
        raise ValueError("Unknown requested table {}"
                         "".format(outpath))

    table = analyze.median_qi_table(
        tidy_results["hls"].filter(
            pl.col("medium_state") == state),
        columns=columns,
        group_columns=["condition",
                       "surface",
                       "medium",
                       "temperature",
                       "relative_humidity_pct",
                       "strain"],
        rename=rename)

    print("Saving table to {}".format(outpath))
    if outpath.endswith(".tsv"):
        table.write_csv(outpath,
                        separator="\t")
    elif outpath.endswith(".tex"):
        sort_order_cols = [
            pl.col("temperature"),
            pl.col("relative_humidity_pct"),
            pl.col("medium"),
            pl.col("surface")
        ]
        display_order_cols = [
            pl.col("medium"),
            pl.col("surface"),
            pl.col("temperature"),
            pl.col("relative_humidity_pct")
        ]

        pandas_tab = table.filter(
            pl.col("strain") == 2022
        ).sort(
            sort_order_cols
        ).select(
            display_order_cols + [
                "halflife_wet_formatted",
                "halflife_dry_formatted"
            ]
        ).to_pandas()

        pandas_tab.rename(
            column_name_latex_formatter,
            axis="columns"
        ).style.format(
            na_rep="",
            precision=0,
            formatter={
                "Surface": lambda x: x.replace("_", " ").capitalize()
            }
        ).hide(
            axis="index"
        ).to_latex(
            buf=outpath,
        )


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("\nUSAGE: {} <data file> "
              "<titer mcmc file> "
              "<liquid halflife mcmc file> "
              "<surface halflife mcmc file> "
              "<output path>\n"
              "".format(sys.argv[0]))
    else:
        pd.DataFrame()
        main(data_path=sys.argv[1],
             titer_infer_path=sys.argv[2],
             hl_liquid_infer_path=sys.argv[3],
             hl_surface_infer_path=sys.argv[4],
             outpath=sys.argv[-1])
