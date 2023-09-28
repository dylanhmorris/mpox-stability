#!/usr/bin/env python3

# filename: clean_data.py
# description: take raw excel
# datasheet and output clean tsv
# for inference

import pandas as pd
import numpy as np
import polars as pl
import sys


def fix_decon_cols(df):
    df.columns = [
        "time_minutes" if i_col == 0 else
        df.columns[i_col - (i_col - 1) % 3]
        for i_col, _ in enumerate(df.columns)]
    return df


def read_in_medium(raw_data_path,
                   start_row,
                   medium,
                   sheet_name="wastewater_disinfection"):
    return pd.read_excel(
        raw_data_path,
        sheet_name="wastewater_disinfection",
        skiprows=start_row,
        nrows=8,
        header=1
    ).pipe(
        fix_decon_cols
    ).melt(
        id_vars="time_minutes",
        var_name="chlorine_concentration_ppm",
        value_name="titer_pfu"
    ).dropna(
    ).pipe(
        pl.DataFrame
    ).with_columns(
        [
            pl.col("titer_pfu").cast(pl.Int64),
            pl.lit(medium).alias("medium")
        ]
    )


def n_trailing_zeros(
        number):
    strn = str(number)
    n_zeros = len(strn) - len(strn.rstrip("0"))
    return n_zeros


def plaques_dilution_from_pfu_ml(
        pfu_ml):

    if pfu_ml == 0:
        plaque_count, log10_dilution = 0, 0
    else:
        log10_dilution = -n_trailing_zeros(pfu_ml)
        if not pfu_ml % 10**-log10_dilution == 0:
            raise ValueError(
                "Estimated PFU in undilute "
                "sample ({}) is not "
                "consistent with assumed "
                "experimental practice "
                "underlying this conversion; "
                "it should be divisible by "
                "a non-zero power of 10 "
                "unless it is itself "
                "less than 10".format(
                    pfu_ml
                )
            )
        plaque_count = pfu_ml * (10 ** log10_dilution)

        if not plaque_count % 1 == 0:
            raise ValueError("Failed to get an "
                             "integer number of "
                             "plauqes. Got {}"
                             "".format(plaque_count))
        pass

    return int(plaque_count), log10_dilution


def plaque_count_from_pfu(titer_pfu):
    return plaques_dilution_from_pfu_ml(titer_pfu)[0]


def log10_dilution_from_pfu(titer_pfu):
    return plaques_dilution_from_pfu_ml(titer_pfu)[1]


def get_decon_df(raw_data_path):
    col_names = ["timepoint_min",
                 "replicate",
                 "plaque_count",
                 "dilution",
                 "pfu_per_ml",
                 "log10_pfu_per_ml"]

    n_rows = 24
    dat = None
    for vertical_ind, medium in enumerate(["wastewater", "di_water"]):
        for chlorine, cols in zip(
                [0., 1., 5., 10.],
                ["B:G", "J:O", "R:W", "Z:AE"]):
            temp = pd.read_excel(
                raw_data_path,
                sheet_name="wastewater_disinfection",
                usecols=cols,
                names=col_names,
                skiprows=(0 + (n_rows + 4) * vertical_ind),
                nrows=n_rows)
            temp["timepoint_min"] = temp["timepoint_min"].ffill()
            temp = temp.assign(
                medium=medium,
                chlorine_concentration_ppm=chlorine)
            dat = pd.concat([dat, temp])

    dat = pl.DataFrame(dat)

    dat = dat.with_columns(
        pl.when(
            ((pl.col("dilution").is_null()) |
             (pl.col("dilution") == 0)) &
            (pl.col("plaque_count") == 0) &
            (pl.col("pfu_per_ml") == 0)
        ).then(
            1
        ).otherwise(
            pl.col("dilution")
        ).cast(
            pl.Int64
        ).alias("fold_dilution")
    ).drop(
        ["pfu_per_ml",
         "log10_pfu_per_ml",
         "dilution"]
    ).drop_nulls(
    ).with_columns(
        [
            (-pl.col("fold_dilution").log10()
             ).round(2
                     ).cast(
                         pl.Float64
                     ).alias("log10_dilution"),

            pl.col("plaque_count").cast(pl.Int64)
        ])

    dat = dat.with_columns(
        [
            pl.lit(
                "liquid"
            ).alias("medium_state"),

            pl.when(
                pl.col("chlorine_concentration_ppm") > 0
            ).then(
                (pl.col("medium") +
                 "_" +
                 pl.col(
                     "chlorine_concentration_ppm"
                 ).cast(pl.Utf8) +
                 "_2022"
                 )
            ).when(
                pl.col("chlorine_concentration_ppm") == 0
            ).then(
                pl.col("medium") + "_control_2022"
            ).otherwise(
                None
            ).alias("condition"),

            (pl.col("timepoint_min") / (60.0 * 24.0)
             ).alias("timepoint"),

            pl.lit("2022").alias("strain"),
            pl.lit(None).cast(pl.Utf8).alias("surface"),
            pl.lit(None).cast(pl.Float64).alias("temperature"),
            pl.lit(None).cast(pl.Float64).alias("relative_humidity_pct"),
            pl.lit(None).cast(pl.Float64).alias("concentration_pct"),
            pl.lit(1.0).alias("well_volume_ml")
        ]).with_columns(
            pl.when(
                pl.col("condition").str.contains("_control_2022")
            ).then(
                pl.col("condition") + "_short_experiment"
            ).otherwise(
                pl.col("condition")
            ).alias("intercept_condition")
        ).drop("timepoint_min")

    return dat


def get_stability_df(raw_data_path):
    col_names = ["condition",
                 "timepoint",
                 "replicate",
                 "plaque_count",
                 "dilution",
                 "pfu_per_ml",
                 "log10_pfu_per_ml"]
    dat = None

    # loop over the four-groups-of-
    # columns format of the raw
    # data
    for col_range in [
            "A:G",
            "J:P",
            "R:X",
            "Z:AF"]:
        if not col_range == "Z:AF":

            first_col = col_range[0]
            top_rows = pd.read_excel(
                raw_data_path,
                usecols=first_col,
                nrows=3,
                sheet_name="All_Data",
                skiprows=0,
                names=["col"],
                header=None).dropna()
            strain = (top_rows["col"].str.extract(
                "([0-9]+)").iloc[0].item())

            first_condition = top_rows["col"].iloc[1]
        else:
            strain = "2022"

        temp = pd.read_excel(raw_data_path,
                             usecols=col_range,
                             sheet_name="All_Data",
                             skiprows=1,
                             names=col_names,
                             header=1)

        temp = temp.assign(
            strain=strain)

        # fix misaligned first condition name
        # in certain columns
        if not col_range == "Z:AF":
            temp["condition"].iloc[0] = first_condition

        # fill down for merged cells
        temp["condition"] = temp["condition"].ffill()
        temp["timepoint"] = temp["timepoint"].ffill()

        # remove repeated header rows
        temp = temp[~(temp["replicate"] == "Replicate")]

        # remove manual NA values
        temp = temp[~(temp["dilution"] == "na")]

        # fix 1-fold dilution labeled as zero
        temp["fold_dilution"] = np.where(
            temp["dilution"] == 0,
            1,
            temp["dilution"])
        temp = temp.dropna(
            subset="fold_dilution")

        temp["log10_dilution"] = -np.log10(
            temp["fold_dilution"].astype("float")
        )

        # remove extraneous columns and
        # drop NAs in needed columns
        temp = temp.drop(
            ["pfu_per_ml",
             "log10_pfu_per_ml",
             "dilution"],
            axis=1)
        temp = temp.dropna()

        # add to growing full dataset
        dat = pd.concat([dat, temp])
        pass  # end loop over columns of data

    # fix datatypes
    dat["replicate"] = dat["replicate"].astype("int")

    # convert to polars and add human-readable
    # condition variables
    data = pl.DataFrame(dat)
    data = data.with_columns(
        (pl.when(
            pl.col(
                "condition"
            ).str.contains(
                "(Serum [0-9]+%)|(deionized_water)"
            )
        ).then(
            pl.col("condition") + " liquid"
        ).otherwise(
            pl.col("condition")
        ).str.replace(
            "liqiud",
            "liquid"
        ).str.replace_all(
            "\\s+",
            "_"
        ) + "_" + pl.col("strain")
         ).str.replace("__", "_").alias("condition")
    ).with_columns(
        [pl.col("log10_dilution").cast(pl.Float64),
         pl.col("timepoint").cast(pl.Float64)]
    )

    data = data.with_columns(
        [
            pl.when(
                pl.col(
                    "condition"
                ).str.contains(
                    "SS"
                )
            ).then(
                pl.lit("stainless_steel")
            ).when(
                pl.col(
                    "condition"
                ).str.contains(
                    "PP"
                )
            ).then(
                pl.lit("polypropylene_plastic")
            ).when(
                pl.col(
                    "condition"
                ).str.contains(
                    "CC"
                )
            ).then(
                pl.lit("cotton")
            ).when(
                pl.col("condition").str.contains("surface")
            ).then(
                pl.lit("polypropylene_plastic")
            ).otherwise(
                None
            ).alias(
                "surface"
            ),

            pl.col(
                "condition"
            ).str.extract(
                "([0-9]+)(?:-[0-9]+)?°C"
            ).cast(
                pl.Float64
            ).alias("temperature"),

            pl.col(
                "condition"
            ).str.extract(
                "/([0-9]+)%"
            ).cast(
                pl.Float64
            ).alias("relative_humidity_pct")
        ]
    ).with_columns(
        pl.when(
            pl.col("condition").str.contains(
                "surface")
            |
            pl.col("condition").str.contains(
                "liquid")
        ).then(
            pl.col("condition").str.extract(
                "([aA-zZ]+)_([0-9]+%_)?(surface|liquid)"
            ).cast(
                pl.Utf8
            )
        ).when(
                ~pl.col("surface").is_null()
        ).then(
                pl.lit("DMEM")
        ).when(
                pl.col("condition").str.contains(
                    "Wastewater"
                )
        ).then(
                pl.lit("wastewater")
        ).when(
                pl.col("condition").str.contains(
                    "DI_water"
                )
        ).then(
                pl.lit("di_water")
        ).otherwise(
                None
        ).alias("medium")
    ).with_columns(
        pl.when(
            (pl.col(
                "condition"
             ).str.contains(
                "liquid"
             ) |
             pl.col("medium").is_in(
                 ["wastewater", "di_water"]
             ))
        ).then(
            pl.lit("liquid")
        ).when(
            (pl.col(
                "condition"
            ).str.contains(
                "surface"
            ) | ~pl.col("surface").is_null()
            )
        ).then(
            pl.lit("surface")
        ).otherwise(
            None
        ).alias("medium_state")
    ).with_columns(
        [
            pl.col(
                "condition"
            ).str.extract(
                "Serum_([0-9]+)%_liquid"
            ).cast(
                pl.Float64
            ).alias(
                "concentration_pct"
            ),

            pl.lit(0.0).alias("chlorine_concentration_ppm"),
            pl.lit(0.25).alias("well_volume_ml")
        ]
    ).with_columns(
        [
            pl.when(
                pl.col("condition").str.contains(
                    "Wastewater"
                )
            ).then(
                pl.lit("wastewater_control_2022")
            ).when(
                pl.col("condition").str.contains(
                    "DI_water"
                )
            ).then(
                pl.lit("di_water_control_2022")
            ).otherwise(
                pl.col("condition")
            ).alias("condition"),

            pl.col("log10_dilution").cast(pl.Float64)
        ]
    ).with_columns(
        pl.when(
            pl.col("condition").str.contains(
                "(wastewater|di_water)(_control_2022)")
        ).then(
            pl.col("condition") + "_long_experiment"
        ).otherwise(
            pl.col("condition")
        ).alias("intercept_condition")
    )

    return data


def main(raw_data_path,
         clean_data_path):

    # read in data sheet by sheet
    stab = get_stability_df(raw_data_path)
    decon = get_decon_df(raw_data_path)

    # concatenate the two sheets and
    # add titer index
    df = pl.concat(
        [stab,
         decon.select(stab.columns)]
    ).sort(
        ["condition", "timepoint"]
    ).with_row_count(
        "titer_id"
    ).with_columns(
        pl.when(
            pl.col("medium_state") == "surface"
        ).then(
            pl.col("intercept_condition") +
            "_" +
            pl.col("replicate").cast(pl.Utf8) +
            "_" +
            pl.col("timepoint").cast(pl.Utf8)
        ).when(
            pl.col("medium_state") == "liquid"
        ).then(
            pl.col("intercept_condition") +
            "_" +
            pl.col("replicate").cast(pl.Utf8)
        ).otherwise(
            None
        ).alias("timeseries_id")
    )
    # save result to outpath
    df.write_csv(clean_data_path, separator="\t")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUSAGE: ./{} <raw data path> <output path>"
              "\n\n".format(sys.argv[0]))
    else:
        main(sys.argv[1],
             sys.argv[2])
