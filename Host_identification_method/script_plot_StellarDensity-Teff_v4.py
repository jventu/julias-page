# Script to plot stellar densities from transits and from stellar parameters,
# including Berger+2026 fit.
# Input files: 
#   - input_stellar_params.csv
#   - input_transit_params.csv
# Important! Both files must have the same list of planets in the same order. 
#
# @ Arianna Nigioni & Julia Venturini. UNIGE. Version August 2026.
#
# IMPORTANT:
# Berger+2026 polynomial is fitted using scaled Teff, to get a stable fit:
#
#     Teff_scaled = (Teff - x_mean) / x_std
#
# Therefore the corresponding x_mean and x_std must be loaded
# together with the polynomial coefficients.


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MultipleLocator


# ---------- SETTINGS ----------
sns.set(style='ticks', font='STIXGeneral', color_codes=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 16


# ---------- CONSTANTS ----------
RHO_MAX = 30


# ---------- PLOT SETTINGS ----------
save_figs = True
FONTSIZE = 22
TICKSIZE = 18


# ============================================================
# AXIS FORMATTING FUNCTION
# ============================================================

def format_axes(ax, title):

    ax.set_title(title, fontsize=FONTSIZE)

    ax.set_xlabel(
        r"T$_{\mathrm{eff}}$ [K]",
        fontsize=FONTSIZE
    )

    ax.set_ylabel(
        "Stellar density [cgs]",
        fontsize=FONTSIZE
    )

    ax.set_xlim(6500, 2500)
    ax.set_ylim(0, RHO_MAX)

    # ----- TICKS STYLE -----

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=TICKSIZE,
        length=10,
        width=1.5
    )

    ax.tick_params(
        axis='both',
        which='minor',
        length=4,
        width=0.8
    )

    ax.minorticks_on()

    # X-axis ticks
    ax.set_xticks(np.arange(2500, 6501, 500))

    ax.xaxis.set_minor_locator(
        MultipleLocator(100)
    )

    # Y-axis ticks
    ax.set_yticks(
        np.linspace(0, RHO_MAX, 7)
    )

    ax.yaxis.set_minor_locator(
        MultipleLocator(1)
    )


# ============================================================
# OUTPUT FOLDER
# ============================================================
output_folder = Path("output_plots")
output_folder.mkdir(exist_ok=True)


# ============================================================
# LOAD BERGER MODEL
# ============================================================
# Load polynomial coefficients
coeffs_Berger = np.load(
    "Berger2026_stellar-params/polyfit_coeffs_Berger.npy"
)

# Load the mean and standard deviation used to scale Teff
berger_scaling = np.load(
    "Berger2026_stellar-params/polyfit_x_mean_std_Berger.npy"
)

x_mean = berger_scaling[0]
x_std = berger_scaling[1]


print("Berger polynomial coefficients:")
print(coeffs_Berger)

print("\nBerger Teff scaling:")
print(f"x_mean = {x_mean:.3f} K")
print(f"x_std  = {x_std:.3f} K")


# ============================================================
# LOAD BERGER CLEANED CSV
# ============================================================
df = pd.read_csv(
    "Berger2026_stellar-params/stellar_parameters_with_Berger_densities.csv"
)


# ============================================================
# FILTER BERGER DATA
# ============================================================
df_filtered = df[
    df["Teff_K"] < 6500
].copy()


df_filtered = df_filtered.dropna(
    subset=[
        "Teff_K",
        "Teff_eup",
        "Teff_elow",
        "FeH_dex",
        "rho_gcm3",
        "rho_err_up",
        "rho_err_low"
    ]
)


# ============================================================
# FILTER BY ERROR IN TEFF < 2%
# ============================================================
df_filtered["Teff_mean_err_percentage"] = (
    100
    * 0.5
    * (
        df_filtered["Teff_eup"]
        + df_filtered["Teff_elow"]
    )
    / df_filtered["Teff_K"]
)

df_filtered = df_filtered[
    df_filtered["Teff_mean_err_percentage"] < 2
].copy()


# ============================================================
# FILTER BY ERROR IN RHO_STAR < 20%
# ============================================================
df_filtered["rho_mean_err_percentage"] = (
    100
    * 0.5
    * (
        df_filtered["rho_err_up"]
        + df_filtered["rho_err_low"]
    )
    / df_filtered["rho_gcm3"]
)

df_filtered = df_filtered[
    df_filtered["rho_mean_err_percentage"] < 20
].copy()


# ============================================================
# BERGER DATA
# ============================================================
x = df_filtered["Teff_K"].values
y = df_filtered["rho_gcm3"].values

x_err_plus = df_filtered["Teff_eup"].values
x_err_minus = df_filtered["Teff_elow"].values

y_err_plus = df_filtered["rho_err_up"].values
y_err_minus = df_filtered["rho_err_low"].values

feh = df_filtered["FeH_dex"].values


# ============================================================
# LOAD TRANSIT DATA
# ============================================================
# very important: both files must contain the same list of planets in the same order!

transit = pd.read_csv(
    "input_transit_params.csv"
)

stellar = pd.read_csv(
    "input_stellar_params.csv"
)


print(transit)
print(stellar)
# ============================================================
# TARGET
# ============================================================
target = transit["target"]

# ============================================================
# TRANSIT DENSITIES
# ============================================================
rho1_transit = transit["rho1"]
rho1_transit_err_up = transit["rho1_err_up"]
rho1_transit_err_low = transit["rho1_err_low"]

rho2_transit = transit["rho2"]
rho2_transit_err_up = transit["rho2_err_up"]
rho2_transit_err_low = transit["rho2_err_low"]


# ============================================================
# EFFECTIVE TEMPERATURES
# ============================================================
Teff1 = stellar["Teff1"]
Teff2 = stellar["Teff2"]

Teff1_err_up = stellar["e_Teff1_up"]
Teff1_err_low = stellar["e_Teff1_low"]

Teff2_err_up = stellar["e_Teff2_up"]
Teff2_err_low = stellar["e_Teff2_low"]


# ============================================================
# MODEL DENSITIES
# ============================================================
has_model_density = "rho1" in stellar.columns

if has_model_density:

    rho1_model = stellar["rho1"]
    rho2_model = stellar["rho2"]

    rho1_model_err_up = stellar["e_rho1_up"]
    rho1_model_err_low = stellar["e_rho1_low"]

    rho2_model_err_up = stellar["e_rho2_up"]
    rho2_model_err_low = stellar["e_rho2_low"]


# ============================================================
# BERGER MODEL CURVE
# ============================================================
Teff_grid = np.linspace(
    2000,
    6800,
    800
)


# IMPORTANT:
# The Berger polynomial was fitted using
#
#     Teff_scaled = (Teff - x_mean) / x_std
#
# Therefore the model grid must be scaled in exactly the
# same way before evaluating the polynomial.

Teff_grid_scaled = (
    Teff_grid - x_mean
) / x_std


rho_fit_Berger = np.polyval(
    coeffs_Berger,
    Teff_grid_scaled
)


# ============================================================
# PLOTTING FUNCTION PER TARGET
# ============================================================
def plot_target(i, ax):

    # --------------------------------------------------------
    # BACKGROUND BERGER SAMPLE
    # --------------------------------------------------------
    sc = ax.scatter(
        x,
        y,
        c=feh,
        cmap="viridis",
        s=25,
        alpha=0.7,
        edgecolors='none',
        zorder=1
    )


    # --------------------------------------------------------
    # BERGER+2026 FIT
    # --------------------------------------------------------
    line_berger, = ax.plot(
        Teff_grid,
        rho_fit_Berger,
        linestyle="--",
        color='#e7298a',
        linewidth=2,
        label="Berger+2026",
        zorder=2
    )


    # --------------------------------------------------------
    # ERROR ARRAYS
    # --------------------------------------------------------
    Teff_err1 = np.array([
        [
            Teff1_err_low[i],
            Teff1_err_up[i]
        ]
    ]).T

    Teff_err2 = np.array([
        [
            Teff2_err_low[i],
            Teff2_err_up[i]
        ]
    ]).T

    rho1_t_err = np.array([
        [
            rho1_transit_err_low[i],
            rho1_transit_err_up[i]
        ]
    ]).T

    rho2_t_err = np.array([
        [
            rho2_transit_err_low[i],
            rho2_transit_err_up[i]
        ]
    ]).T


    # --------------------------------------------------------
    # TRANSIT POINTS
    # Original colors preserved
    # --------------------------------------------------------
    pt1 = ax.errorbar(
        Teff1[i],
        rho1_transit[i],
        xerr=Teff_err1,
        yerr=rho1_t_err,
        fmt='o',
        color='#fdae6b',
        markeredgecolor='black',
        label=r"$\rho_{\star 1, \rm transit}$",
        markersize=10,
        zorder=4
    )


    pt2 = ax.errorbar(
        Teff2[i],
        rho2_transit[i],
        xerr=Teff_err2,
        yerr=rho2_t_err,
        fmt='o',
        color='#e6550d',
        markeredgecolor='black',
        label=r"$\rho_{\star 2, \rm transit}$",
        markersize=10,
        zorder=4
    )


    data_handles = [
        pt1,
        pt2
    ]


    # --------------------------------------------------------
    # MODEL POINTS
    # Original colors preserved
    # --------------------------------------------------------
    if has_model_density:

        rho1_m_err = np.array([
            [
                rho1_model_err_low[i],
                rho1_model_err_up[i]
            ]
        ]).T

        rho2_m_err = np.array([
            [
                rho2_model_err_low[i],
                rho2_model_err_up[i]
            ]
        ]).T


        pt3 = ax.errorbar(
            Teff1[i],
            rho1_model[i],
            xerr=Teff_err1,
            yerr=rho1_m_err,
            fmt='D',
            color='#b2abd2',
            markeredgecolor='black',
            label=r"$\rho_{\star 1, \rm model}$",
            markersize=10,
            zorder=3
        )


        pt4 = ax.errorbar(
            Teff2[i],
            rho2_model[i],
            xerr=Teff_err2,
            yerr=rho2_m_err,
            fmt='D',
            color='#5e3c99',
            markeredgecolor='black',
            label=r"$\rho_{\star 2, \rm model}$",
            markersize=10,
            zorder=3
        )


        data_handles.extend([
            pt3,
            pt4
        ])


    # --------------------------------------------------------
    # LEGENDS
    # --------------------------------------------------------
    legend_fits = ax.legend(
        handles=[line_berger],
        loc="upper left",
        fontsize=18
    )


    legend_points = ax.legend(
        handles=data_handles,
        bbox_to_anchor=(0.4, 0.85),
        fontsize=22,
        handletextpad=0.1
    )


    ax.add_artist(
        legend_fits
    )


    return sc


# ============================================================
# MAIN LOOP
# ============================================================
for i in range(len(target)):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    sc = plot_target(
        i,
        ax
    )


    format_axes(
        ax,
        target[i]
    )


    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------
    cbar = fig.colorbar(
        sc,
        ax=ax
    )

    cbar.set_label(
        "[Fe/H] (dex)",
        fontsize=18
    )

    cbar.ax.tick_params(
        labelsize=18
    )

    sc.set_clim(
        vmin=-2.5,
        vmax=0.5
    )


    ax.set_xlim(
        6500,
        2500
    )


    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------
    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    if save_figs:

        plt.savefig(
            output_folder / f"rho_Teff_{target[i]}.pdf"
        )


    plt.show()
