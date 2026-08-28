#--------------
# HOST
#--------------
# Code to compute the probability that a transiting S-type planet
# orbits either the primary or the secondary star.
#
# Input files: 
#   - input_stellar_params.csv
#   - input_transit_params.csv
# Important! Both files must have the same list of planets in the same order. 
#
# The host probabilties can be computed with different stellar models, which impact the value of the "model density". These different stellar models are: 
#--------- STELLAR MODELS --------#
# INPUT_MODEL = 0 # --> uses tailored model stellar densities like the ones computed in CHEOPS by TS3.  
# BARAFFE_2015 = 1 # --> stellar evolution models for Baraffe, for solar metalicty, at different ages of evolution (also to be defined).
# BARAFFE_MANN = 2 # --> same as above but using the fit from Mann+2015 for M-dwarfs
# BERGER_2026 = 3 # --> uses fit to Berger+206 data.
#
# The model is set in in the variable "which_stellar_model". 
# We recommend to use either INPUT_MODEL (best option if tailored model densities are available) or BERGER_2026.
#-----------------------------------------------------------
# IMPORTANT:
# Berger+2026 polynomial is fitted using scaled Teff, to get a stable fit:
#
#     Teff_scaled = (Teff - x_mean) / x_std
#
# Therefore the corresponding x_mean and x_std must be loaded
# together with the polynomial coefficients.
#-----------------------------------------------------------------
# @ Arianna Nigioni & Julia Venturini. UNIGE. Version August 2026.
#-----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import math as math
import seaborn as sns
import scipy.stats as stats
import sys
import functions


#--------- CONSTANTS --------#
AU = 1.496e13              # cm
MSUN = 1.9885e33           # g
RSUN = 6.957e10            # cm
REARTH = 4.26352e-5 * AU   # cm

FACTOR_DAYS_TO_SEC = 24 * 3600
FACTOR_HOURS_TO_SEC = 3600

print("                                          ")
print("==========================================")
print("           Host Probability Code          ")
print("==========================================")


#--------- SETTINGS --------#
# Flags to generate output data files, plot and save figures
print_probabilties_on_screen = True
plot_probability_functions = False


#--------- STELLAR MODELS --------#
INPUT_MODEL = 0
BARAFFE_2015 = 1
BARAFFE_MANN = 2
BERGER_2026 = 3

# SET which stellar model to implement!
which_stellar_model = BERGER_2026


#--------- BARAFFE SETTINGS --------#
which_Baraffe_to_plot = None

if which_stellar_model in (BARAFFE_2015, BARAFFE_MANN):
    which_Baraffe_to_plot = 0
    # 0 = 1 Gy
    # 1 = 2 Gy
    # 2 = 3 Gy
    # 3 = 4 Gy
    # 4 = 5 Gy
    # 5 = 8 Gy
    # 6 = 10 Gy


#--------- JOHNSON SU SETTINGS --------#
# Shape parameter controlling kurtosis
delta = 2.


#==============================================================
# LOAD STELLAR MODEL FILES
#==============================================================
if which_stellar_model == INPUT_MODEL:

    # Use the densities provided directly in the input file
    pass


elif which_stellar_model == BARAFFE_2015:

    # Check if the Baraffe model folder exists
    data_folder_Baraffe = Path('B15models_from1to10Gy')

    if not data_folder_Baraffe.exists() or not data_folder_Baraffe.is_dir():
        sys.exit(
            f"Baraffe model folder not found: {data_folder_Baraffe}"
        )

    Baraffe_files = [
        "Baraffe15_1Gy",
        "Baraffe15_2Gy",
        "Baraffe15_3Gy",
        "Baraffe15_4Gy",
        "Baraffe15_5Gy",
        "Baraffe15_8Gy",
        "Baraffe15_10Gy"
    ]

    file = data_folder_Baraffe / Baraffe_files[which_Baraffe_to_plot]

    if not file.exists() or not file.is_file():
        sys.exit(
            f"Baraffe model file not found: {file}"
        )

    Baraffe_table = pd.read_csv(
        file,
        sep=r'\s+'
    )


elif which_stellar_model == BARAFFE_MANN:

    # Check if the Baraffe model folder exists
    data_folder_Baraffe = Path('B15models_from1to10Gy')

    if not data_folder_Baraffe.exists() or not data_folder_Baraffe.is_dir():
        sys.exit(
            f"Baraffe model folder not found: {data_folder_Baraffe}"
        )

    Baraffe_files = [
        "Baraffe15_1Gy",
        "Baraffe15_2Gy",
        "Baraffe15_3Gy",
        "Baraffe15_4Gy",
        "Baraffe15_5Gy",
        "Baraffe15_8Gy",
        "Baraffe15_10Gy"
    ]

    file = data_folder_Baraffe / Baraffe_files[which_Baraffe_to_plot]

    if not file.exists() or not file.is_file():
        sys.exit(
            f"Baraffe model file not found: {file}"
        )

    Baraffe_table = pd.read_csv(
        file,
        sep=r'\s+'
    )

    # Check if Mann+2015 file exists
    file_data_Mann = Path('Mann+2015_data.csv')

    if not file_data_Mann.exists() or not file_data_Mann.is_file():
        sys.exit(
            f"Mann+2015 file not found: {file_data_Mann}"
        )

    data_Mann = pd.read_csv(
        file_data_Mann,
        delimiter=','
    )


elif which_stellar_model == BERGER_2026:

    #==========================================================
    # LOAD BERGER POLYNOMIAL
    #==========================================================
    berger_folder = Path(
        "Berger2026_stellar-params"
    )

    coeff_file = (
        berger_folder /
        "polyfit_coeffs_Berger.npy"
    )

    scaling_file = (
        berger_folder /
        "polyfit_x_mean_std_Berger.npy"
    )


    # Check polynomial coefficients file
    if not coeff_file.exists() or not coeff_file.is_file():
        sys.exit(
            f"Berger polynomial coefficient file not found: "
            f"{coeff_file}"
        )


    # Check scaling file
    if not scaling_file.exists() or not scaling_file.is_file():
        sys.exit(
            f"Berger polynomial scaling file not found: "
            f"{scaling_file}"
        )


    # Load polynomial coefficients
    coeff_fit_Berger = np.load(
        coeff_file
    )


    # Load x_mean and x_std
    berger_scaling = np.load(
        scaling_file
    )

    x_mean = berger_scaling[0]
    x_std = berger_scaling[1]


    print("Berger+2026 polynomial loaded.")
    print(
        f"Berger Teff mean = {x_mean:.3f} K"
    )
    print(
        f"Berger Teff std  = {x_std:.3f} K"
    )

    print(
        "Berger polynomial coefficients:"
    )
    print(coeff_fit_Berger)


#==============================================================
# LOAD INPUT FILES
#==============================================================
file_input_data = Path(
    'input_transit_params.csv'
)

file_stellar_properties = Path(
    'input_stellar_params.csv'
)


# Check existence of input files

if (
    not file_input_data.exists()
    or not file_input_data.is_file()
):
    sys.exit(
        f"Input file not found: {file_input_data}"
    )


if (
    not file_stellar_properties.exists()
    or not file_stellar_properties.is_file()
):
    sys.exit(
        f"Stellar properties file not found: "
        f"{file_stellar_properties}"
    )


# Read CSV data

data = pd.read_csv(
    file_input_data,
    delimiter=','
)

stellar_properties = pd.read_csv(
    file_stellar_properties,
    delimiter=','
)


#==============================================================
# REQUIRED COLUMNS
#==============================================================
required_data_cols = [
    'target',
    'rho1',
    'rho1_err_up',
    'rho1_err_low',
    'rho2',
    'rho2_err_up',
    'rho2_err_low'
]


required_stellar_cols = [
    'Teff1',
    'e_Teff1_up',
    'e_Teff1_low',
    'Teff2',
    'e_Teff2_up',
    'e_Teff2_low'
]


if which_stellar_model == INPUT_MODEL:

    required_stellar_cols += [
        'rho1',
        'e_rho1_up',
        'e_rho1_low',
        'rho2',
        'e_rho2_up',
        'e_rho2_low'
    ]


# Check missing columns

missing_data_cols = [
    col
    for col in required_data_cols
    if col not in data.columns
]

if missing_data_cols:
    sys.exit(
        f"Missing columns in data: "
        f"{missing_data_cols}"
    )


missing_stellar_cols = [
    col
    for col in required_stellar_cols
    if col not in stellar_properties.columns
]

if missing_stellar_cols:
    sys.exit(
        f"Missing columns in stellar_properties: "
        f"{missing_stellar_cols}"
    )


#==============================================================
# EXTRACT DATA
#==============================================================
# Transit densities

target = data['target']

rho1_transit = data['rho1']
rho1_transit_err_up = data['rho1_err_up']
rho1_transit_err_low = data['rho1_err_low']

rho2_transit = data['rho2']
rho2_transit_err_up = data['rho2_err_up']
rho2_transit_err_low = data['rho2_err_low']


# Stellar effective temperatures

Teff1 = stellar_properties['Teff1']
Teff1_err_up = stellar_properties['e_Teff1_up']
Teff1_err_low = stellar_properties['e_Teff1_low']

Teff2 = stellar_properties['Teff2']
Teff2_err_up = stellar_properties['e_Teff2_up']
Teff2_err_low = stellar_properties['e_Teff2_low']


#==============================================================
# CALCULATE MODEL DENSITIES
#==============================================================
if which_stellar_model == INPUT_MODEL:

    # Read model densities directly from input

    rho1_model = stellar_properties['rho1']
    rho1_model_err_up = stellar_properties['e_rho1_up']
    rho1_model_err_low = stellar_properties['e_rho1_low']

    rho2_model = stellar_properties['rho2']
    rho2_model_err_up = stellar_properties['e_rho2_up']
    rho2_model_err_low = stellar_properties['e_rho2_low']


else:

    rho1_model = []
    rho1_model_err_up = []
    rho1_model_err_low = []

    rho2_model = []
    rho2_model_err_up = []
    rho2_model_err_low = []


    #==========================================================
    # BARAFFE 2015
    #==========================================================
    if which_stellar_model == BARAFFE_2015:

        min_Teff = 3000

        filtered_data = Baraffe_table[
            Baraffe_table['Teff'] > min_Teff
        ]

        max_Teff = max(
            filtered_data['Teff']
        )


        density_Baraffe_data = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data['M/Ms'] * MSUN,
                filtered_data['R/Rs'] * RSUN
            )
        )


        # Switch temperature

        if which_Baraffe_to_plot in [0, 2, 3]:
            switch = 3520

        elif which_Baraffe_to_plot == 1:
            switch = 3519

        elif which_Baraffe_to_plot == 4:
            switch = 3521

        elif which_Baraffe_to_plot == 5:
            switch = 3522

        elif which_Baraffe_to_plot == 6:
            switch = 3524


        # Above switch

        filtered_data_above_switch = (
            filtered_data[
                filtered_data['Teff'] >= switch
            ]
        )

        density_Baraffe_above_switch = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data_above_switch['M/Ms'] * MSUN,
                filtered_data_above_switch['R/Rs'] * RSUN
            )
        )


        coeff_fit_Baraffe_above_switch = np.polyfit(
            filtered_data_above_switch['Teff'],
            density_Baraffe_above_switch,
            6
        )


        # Below switch

        filtered_data_below_switch = (
            filtered_data[
                filtered_data['Teff'] <= switch
            ]
        )

        density_Baraffe_below_switch = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data_below_switch['M/Ms'] * MSUN,
                filtered_data_below_switch['R/Rs'] * RSUN
            )
        )


        coeff_fit_Baraffe_below_switch = np.polyfit(
            filtered_data_below_switch['Teff'],
            density_Baraffe_below_switch,
            6
        )


        # Evaluate polynomial for each target

        for i in range(len(target)):

            if Teff1[i] < switch:

                rho1_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_below_switch,
                        Teff1[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff1[i],
                    coeff_fit_Baraffe_below_switch,
                    Teff1_err_low[i],
                    Teff1_err_up[i],
                    rho1_model_err_low,
                    rho1_model_err_up
                )

            if Teff1[i] >= switch:

                rho1_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_above_switch,
                        Teff1[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff1[i],
                    coeff_fit_Baraffe_above_switch,
                    Teff1_err_low[i],
                    Teff1_err_up[i],
                    rho1_model_err_low,
                    rho1_model_err_up
                )


            if Teff2[i] < switch:

                rho2_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_below_switch,
                        Teff2[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff2[i],
                    coeff_fit_Baraffe_below_switch,
                    Teff2_err_low[i],
                    Teff2_err_up[i],
                    rho2_model_err_low,
                    rho2_model_err_up
                )

            if Teff2[i] >= switch:

                rho2_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_above_switch,
                        Teff2[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff2[i],
                    coeff_fit_Baraffe_above_switch,
                    Teff2_err_low[i],
                    Teff2_err_up[i],
                    rho2_model_err_low,
                    rho2_model_err_up
                )


    #==========================================================
    # BARAFFE + MANN
    #==========================================================
    if which_stellar_model == BARAFFE_MANN:

        min_Teff = 3000

        filtered_data = Baraffe_table[
            Baraffe_table['Teff'] > min_Teff
        ]

        max_Teff = max(
            filtered_data['Teff']
        )

        filtered_data = filtered_data[
            filtered_data['Teff'] <= max_Teff
        ]


        density_Baraffe_data = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data['M/Ms'] * MSUN,
                filtered_data['R/Rs'] * RSUN
            )
        )


        # Switch temperature

        if which_Baraffe_to_plot in [0, 2, 3]:
            switch = 3520

        elif which_Baraffe_to_plot == 1:
            switch = 3519

        elif which_Baraffe_to_plot == 4:
            switch = 3521

        elif which_Baraffe_to_plot == 5:
            switch = 3522

        elif which_Baraffe_to_plot == 6:
            switch = 3524


        # Above switch

        filtered_data_above_switch = (
            filtered_data[
                filtered_data['Teff'] >= switch
            ]
        )

        density_Baraffe_above_switch = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data_above_switch['M/Ms'] * MSUN,
                filtered_data_above_switch['R/Rs'] * RSUN
            )
        )


        coeff_fit_Baraffe_above_switch = np.polyfit(
            filtered_data_above_switch['Teff'],
            density_Baraffe_above_switch,
            6
        )


        # Below switch

        filtered_data_below_switch = (
            filtered_data[
                filtered_data['Teff'] <= switch
            ]
        )

        density_Baraffe_below_switch = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                filtered_data_below_switch['M/Ms'] * MSUN,
                filtered_data_below_switch['R/Rs'] * RSUN
            )
        )


        coeff_fit_Baraffe_below_switch = np.polyfit(
            filtered_data_below_switch['Teff'],
            density_Baraffe_below_switch,
            6
        )


        # Mann model

        density_Mann = (
            functions.calculate_model_stellar_density_from_mass_and_radius(
                data_Mann['M/Ms'] * MSUN,
                data_Mann['R/Rs'] * RSUN
            )
        )


        coeff_fit_Mann = np.polyfit(
            data_Mann['Teff'],
            density_Mann,
            4
        )


        # Evaluate polynomial for each target

        for i in range(len(target)):

            if Teff1[i] < 4000:

                rho1_model.append(
                    np.polyval(
                        coeff_fit_Mann,
                        Teff1[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff1[i],
                    coeff_fit_Mann,
                    Teff1_err_low[i],
                    Teff1_err_up[i],
                    rho1_model_err_low,
                    rho1_model_err_up
                )

            else:

                rho1_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_above_switch,
                        Teff1[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff1[i],
                    coeff_fit_Baraffe_above_switch,
                    Teff1_err_low[i],
                    Teff1_err_up[i],
                    rho1_model_err_low,
                    rho1_model_err_up
                )


            if Teff2[i] < 4000:

                rho2_model.append(
                    np.polyval(
                        coeff_fit_Mann,
                        Teff2[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff2[i],
                    coeff_fit_Mann,
                    Teff2_err_low[i],
                    Teff2_err_up[i],
                    rho2_model_err_low,
                    rho2_model_err_up
                )

            else:

                rho2_model.append(
                    np.polyval(
                        coeff_fit_Baraffe_above_switch,
                        Teff2[i]
                    )
                )

                functions.calculate_error_on_rho_from_fit(
                    Teff2[i],
                    coeff_fit_Baraffe_above_switch,
                    Teff2_err_low[i],
                    Teff2_err_up[i],
                    rho2_model_err_low,
                    rho2_model_err_up
                )


    
    #==========================================================
    # BERGER 2026
    #==========================================================
    if which_stellar_model == BERGER_2026:

        #------------------------------------------------------
        # IMPORTANT:
        #
        # The Berger polynomial was fitted using:
        #
        # Teff_scaled = (Teff - x_mean) / x_std
        #
        # Therefore both the temperatures AND their errors
        # must be transformed before using the polynomial.
        #------------------------------------------------------

        for i in range(len(target)):

            #==================================================
            # PRIMARY STAR
            #==================================================

            Teff1_scaled = (
                Teff1[i] - x_mean
            ) / x_std


            Teff1_err_low_scaled = (
                Teff1_err_low[i]
                / x_std
            )

            Teff1_err_up_scaled = (
                Teff1_err_up[i]
                / x_std
            )


            # Calculate model density

            rho1_model.append(
                np.polyval(
                    coeff_fit_Berger,
                    Teff1_scaled
                )
            )


            # Propagate Teff uncertainty
            #
            # The polynomial now expects scaled Teff,
            # so we pass the scaled Teff and scaled errors.

            functions.calculate_error_on_rho_from_fit(
                Teff1_scaled,
                coeff_fit_Berger,
                Teff1_err_low_scaled,
                Teff1_err_up_scaled,
                rho1_model_err_low,
                rho1_model_err_up
            )


            #==================================================
            # SECONDARY STAR
            #==================================================

            Teff2_scaled = (
                Teff2[i] - x_mean
            ) / x_std


            Teff2_err_low_scaled = (
                Teff2_err_low[i]
                / x_std
            )

            Teff2_err_up_scaled = (
                Teff2_err_up[i]
                / x_std
            )


            # Calculate model density

            rho2_model.append(
                np.polyval(
                    coeff_fit_Berger,
                    Teff2_scaled
                )
            )


            # Propagate Teff uncertainty

            functions.calculate_error_on_rho_from_fit(
                Teff2_scaled,
                coeff_fit_Berger,
                Teff2_err_low_scaled,
                Teff2_err_up_scaled,
                rho2_model_err_low,
                rho2_model_err_up
            )


#==============================================================
# CALCULATION OF PROBABILITY
#==============================================================
# Initialize lists to store asymmetric Gaussian parameters
# and PDFs

mu1_t = []
y1_t = []

mu1_m = []
y1_m = []

mu2_t = []
y2_t = []

mu2_m = []
y2_m = []


probability_primary_host = []
probability_secondary_host = []


rho_max = 30.


#==============================================================
# LOOP OVER ALL TARGETS
#==============================================================
for i in range(len(target)):

    # Dense grid for density values

    x = np.linspace(
        0,
        rho_max,
        100000
    )


    #----------------------------------------------------------
    # Primary star transit PDF
    #----------------------------------------------------------

    mu1_t.append(
        rho1_transit[i]
    )

    sigma_left1_t = (
        rho1_transit_err_low[i]
    )

    sigma_right1_t = (
        rho1_transit_err_up[i]
    )


    gamma1_t = -(
        sigma_right1_t - sigma_left1_t
    ) / (
        sigma_left1_t + sigma_right1_t
    )


    delta1_t = delta


    johnson_su1_t = stats.johnsonsu(
        gamma1_t,
        delta1_t,
        loc=mu1_t[i],
        scale=sigma_right1_t
    )


    y1_t.append(
        johnson_su1_t.pdf(x)
    )


    #----------------------------------------------------------
    # Primary star model PDF
    #----------------------------------------------------------

    mu1_m.append(
        rho1_model[i]
    )

    sigma_left1_m = (
        rho1_model_err_low[i]
    )

    sigma_right1_m = (
        rho1_model_err_up[i]
    )


    gamma1_m = -(
        sigma_right1_m - sigma_left1_m
    ) / (
        sigma_left1_m + sigma_right1_m
    )


    delta1_m = delta


    johnson_su1_m = stats.johnsonsu(
        gamma1_m,
        delta1_m,
        loc=mu1_m[i],
        scale=sigma_right1_m
    )


    y1_m.append(
        johnson_su1_m.pdf(x)
    )


    #----------------------------------------------------------
    # Secondary star transit PDF
    #----------------------------------------------------------

    mu2_t.append(
        rho2_transit[i]
    )

    sigma_left2_t = (
        rho2_transit_err_low[i]
    )

    sigma_right2_t = (
        rho2_transit_err_up[i]
    )


    gamma2_t = -(
        sigma_right2_t - sigma_left2_t
    ) / (
        sigma_left2_t + sigma_right2_t
    )


    delta2_t = delta


    johnson_su2_t = stats.johnsonsu(
        gamma2_t,
        delta2_t,
        loc=mu2_t[i],
        scale=sigma_right2_t
    )


    y2_t.append(
        johnson_su2_t.pdf(x)
    )


    #----------------------------------------------------------
    # Secondary star model PDF
    #----------------------------------------------------------
    mu2_m.append(
        rho2_model[i]
    )

    sigma_left2_m = (
        rho2_model_err_low[i]
    )

    sigma_right2_m = (
        rho2_model_err_up[i]
    )


    gamma2_m = -(
        sigma_right2_m - sigma_left2_m
    ) / (
        sigma_left2_m + sigma_right2_m
    )


    delta2_m = delta


    johnson_su2_m = stats.johnsonsu(
        gamma2_m,
        delta2_m,
        loc=mu2_m[i],
        scale=sigma_right2_m
    )


    y2_m.append(
        johnson_su2_m.pdf(x)
    )


    #----------------------------------------------------------
    # Calculate host probabilities
    #----------------------------------------------------------
    functions.calculate_probability(
        x,
        y1_t[i],
        y1_m[i],
        y2_t[i],
        y2_m[i],
        probability_primary_host,
        probability_secondary_host,
        target[i],
        plot_probability_functions
    )


#==============================================================
# GENERATE OUTPUT DATA FILE
#==============================================================
data_out = {
    'target': target,

    'Prob_1': probability_primary_host,
    'Prob_2': probability_secondary_host,

    'rho1_transit': rho1_transit,
    'err_rho1_transit_up': rho1_transit_err_up,
    'err_rho1_transit_low': rho1_transit_err_low,

    'rho2_transit': rho2_transit,
    'err_rho2_transit_up': rho2_transit_err_up,
    'err_rho2_transit_low': rho2_transit_err_low,

    'rho1_model': rho1_model,
    'err_rho1_model_up': rho1_model_err_up,
    'err_rho1_model_low': rho1_model_err_low,

    'rho2_model': rho2_model,
    'err_rho2_model_up': rho2_model_err_up,
    'err_rho2_model_low': rho2_model_err_low
}



#==============================================================
# CREATE DATAFRAME
#==============================================================
df_out = pd.DataFrame(
    data_out
)


#==============================================================
# SAVE OUTPUT FILE
#==============================================================
output_file = {
    INPUT_MODEL:
        "output_host-probabilties_stellar_parameters.csv",

    BARAFFE_2015:
        "output_host-probabilties_baraffe2015.csv",

    BARAFFE_MANN:
        "output_host-probabilties_baraffe2015_mann2015.csv",

    BERGER_2026:
        "output_host-probabilties_berger2026.csv"

}[which_stellar_model]


df_out.to_csv(
    output_file,
    header=True,
    index=None,
    sep=','
)


#==============================================================
# PRINT PROBABILITY ON SCREEN
#==============================================================

if print_probabilties_on_screen:

   stellar_model_names = {
    INPUT_MODEL: "Input stellar parameters",
    BARAFFE_2015: "Baraffe+2015",
    BARAFFE_MANN: "Baraffe+2015 + Mann+2015",
    BERGER_2026: "Berger+2026"
}
print(" ")
print(
    "stellar model:",
    stellar_model_names[which_stellar_model]
)

data_print = {
    'Target': target,

    'probability_primary_host':
        probability_primary_host,

    'probability_secondary_host':
         probability_secondary_host
    }


df_print = pd.DataFrame(
     data_print
    )


print(
    df_print
    )

#==============================================================

