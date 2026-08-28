#--------------
# HOST-S functions
#--------------
# Code to compute the probability that an S-type planet orbits either the primary or the secondary star.
# Main developer: Arianna Nigioni. Co-developer: Julia Venturini. (Version August 2025).

# File containing the function definitions used in the main Host_probability_code.py

import numpy as np
import matplotlib. pyplot as plt 
import matplotlib as mpl
import math as math
from scipy.integrate import simpson
import seaborn as sns

#--------- FUNCTIONS ---------#
        
def ask_bool(question, default=True):
    """
    Ask the user a True/False question from terminal.
    Keeps asking until input is valid.
    """
    default_str = "True" if default else "False"
    while True:
        val = input(f"{question} (True/False) [default={default_str}]: ") or default_str
        val = val.strip().lower()
        if val in ("True","true", "1", "yes", "y", "t"):
            return True
        elif val in ("False","false", "0", "no", "n", "f"):
            return False
        else:
            print("⚠️ Please enter True/true/1/yes/y/t or False/false/0/no/n/f (or press Enter for default).")
            
def ask_int(question, default=0, min_val=None, max_val=None):
    """
    Ask the user an integer value from terminal.
    Keeps asking until input is valid.
    """
    while True:
        val = input(f"{question} [default={default}]: ") or str(default)
        try:
            val = int(val)
            if min_val is not None and val < min_val:
                print(f"⚠️ Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"⚠️ Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("⚠️ Please enter a valid integer.")

def ask_float(question, default=0.0, min_val=None, max_val=None):
    """
    Ask the user a float value from terminal.
    Keeps asking until input is valid.
    """
    while True:
        val = input(f"{question} [default={default}]: ") or str(default)
        try:
            val = float(val)
            if min_val is not None and val < min_val:
                print(f"⚠️ Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"⚠️ Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("⚠️ Please enter a valid number.")

def calculate_error_on_rho_from_fit(Teff,coeff,sigma_Teff_low,sigma_Teff_up,sigma_low, sigma_up):
    """
    Calculate the propagated error on stellar density from a polynomial fit.
    Append the results to the lists sigma_low and sigma_up.
    """
    derivative = np.polyder(coeff)
    dsigma_dTeff = np.polyval(derivative, Teff)
    sigma_low.append(np.abs(dsigma_dTeff)*sigma_Teff_low)
    sigma_up.append(np.abs(dsigma_dTeff)*sigma_Teff_up) 

def calculate_model_stellar_density_from_mass_and_radius(mass,radius):
    """
    Calculate stellar density from mass and radius: rho = M / (4/3*pi*R^3)
    """
    PI = np.pi
    factor = (4/3)*PI
    return mass/(factor*radius**3)


def calculate_probability(x, y1, y1_model, y2, y2_model, probability_primary_host, probability_secondary_host,target,plot_probability_functions):
    """
    Calculate the probability that a planet orbits the primary or secondary star.
    Uses Simpson integration of the product of transit and model PDFs.
    """
    # Integrate the product of PDFs for primary and secondary
    integral1 = simpson(y1 * y1_model, x)
    integral2 = simpson(y2 * y2_model, x)
 
    # Calculate and normalise the probabilities
    prob1 = (integral1 / (integral1+integral2))
    prob2 = (integral2 / (integral1+integral2))
 
    prob1 = prob1*100
    prob2 = prob2*100
       
    probability_primary_host.append(prob1)
    probability_secondary_host.append(prob2)
 
    # Optionally plot the probability distributions
    if (plot_probability_functions):
 
        sns.set(style='ticks', font='STIXGeneral', color_codes=True)
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.rcParams['font.size'] = 18
        
        plt.plot(x, y1, label=r'$\rho_{1,\mathrm{transit}}$', color='coral')
        plt.plot(x, y1_model, label=r'$\rho_{1,\mathrm{model}}$', color='blue')
        plt.plot(x, y2, label=r'$\rho_{2,\mathrm{transit}}$', color='orange')
        plt.plot(x, y2_model, label=r'$\rho_{2,\mathrm{model}}$', color='cornflowerblue')
        plt.xlabel(r"$\rho$")
        plt.ylabel("Probability density function")
        plt.title(f"({target})")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y1 * y1_model, label=r'$y_{1,\mathrm{transit}} \cdot y_{1,\mathrm{model}}$', color='blue')
        plt.fill_between(x, y1 * y1_model, alpha=0.3, color='blue') # color the overlapping area
        plt.xlabel("x")
        plt.ylabel("Product of PDFs")
        plt.title(r"Integral of $y_{1,\mathrm{transit}} \cdot y_{1,\mathrm{model}}$ " + f"({target})")
        plt.legend()
 
        plt.subplot(1, 2, 2)
        plt.plot(x, y2 * y2_model, label=r'$y_{2,\mathrm{transit}} \cdot y_{2,\mathrm{model}}$', color='red')
        plt.fill_between(x, y2 * y2_model, alpha=0.3, color='red') # color the overlapping area
        plt.xlabel("x")
        plt.ylabel("Product of PDFs")
        plt.title(r"Integral of $y_{2,\mathrm{transit}} \cdot y_{2,\mathrm{model}}$ " + f"({target})")
        plt.legend()
 
        plt.tight_layout()
        plt.show()    
 
 
    
#--------- END FUNCTIONS ---------#
