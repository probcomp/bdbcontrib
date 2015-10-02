import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rc

try:
    import seaborn
except ImportError:
    pass

# Set up matplotlib.
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['font.size'] = 24
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 24
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['legend.frameon'] = False

# Global constants.
GM = 398600.4418
EARTH_RADIUS = 6378
T = 1436

# Helper functions.
def compute_period(apogee_km, perigee_km):
    """Computes the period of the satellite in seconds given the apogee_km
    and perigee_km of the satellite.
    """
    a = 0.5*(abs(apogee_km) + abs(perigee_km)) + EARTH_RADIUS
    T = 2 * np.pi * np.sqrt(a**3/GM) / 60.
    return T

def compute_a(T):
    a = ( (60*T/(2*np.pi))**2 * GM)**(1./3)
    return a

def compute_T(a):
    T = 2 * np.pi * np.sqrt(a**3/GM) / 60.
    return T

# Read simulated data.
KC = pd.read_csv('simulated/kc.csv')
DC = pd.read_csv('simulated/dc.csv')
EC = pd.read_csv('simulated/ec.csv')

# Plot of the samples.
fig, ax = plt.subplots()
ax.scatter(KC['Apogee_km'], KC['Perigee_km'],
    color='green', label='Orbital Model Simulation')
ax.scatter(DC['Apogee_km'], DC['Perigee_km'],
    color='red', label='Default Model Simulation')
ax.scatter(EC['Apogee_km'], EC['Perigee_km'],
    color='blue', label='Emperical Data')

ecc = np.linspace(-1,1,100)
implied_apogee = (compute_a(T))*(1+ecc) - EARTH_RADIUS
implied_perigee = (compute_a(T))*(1-ecc) - EARTH_RADIUS
ax.plot(implied_perigee, implied_apogee, color='purple',
    label='Eccentricity [-1,1]')

ax.set_title('Simulated and Observed (Apogee, Perigee) Given Period = 1436')
ax.set_xlabel('Apogee [km]')
ax.set_ylabel('Perigee [km]')
ax.set_xlim([23000, 52000])
ax.set_ylim([23000, 42000])
ax.legend()


# Plot for Implied Period and SMA.
fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Derived Orbital Statistics From Simulated '
    '(Apogee, Perigee) Given Period = 1436',
    )
xs = range(len(KC))
# -- Period.
KC_period = compute_period(KC['Apogee_km'], KC['Perigee_km'])
DC_period = compute_period(DC['Apogee_km'], DC['Perigee_km'])

ax.scatter(xs, KC_period, color='green', label='Orbital Model')
ax.scatter(xs, DC_period, color='red', label='Default Model')
ax.hlines(T, min(xs), max(xs), color='black', label='Theoretical Period')

ax.set_title('Implied Period')
ax.set_xlabel('Simulation Trial')
ax.set_xlim([min(xs), max(xs)])
ax.legend(loc=3)
# -- SMA.
KC_sma = 0.5*(KC['Apogee_km']+KC['Perigee_km']) + EARTH_RADIUS
DC_sma = 0.5*(DC['Apogee_km']+DC['Perigee_km']) + EARTH_RADIUS

ax2.scatter(xs, KC_sma, color='green', label='Orbital Model Simulation')
ax2.scatter(xs, DC_sma, color='red', label='Default Model Simulation')
ax2.hlines(compute_a(T), min(xs), max(xs), color='black',
    label='Theoretical Semi-Major Axis')

ax2.set_title('Implied Semi-Major Axis')
ax2.set_xlabel('Simulation Trial')
ax2.set_xlim([min(xs), max(xs)])


# Overall Residual Plot.
fig, ax = plt.subplots(nrows=1,ncols=2)
ax=ax.ravel()
fig.suptitle('Residuals of Derived Orbital Statistics From Simulated '
    '(Apogee, Perigee) Given Period = 1436')

KC_residT = KC_period - T
DC_residT = DC_period - T
ax[0].set_title('Implied Period')
ax[0].set_xlabel('Residual Minutes')
ax[0].hist(DC_residT, bins=50, color='red', alpha=0.2, label='Default Model')
ax[0].hist(KC_residT, bins=50, color='green', alpha=1., label='Oribtal Model')
ax[0].set_ylim(0, ax[0].get_ylim()[1]+2)
ax[0].legend(loc=2)

KC_residA = KC_sma - compute_a(T)
DC_residA = DC_sma - compute_a(T)
ax[1].set_title('Implied Semi-Major Axis')
ax[1].set_xlabel('Residual Kilometers')
ax[1].hist(DC_residA, bins=50, color='red', alpha=0.2, label='Default Model')
ax[1].hist(KC_residA, bins=50, color='green', alpha=1., label='Orbital Model')
ax[1].set_ylim(0, ax[1].get_ylim()[1]+2)


# Apogee vs Period Simulations
KJ = pd.read_csv('simulated/kj.csv')
DJ = pd.read_csv('simulated/dj.csv')
EJ = pd.read_csv('simulated/ej.csv')

fig, ax = plt.subplots()
ax.scatter(KJ['Apogee_km'], KJ['Period_minutes'],
    color='green', label='Orbital Model Simulation')
ax.scatter(DJ['Apogee_km'], DJ['Period_minutes'],
    color='red', label='Default Model Simulation')
ax.scatter(EJ['Apogee_km'], EJ['Period_minutes'],
    color='blue', label='Empirical Data')

#  -- Parameterize by eccentricity.
apogees = np.linspace(np.min(EJ['Apogee_km']), np.max(EJ['Apogee_km']), 100)
for ecc in [0.0, 0.4, 0.8]:
    perigees_ecc = (apogees + EARTH_RADIUS) * (1-ecc)/(1+ecc)- EARTH_RADIUS
    periods_ecc = compute_period(apogees, perigees_ecc)
    ax.plot(apogees, periods_ecc, color='purple', linestyle='dashed',
        label='Theoretical [ecc={:1.1f}]'.format(ecc))

ax.set_title('Observations and Simulations from Joint Distribution of '
    '(Apogee, Period)')
ax.set_xlabel('Apogee [km]')
ax.set_ylabel('Period [Minutes]')
ax.set_xlim([-200, 48000])
ax.set_ylim([-20, 1700])
ax.legend(loc=4)