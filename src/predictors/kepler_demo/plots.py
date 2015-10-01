import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

from matplotlib import rc

GM = 398600.4418
EARTH_RADIUS = 6378

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

T = 1436
KC = pd.read_csv('simulated/kc.csv')
DC = pd.read_csv('simulated/dc.csv')
EC = pd.read_csv('simulated/ec.csv')

KC_period = compute_period(KC['Apogee_km'], KC['Perigee_km'])
DC_period = compute_period(DC['Apogee_km'], DC['Perigee_km'])

TITLE_SIZE = 26
AXES_SIZE = 22
TICK_SIZE = 16
LEGEND_SIZE = 18
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['legend.fontsize'] = LEGEND_SIZE

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

ax.set_title('Simulated and Observed (Apogee, Perigee) Given Period = 1436',
    fontsize=TITLE_SIZE)
ax.set_xlabel('Apogee [km]', fontsize=AXES_SIZE, fontweight='bold')
ax.set_ylabel('Perigee [km]', fontsize=AXES_SIZE, fontweight='bold')
ax.set_xlim([23000, 52000])
ax.set_ylim([23000, 42000])
ax.tick_params(labelsize=TICK_SIZE)
ax.legend()

# Plot for Implied Period.
fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
xs = range(len(KC))

ax.scatter(xs, KC_period, color='green', label='Orbital Model')
ax.scatter(xs, DC_period, color='red', label='Default Model')
ax.hlines(T, min(xs), max(xs), color='black', label='Theoretical Period')

ax.set_title('Implied Period of Simulated (Apogee, Perigee) '
    'Given Period = 1436 minutes', fontsize=TITLE_SIZE)
ax.set_xlabel('Simulation Trial', fontsize=AXES_SIZE, fontweight='bold')
ax.set_ylabel('Period [Minutes] (Kepler\'s Third Law)', fontsize=AXES_SIZE,
    fontweight='bold')
ax.set_xlim([min(xs), max(xs)])
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(loc=3)

#   -- Residual Analysis

KC_residT = KC_period - T
DC_residT = DC_period - T

ax2.set_title('Residuals', fontsize=TITLE_SIZE)
ax2.set_xlabel('Minutes', fontsize=AXES_SIZE, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=AXES_SIZE, fontweight='bold')
ax2.tick_params(labelsize=TICK_SIZE)

ax2.hist(KC_residT, bins=50, color='green', alpha=0.4,
    label='Orbital Model')
ax2.hist(DC_residT, bins=50, color='red', label='Default Model',
    alpha=0.4)
ax2.legend(loc=2)

# Plot for Implied Semi-Major Axis.
KC_sma = 0.5*(KC['Apogee_km']+KC['Perigee_km']) +\
    EARTH_RADIUS
DC_sma = 0.5*(DC['Apogee_km']+DC['Perigee_km']) +\
    EARTH_RADIUS

fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
xs = range(len(KC))

ax.set_title('Implied Semi-Major Axis of Simulated (Apogee, Perigee) '
    'Given Period = 1436 minutes', fontsize=TITLE_SIZE)
ax.set_xlabel('Simulation Trial', fontsize=AXES_SIZE, fontweight='bold')
ax.set_ylabel('Semi-Major Axis [km]', fontsize=AXES_SIZE, fontweight='bold')
ax.set_xlim([min(xs), max(xs)])
ax.tick_params(labelsize=TICK_SIZE)

ax.scatter(xs, KC_sma, color='green', label='Orbital Model Simulation')
ax.scatter(xs, DC_sma, color='red', label='Default Model Simulation')
ax.hlines(compute_a(T), min(xs), max(xs), color='black',
    label='Theoretical Semi-Major Axis')
ax.legend(loc=3)

#   -- Residual Analysis
KC_residA = KC_sma - compute_a(T)
DC_residA = DC_sma - compute_a(T)

ax2.set_title('Residuals', fontsize=TITLE_SIZE)
ax2.set_xlabel('Kilometers', fontsize=AXES_SIZE, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=AXES_SIZE, fontweight='bold')
ax2.tick_params(labelsize=TICK_SIZE)

ax2.hist(KC_residA, bins=50, color='green', alpha=0.4,
    label='Orbital Model')
ax2.hist(DC_residA, bins=50, color='red', label='Default Model',
    alpha=0.4)
ax2.legend(loc=2)


# Overall Residual Plot.
fig, ax = plt.subplots(nrows=1,ncols=2)
ax=ax.ravel()
fig.suptitle('Residuals of Derived Orbital Statistics From Simulated '
    'Apogee, Perigee', fontsize=TITLE_SIZE)
ax[0].set_title('Implied Period', fontsize=AXES_SIZE)
ax[0].set_xlabel('Residual Minutes', fontweight='bold', fontsize=AXES_SIZE)
ax[0].hist(DC_residT, bins=50, color='red', alpha=0.2, label='Default Model')
ax[0].hist(KC_residT, bins=50, color='green', alpha=1., label='Oribtal Model')
ax[0].tick_params(labelsize=TICK_SIZE-2)
ax[0].set_ylim(0, ax[0].get_ylim()[1]+2)
ax[0].legend(loc=2)

ax[1].set_title('Implied Semi-Major Axis', fontsize=AXES_SIZE)
ax[1].set_xlabel('Residual Kilometers', fontweight='bold', fontsize=AXES_SIZE)
ax[1].hist(DC_residA, bins=50, color='red', alpha=0.2, label='Default Model')
ax[1].hist(KC_residA, bins=50, color='green', alpha=1., label='Orbital Model')
ax[1].tick_params(labelsize=TICK_SIZE-2)
ax[1].set_ylim(0, ax[1].get_ylim()[1]+2)
ax[1].legend(loc=2)

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
perigees_ecc0 = apogees
perigees_ecc2 = (apogees + EARTH_RADIUS) * (1-0.2)/(1+0.2) - EARTH_RADIUS
periods_ecc0 = compute_period(apogees, perigees_ecc0)
periods_ecc2 = compute_period(apogees, perigees_ecc2)

ax.plot(apogees, periods_ecc0, color='purple', linestyle='dashed',
    label='Theoretical [ecc=0.0]')
ax.plot(apogees, periods_ecc2, color='purple', linestyle='dashed',
    label='Theoretical [ecc=0.2]')

ax.set_title('Observations and Simulations from Joint Distribution of '
    '(Apogee, Period)', fontsize=TITLE_SIZE)
ax.set_xlabel('Apogee [km]', fontsize=AXES_SIZE, fontweight='bold')
ax.set_ylabel('Period [Minutes]', fontsize=AXES_SIZE, fontweight='bold')
ax.set_xlim([-200, 48000])
ax.set_ylim([-20, 1700])
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(loc=4)
