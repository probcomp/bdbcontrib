import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn

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

matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['font.weight'] = 'bold'

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
ax.plot(implied_perigee, implied_apogee, color='purple')

ax.set_title('Simulated and Observed (Apogee, Perigee) Given Period = 1436',
    fontsize=16)
ax.set_xlabel('Apogee [km]', fontsize=16)
ax.set_ylabel('Perigee [km]', fontsize=16)
ax.set_xlim([23000, 52000])
ax.set_ylim([23000, 42000])
ax.legend()

# Plot for Implied Period.
fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
xs = range(len(KC))

ax.scatter(xs, KC_period, color='green', label='Orbital Model')
ax.scatter(xs, DC_period, color='red', label='Default Model')
ax.hlines(T, min(xs), max(xs), color='black', label='Theoretical Period')

ax.set_title('Implied Period of Simulated (Apogee, Perigee) '
    'Given Period = 1436 minutes', fontsize=16)
ax.set_xlabel('Simulation Trial', fontsize=16)
ax.set_ylabel('Period [Minutes] (Kepler\'s Third Law)', fontsize=16)
ax.set_xlim([min(xs), max(xs)])
ax.legend()

#   -- Residual Analysis

KC_resid = KC_period - T
DC_resid = DC_period - T

ax2.set_title('Residuals', fontsize=16)
ax2.set_xlabel('Minutes', fontsize=16)
ax2.set_ylabel('Frequency', fontsize=16)

ax2.hist(KC_resid, bins=50, color='green', alpha=0.4,
    label='Orbital Model')
ax2.hist(DC_resid, bins=50, color='red', label='ODefault Model',
    alpha=0.4)
ax2.legend()

# Plot for Implied Semi-Major Axis.
KC_sma = 0.5*(KC['Apogee_km']+KC['Perigee_km']) +\
    EARTH_RADIUS
DC_sma = 0.5*(DC['Apogee_km']+DC['Perigee_km']) +\
    EARTH_RADIUS

fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
xs = range(len(KC))

ax.set_title('Implied Semi-Major Axis of Simulated (Apogee, Perigee) '
    'Given Period = 1436 minutes', fontsize=16)
ax.set_xlabel('Simulation Trial', fontsize=16)
ax.set_ylabel('Semi-Major Axis [km]', fontsize=16)
ax.set_xlim([min(xs), max(xs)])

ax.scatter(xs, KC_sma, color='green', label='Orbital Model Simulation')
ax.scatter(xs, DC_sma, color='red', label='Default Model Simulation')
ax.hlines(compute_a(T), min(xs), max(xs), color='black',
    label='Theoretical Semi-Major Axis')
ax.legend()

#   -- Residual Analysis
KC_resid = KC_sma - compute_a(T)
DC_resid = DC_sma - compute_a(T)

ax2.set_title('Residuals', fontsize=16)
ax2.set_xlabel('Kilometers', fontsize=16)
ax2.set_ylabel('Frequency', fontsize=16)

ax2.hist(KC_resid, bins=50, color='green', alpha=0.4,
    label='Orbital Model')
ax2.hist(DC_resid, bins=50, color='red', label='ODefault Model',
    alpha=0.4)
ax2.legend()

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
    label='Theoretical [at ecc=0.0]')
ax.plot(apogees, periods_ecc2, color='purple', linestyle='dashed',
    label='Theoretical [at ecc=0.2]')

ax.set_title('Observations and Simulations from Joint Distribution of '
    '(Apogee, Period)', fontsize=16)
ax.set_xlabel('Apogee [km]', fontsize=16)
ax.set_ylabel('Period [Minutes]', fontsize=16)
ax.set_xlim([-200, 48000])
ax.set_ylim([-20, 1700])
ax.legend(loc='lower right')
