"""
Physical Bounds and Link-Budget Simulation for Microscopic RF Nodes (Smart Dust)
Models the Chu-Harrington limit, Friis transmission loss, and ambient harvesting limits.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Physical constants
C = 299792458.0             # Speed of light (m/s)
K_B = 1.380649e-23          # Boltzmann constant (J/K)
T_KELVIN = 290.0            # Ambient temperature (K)
ETA_0 = 376.73              # Free-space impedance (Ohms)
SIGMA_CU = 5.8e7            # Conductivity of copper (S/m)

def chu_harrington_minimum_q(ka):
    """
    Lower bound on quality factor Q for an electrically small antenna enclosed in a sphere of radius a.
    ka = wave number * radius = 2*pi*a / lambda
    """
    return (1.0 / (ka**3)) + (1.0 / ka)

def antenna_efficiency(radius_meters, freq_hz, thickness=1e-5):
    """
    Estimates radiation efficiency eta = R_rad / (R_rad + R_loss) for an electrically small dipole.
    """
    wavelength = C / freq_hz
    k = 2 * np.pi / wavelength
    ka = k * radius_meters
    
    # Radiation resistance for small dipole of length 2a
    r_rad = 20.0 * (np.pi**2) * ((2 * radius_meters / wavelength)**2)
    
    # Skin depth in copper
    skin_depth = np.sqrt(1.0 / (np.pi * freq_hz * 4 * np.pi * 1e-7 * SIGMA_CU))
    # Conductor loss resistance
    r_loss = (2 * radius_meters) / (2 * np.pi * radius_meters * SIGMA_CU * np.minimum(thickness, skin_depth))
    
    eta = r_rad / (r_rad + r_loss)
    return np.clip(eta, 1e-12, 1.0), ka

def friis_received_power_dbm(p_tx_dbm, g_tx_dbi, g_rx_dbi, freq_hz, distance_meters):
    """
    Calculates received power using Friis transmission equation.
    """
    wavelength = C / freq_hz
    path_loss_linear = (wavelength / (4 * np.pi * distance_meters))**2
    path_loss_db = 10 * np.log10(path_loss_linear)
    p_rx_dbm = p_tx_dbm + g_tx_dbi + g_rx_dbi + path_loss_db
    return p_rx_dbm

def thermal_noise_floor_dbm(bandwidth_hz, noise_figure_db=5.0):
    """
    Thermal noise power P = k_B * T * B + NF.
    """
    noise_power_watts = K_B * T_KELVIN * bandwidth_hz
    noise_power_dbm = 10 * np.log10(noise_power_watts * 1e3) + noise_figure_db
    return noise_power_dbm

def run_simulation(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Range of node radii: 50 microns to 5 millimeters
    radii = np.logspace(-5, -2.3, 100) # 10 um to 5 mm
    frequencies = [915e6, 2.45e9, 24e9, 60e9] # 915 MHz, 2.45 GHz, 24 GHz, 60 GHz
    labels = ["915 MHz (ISM)", "2.45 GHz (WiFi/BLE)", "24 GHz (K-Band)", "60 GHz (mmWave)"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Plot 1: Antenna Efficiency vs Radius
    for freq, lbl in zip(frequencies, labels):
        effs = []
        for r in radii:
            eff, _ = antenna_efficiency(r, freq)
            effs.append(10 * np.log10(eff))
        ax1.plot(radii * 1e3, effs, label=lbl, linewidth=2)
        
    ax1.axvline(x=0.1, color='red', linestyle='--', label='100 µm ("Smart Dust" scale)')
    ax1.axvline(x=1.0, color='gray', linestyle=':', label='1 mm scale')
    ax1.set_xscale('log')
    ax1.set_xlabel('Node Radius $a$ (millimeters)', fontsize=11)
    ax1.set_ylabel('Radiation Efficiency $\\eta$ (dB)', fontsize=11)
    ax1.set_title('Antenna Radiation Efficiency vs. Physical Scale', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(-120, 5)
    
    # Plot 2: Received Power vs Distance for a 100 µm node emitting 1 µW (optimistic ambient harvest)
    distances = np.logspace(-1, 3, 100) # 10 cm to 1 kilometer (1000 meters)
    node_radius = 100e-6 # 100 microns
    p_harvested_watts = 1e-6 # 1 microwatt harvested
    p_tx_dbm = 10 * np.log10(p_harvested_watts * 1e3) # -30 dBm
    
    rx_sensitivity_dbm = thermal_noise_floor_dbm(10e3, noise_figure_db=4.0) + 10.0 # 10 kHz BW, 10 dB SNR
    thermal_floor_dbm = thermal_noise_floor_dbm(10e3, noise_figure_db=0.0) # Raw k_B * T * B
    
    for freq, lbl in zip(frequencies, labels):
        eff, _ = antenna_efficiency(node_radius, freq)
        g_tx_dbi = 1.76 + 10 * np.log10(eff) # isotropic dipole directivity ceiling (1.76 dBi) + efficiency
        g_rx_dbi = 6.0 # standard directional base station antenna
        
        prx = [friis_received_power_dbm(p_tx_dbm, g_tx_dbi, g_rx_dbi, freq, d) for d in distances]
        ax2.plot(distances, prx, label=lbl, linewidth=2)
        
    ax2.axhline(y=thermal_floor_dbm, color='red', linestyle='--', linewidth=1.5,
                label=f'Thermal Noise Floor $k_B T B$ ({thermal_floor_dbm:.1f} dBm)')
    ax2.axhline(y=rx_sensitivity_dbm, color='black', linestyle=':', linewidth=1.5,
                label=f'Detection Threshold (+10 dB SNR, {rx_sensitivity_dbm:.1f} dBm)')
    ax2.axvline(x=10.0, color='gray', linestyle=':', alpha=0.7, label='10 m (Room Scale)')
    ax2.axvline(x=1000.0, color='purple', linestyle=':', alpha=0.7, label='1 km (Macro/Cell Scale)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Distance to Base Station $d$ (meters)', fontsize=11)
    ax2.set_ylabel('Received Power $P_r$ (dBm)', fontsize=11)
    ax2.set_title('Received Signal from 100 µm Node vs. Distance (up to 1 km)', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', fontsize=8.5)
    ax2.set_ylim(-240, -20)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "antenna_efficiency_and_link_budget.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    run_simulation(figures_dir)
