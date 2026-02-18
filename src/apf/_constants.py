"""Centralized observational constants and experimental data.

All observed values used by the validation module live here.
Updating for a new data release (e.g., Planck 2024) requires
changing ONLY this file.

Sources:
  Planck 2018: arXiv:1807.06209 (Table 2, TT,TE,EE+lowE+lensing)
  PDG 2024:    pdg.lbl.gov
  Fields 2020: arXiv:1912.01132 (BBN review)
  LIGO 2021:   arXiv:2112.06861 (graviton mass bound)

APF v5.0
"""

__all__ = ['OBS', 'PLANCK', 'PDG', 'BBN', 'PHYSICAL']


# ── Physical constants ────────────────────────────────────────────────

PHYSICAL = {
    'M_Pl_GeV':        1.22e19,       # Planck mass (GeV)
    'M_Pl_red_GeV':    2.435e18,      # Reduced Planck mass (GeV)
    'G_N':             6.674e-11,     # Newton's constant (m^3 kg^-1 s^-2)
    'hbar_eV_s':       6.582e-16,     # ℏ (eV·s)
    'k_B_eV_K':        8.617e-5,      # k_B (eV/K)
    'l_Pl_m':          1.616e-35,     # Planck length (m)
    'T_CMB_K':         2.7255,        # CMB temperature (K)
}


# ── Planck 2018 cosmological parameters ──────────────────────────────

PLANCK = {
    # Density fractions (Planck 2018 TT,TE,EE+lowE+lensing)
    'Omega_Lambda':     (0.6889, 0.0056),    # (value, 1σ error)
    'Omega_m':          (0.3111, 0.0056),
    'Omega_b':          (0.0490, 0.0003),
    'Omega_DM':         (0.2607, 0.0050),
    'f_b':              (0.1571, 0.0010),     # Ω_b / Ω_m

    # Spectral index and tensor-to-scalar ratio
    'n_s':              (0.9649, 0.0042),
    'r_upper':          0.036,                # 95% CL upper bound

    # Hubble constant
    'H0_km_s_Mpc':     (67.36, 0.54),
}


# ── PDG 2024 particle physics ────────────────────────────────────────

PDG = {
    # Quark masses (GeV, MS-bar at 2 GeV unless noted)
    'm_u_GeV':          (0.00216, 0.00049),
    'm_d_GeV':          (0.00467, 0.00048),
    'm_s_GeV':          (0.0934,  0.0084),
    'm_c_GeV':          (1.27,    0.02),      # MS-bar at m_c
    'm_b_GeV':          (4.18,    0.03),      # MS-bar at m_b
    'm_t_GeV':          (172.69,  0.30),      # pole mass

    # Lepton masses (GeV)
    'm_e_GeV':          5.110e-4,
    'm_mu_GeV':         0.10566,
    'm_tau_GeV':        1.7768,

    # Gauge boson masses (GeV)
    'm_W_GeV':          (80.377, 0.012),
    'm_Z_GeV':          (91.1876, 0.0021),
    'm_H_GeV':          (125.25, 0.17),

    # CKM parameters (Wolfenstein)
    'lambda_CKM':       (0.22650, 0.00048),
    'A_CKM':            (0.790, 0.017),
    'rho_bar_CKM':      (0.141, 0.017),
    'eta_bar_CKM':      (0.357, 0.011),
    'delta_CKM_rad':    (1.144, 0.027),       # ~ 65.6°

    # Mixing angles (sin^2 theta_W)
    'sin2_theta_W':     (0.23122, 0.00004),

    # Coupling constants
    'alpha_em_inv':     137.036,               # at q^2 = 0
    'alpha_s_MZ':       (0.1180, 0.0009),      # at M_Z

    # Graviton mass bound
    'm_graviton_eV':    1.76e-23,              # LIGO 95% CL upper

    # Baryon asymmetry
    'eta_B':            (6.12e-10, 0.04e-10),  # baryon-to-photon ratio

    # Neutron EDM (theta_QCD bound)
    'theta_QCD_bound':  1e-10,                 # |theta| < 10^{-10}
}


# ── BBN observational abundances (Fields 2020) ───────────────────────

BBN = {
    # Primordial helium-4 mass fraction
    'Y_p':              (0.2449, 0.0040),

    # Deuterium abundance (D/H by number)
    'D_over_H':         (2.547e-5, 0.025e-5),

    # Helium-3 abundance (^3He/H by number)
    'He3_over_H':       (1.1e-5, 0.2e-5),

    # Lithium-7 abundance (^7Li/H by number)
    # NOTE: "cosmological lithium problem" — BBN predicts ~3x observed
    'Li7_over_H':       (1.6e-10, 0.3e-10),

    # Effective number of neutrino species
    'N_eff':            (2.99, 0.17),          # Planck 2018
    'N_eff_SM':         3.044,                 # SM prediction
}


# ── Derived observational quantities ─────────────────────────────────

OBS = {
    'log10_Lambda_G':   -121.44,               # log10(Λ·G) observed
    'S_dS_nats':        282.12,                # de Sitter entropy (nats)
    'g_star_full_SM':   106.75,                # relativistic DOF (all SM)
}
