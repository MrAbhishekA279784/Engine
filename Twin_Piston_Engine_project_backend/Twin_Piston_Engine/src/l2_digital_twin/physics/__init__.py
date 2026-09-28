"""
L2 Digital Twin — Physics fix modules from the physics audit.

Defects (D), bugs (B) and missing derivations (M) identified in the audit.
Each module keeps its standalone self-test under ``if __name__ == "__main__":``.

    D03_lambda_derivation              D04_air_mass_flow_speed_density
    B01_vibration_ring_buffer          M01_excess_kurtosis
    M02_half_order_fraction            M03_firing_order_fraction
    M04_envelope_demodulation          (M05_nyquist_guard: moved to src/core/sensor_physics)
    M06_dual_channel_misfire_gate      M07_specific_fuel_consumption
    M08_normalised_engine_load         M09_combustion_stability_index
    M10_lubrication_vibration_indices  M11_residual_expectation_channels
"""
