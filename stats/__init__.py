"""Pure statistics functions for the Experimentation Lab.

Everything here is UI-free and fully unit-tested so the Streamlit layer can stay
a thin presentation shell. Import surface:

    from stats.power import required_sample_size, power_for_sample_size, sensitivity_curve
    from stats.tests import two_proportion_ztest, ABResult
    from stats.cuped import cuped_adjust
"""
