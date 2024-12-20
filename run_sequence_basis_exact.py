from rot_qec_counter import *
from check_operator_dynamics import *
import qutip as q
import matplotlib.pyplot as plt
from implementation_dynamics import *
from check_operator_dynamics import *


"""Runs a simulation of the exact code with different operation times
This file is not used in the manuscript.
"""

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    do_ideal = False
    time_m_list = [0.001, 0.005, 0.01]
    for idx, time_m in enumerate(time_m_list):
        if do_ideal:
            imp_op = CheckOperatorDynamics(
                full_basis=False, do_plot=False, apply_bbr_decay=True
            )
        else:
            imp_op = ImplementationDynamics(
                full_basis=False, do_plot=False, apply_bbr_decay=True
            )

        imp_op.full_sequence_ideal(do_refresh=True, time_decay=0.1, time_m=time_m)
        imp_op.characterize_results()
        imp_op.plot_single_cycle(
            filename=f"test_img/log_outcome_bbr.pdf",
            fignum=101,
            no_datasets=len(time_m_list),
            idx_data=idx,
            plot_prob=True,
            min_prob=1e-3,
        )
    # char_dict = imp_op.characterize_single_cycle()

    do_ideal = True
    time_decay_list = [0.01, 0.1, 0.1]
    for idx, time_decay in enumerate(time_decay_list):
        if do_ideal:
            imp_op = CheckOperatorDynamics(
                full_basis=False, do_plot=False, apply_bbr_decay=True
            )
        else:
            imp_op = ImplementationDynamics(
                full_basis=False, do_plot=False, apply_bbr_decay=True
            )

        imp_op.full_sequence_ideal(
            do_refresh=True, time_decay=time_decay, time_m=time_m
        )
        imp_op.characterize_results()
        imp_op.plot_single_cycle(
            filename=f"test_img/log_outcome_ideal_bbr.pdf",
            fignum=102,
            no_datasets=len(time_m_list),
            idx_data=idx,
            plot_prob=True,
            min_prob=1e-3,
        )
