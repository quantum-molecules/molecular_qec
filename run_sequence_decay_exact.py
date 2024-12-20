from check_operator_dynamics import *
import qutip as q
import matplotlib.pyplot as plt
from implementation_dynamics import *

""""Runs a simulation of fidelity vs decay time for the approximate code
Data is used in plot_sequential.py to generate figure 5.
"""


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    imp_op = ImplementationDynamics(full_basis=False, do_plot=True)
    #     imp_op.full_sequence_ideal(do_refresh=False)
    #     imp_op.characterize_results()

    # elif False:

    no_cycles = 40
    decay_time = 0.05
    do_refresh = True
    do_dephasing = True
    #    for decay_time in [0.1, 0.2, 0.5]:
    for do_dephasing in [True, False]:
        for j_time in [0.002, 0.005, 0.01]:  # [0.001, 0.002, 0.005]:
            time_list = []
            log_op_list = []
            log_op_uncorr_list = []
            imp_op = ImplementationDynamics(full_basis=False)
            final_state_dict = imp_op.basis
            for i in range(no_cycles):
                imp_op = ImplementationDynamics(full_basis=False)
                imp_op.do_dephasing = do_dephasing
                imp_op.full_sequence_ideal(
                    time_m=j_time,
                    time_decay=decay_time,
                    do_refresh=do_refresh,
                    initial_state_dict=final_state_dict,
                )
                final_state_dict = imp_op.get_state_dict(final_state=True)
                time_worst = (
                    12 * do_refresh + 2 * imp_op.time_j_check + 2 * imp_op.time_m_check
                )
                this_time = (i + 1) * (
                    time_worst
                    * imp_op.time_m
                    / imp_op.J_m_time_ratio
                    * (1 - np.exp(-decay_time))
                    + decay_time
                )
                time_list.append(this_time)
                print(f"------- Sequence with refreshing {i + 1} iteration")
                print(f"time: {this_time}")
                log_op, log_op_uncorr = imp_op.characterize_results(
                    effective_time=this_time
                )
                log_op_list.append(log_op)
                log_op_uncorr_list.append(log_op_uncorr)

            if do_refresh:
                marker = "x"
            else:
                marker = "o"

            plt.figure(1, figsize=(10, 7))
            plt.title(f"delta_m timescale: {j_time}")
            plt.plot(
                1 - np.array(log_op_uncorr_list),
                1 - np.array(log_op_list),
                label=f"{j_time} - {do_refresh} - {decay_time}",
            )
            plt.xlabel("1-(<Z>+1)/2 uncorrected")
            plt.ylabel("1-(<Z>+1)/2 corrected")
            plt.legend(loc="best")
            plt.savefig(f"test_img/log_op_refresh{do_refresh}_p.pdf")

            plt.figure(2, figsize=(10, 7))
            plt.plot(
                time_list,
                1 - np.array(log_op_list),
                marker,
                label=f"{j_time} - {do_refresh} - {decay_time}",
            )
            #            plt.plot(time_list, 1 - np.array(log_op_uncorr_list))
            plt.ylabel("1-(<Z>+1)/2")
            plt.xlabel("time")
            plt.legend(loc="best")
            plt.savefig(f"test_img/log_op_times_refresh{do_refresh}_p.pdf")

            plt.figure(3, figsize=(10, 7))
            plt.plot(
                np.arange(no_cycles) + 1,
                1 - np.array(log_op_list),
                marker,
                label=f"{j_time} - {do_refresh} - {decay_time}",
            )
            # plt.plot(np.arange(no_cycles)+1, 1 - np.array(log_op_uncorr_list))
            plt.ylabel("1-(<Z>+1)/2")
            plt.xlabel("cycles")
            plt.legend(loc="best")
            plt.savefig(f"test_img/log_op_cycles_refresh{do_refresh}_p.pdf")
            savefilename = f"sequence_data_0/exact_cycles_ind_{j_time}_{decay_time}_{do_refresh}_{do_dephasing}.txt"
            np.savetxt(savefilename, [time_list, log_op_list, log_op_uncorr_list])

    time_list = []
    log_op_uncorr_list = []
    for this_time in np.linspace(1e-7, 1.7, 100):
        log_op, log_op_uncorr = imp_op.characterize_results(effective_time=this_time)
        time_list.append(this_time)
        log_op_uncorr_list.append(log_op_uncorr)
    plt.figure(2, figsize=(10, 7))
    plt.plot(time_list, 1 - np.array(log_op_uncorr_list), label="uncorrected")
    plt.ylabel("1-(<Z>+1)/2")
    plt.xlabel("time")
    plt.xlim(0, 1.7)
    plt.legend(loc="best")
    plt.savefig(f"test_img/log_op_times_refresh{do_refresh}_p.pdf")
