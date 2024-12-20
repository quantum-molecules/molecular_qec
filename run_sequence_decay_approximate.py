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
    imp_op = ImplementationApprox(full_basis=False, do_plot=True)
    #     imp_op.full_sequence_ideal(do_refresh=False)
    #     imp_op.characterize_results()

    # elif False:
    do_ideal = False
    do_dephasing = False
    # code_man = 11
    no_cycles = 40
    decay_time = 0.05
    do_refresh = False
    for code_man in [7, 9, 11, 13]:
        # for decay_time in [0.05]:
        #    for do_refresh in [True, False]:
        for j_time in [0.002, 0.005, 0.01]:
            time_list = []
            log_op_list = []
            log_op_uncorr_list = []
            # imp_op = ImplementationDynamics(full_basis=False)
            if do_ideal:
                imp_op = CheckOperatorDynamics(
                    full_basis=False,
                    do_plot=True,
                    code_manifold=code_man,
                    apply_bbr_decay=True,
                    approx_code=True,
                )
            else:
                imp_op = ImplementationApprox(
                    full_basis=False,
                    do_plot=True,
                    code_manifold=code_man,
                    apply_bbr_decay=True,
                    approx_code=True,
                )
            imp_op.do_dephasing = do_dephasing

            initial_state = imp_op.code.psi_0
            initial_state = q.ket2dm(initial_state)
            init_dict = {"0": initial_state}

            final_state_dict = init_dict
            for i in range(no_cycles):
                if do_ideal:
                    imp_op = CheckOperatorDynamics(
                        full_basis=False,
                        do_plot=False,
                        code_manifold=code_man,
                        apply_bbr_decay=True,
                    )
                else:
                    imp_op = ImplementationDynamics(
                        full_basis=False,
                        do_plot=False,
                        code_manifold=code_man,
                        apply_bbr_decay=True,
                    )
                imp_op.basis = init_dict
                # imp_op.full_sequence_ideal(do_refresh=False, initial_state_dict=final_state_dict)

                # imp_op = ImplementationDynamics(full_basis=False)
                imp_op.do_dephasing = do_dephasing
                imp_op.full_sequence_ideal(
                    time_m=j_time,
                    time_decay=decay_time,
                    do_refresh=do_refresh,
                    initial_state_dict=final_state_dict,
                )
                final_state_dict = imp_op.get_state_dict(final_state=True)
                time_worst = (
                    12 * do_refresh + 2 * imp_op.time_j_check + 1 * imp_op.time_m_check
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
            plt.savefig(f"test_img/log_op_approx_refresh{do_refresh}_p.pdf")

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
            plt.savefig(f"test_img/log_op_approx_times_refresh{do_refresh}_p.pdf")

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
            savefilename = f"sequence_data_0/approx_cycles_ind_{j_time}_{decay_time}_{code_man}_{do_dephasing}.txt"
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
    plt.savefig(f"test_img/log_op_approx_times_refresh{do_refresh}_p.pdf")
