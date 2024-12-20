from rot_qec_counter import *
from check_operator_dynamics import *
import qutip as q
import matplotlib.pyplot as plt
from implementation_dynamics import *
from check_operator_dynamics import *

"""Runs a simulation of the approximate code with different J0 values
This file generates figure 3.

"""

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    do_ideal = True
    max_list = []
    J0_list = list(range(5, 15))
    for idx, code_man in enumerate(J0_list):
        if do_ideal:
            imp_op = CheckOperatorDynamics(
                full_basis=True,
                do_plot=True,
                code_manifold=code_man,
                apply_bbr_decay=False,
                approx_code=True,
            )
        else:
            imp_op = ImplementationDynamics(
                full_basis=True,
                do_plot=True,
                code_manifold=code_man,
                apply_bbr_decay=False,
                approx_code=True,
            )
        initial_state = imp_op.code.psi_p
        initial_state = q.ket2dm(initial_state)
        init_dict = {"+": initial_state}

        if do_ideal:
            imp_op = CheckOperatorDynamics(
                full_basis=True,
                do_plot=False,
                code_manifold=code_man,
                apply_bbr_decay=False,
                approx_code=True,
            )
        else:
            imp_op = ImplementationDynamics(
                full_basis=True,
                do_plot=False,
                code_manifold=code_man,
                apply_bbr_decay=False,
                approx_code=True,
            )
        imp_op.basis = init_dict

        imp_op.full_sequence_ideal(do_refresh=False, initial_state_dict=init_dict)
        imp_op.characterize_results()
        imp_op.characterize_single_cycle()
        max_inf = imp_op.plot_single_cycle(
            filename=f"test_img/approximate_J_ideal_{do_ideal}.pdf",
            no_datasets=len(J0_list),
            idx_data=idx,
            plot_prob=False,
        )
        max_list.append(max_inf)
    # char_dict = imp_op.characterize_single_cycle()

    plt.figure()
    # plt.plot(J0_list,max_list)
    plt.bar(J0_list, max_list)
    plt.xlabel("J")
    plt.ylabel("logical infidelity")
    plt.tight_layout()
    plt.savefig("test_img/approximate_vs_j0.pdf")

    cm = 1 / 2.54  # centimeters in inches

    # revtext template textwidths for figures
    figwidth_1col = 8.6
    figwidth_2col = 17.8

    np.savetxt("approx_vs_jo.txt", [J0_list, max_list])

    fig1, axs = plt.subplots(1, 1, figsize=(figwidth_1col * cm, 8.75 / 1.4 * cm))
    font = {"family": "Arial", "weight": "normal", "size": 9}
    plt.rc("font", **font)
    COLOR = "k"
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["axes.labelcolor"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR
    plt.rc("axes", edgecolor=COLOR)
    # ylim_fid = [0.5, 1]
    # axs.set_xlim(xlim_fid)
    # axs.set_ylim(ylim_fid)
    axs.bar(
        J0_list,
        max_list,
        color="#FF983C",
        label="full DEC $\hat{\\rho}_0$",
    )
    axs.set_xlabel("$J_{\\text{C}}$")
    axs.set_ylabel("Logical infidelity $1-\\mathcal{F}_+$")

    # FINALIZE FIGURE AND SAVE
    fig1.patch.set_alpha(1)
    fig1.tight_layout()
    # fig1.subplots_adjust(bottom=0.25)  # create space for legend
    # fig1.subplots_adjust(bottom=0.40)  # create space for legend
    fig1.savefig(
        "images/paper/approximate_vs_j0.pdf", facecolor=fig1.get_facecolor(), dpi=300
    )
    plt.show()
