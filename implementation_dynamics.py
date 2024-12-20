from check_operator_dynamics import *
import qutip as q
import matplotlib.pyplot as plt


class ImplementationDynamics(CheckOperatorDynamics):
    """Simulates implementation dynamics by imperfect check operators using unitary gates on ions.
    Includes black body radiation and dephasing

    """

    def __init__(self, *args, **kwargs):
        super(ImplementationDynamics, self).__init__(*args, **kwargs)
        self.J_m_time_ratio = 1
        self.time_j_check = 4.5
        self.time_m_check = 4
        self.time_refresh = 12
        self.dephasing_m_time_ratio = 1.0
        self.do_dephasing = True

    def apply_dephasing(self, rho: q.Qobj, final_time: float) -> q.Qobj:
        """Applies dephasing to a state for time final_time"""
        probability = 1 - np.exp(-final_time)
        final_rho = (1 - probability) * rho + probability * q.qdiags(
            np.diag(rho.data.toarray()), offsets=0
        )
        return final_rho

    def perform_J_check(self, action_name: str, delta_J: int) -> None:
        """Applies a perfect J checke followed by decay"""
        operator = self.code.get_decay_op(
            self.code.code_manifold + delta_J,
            delta_j=-delta_J,
            delta_m=0,
            equal_coupling=False,
        )
        self.perform_check(
            action_name, operator, apply_method=self.apply_J_check_operator
        )
        J_timescale = self.time_m / self.J_m_time_ratio
        j_time = J_timescale * self.time_j_check
        self.perform_decay(action_name + "decay", j_time)

    def apply_J_check_operator(self, rho: q.Qobj, operator: q.Qobj) -> list:
        """Simulate Hamiltonian dynamics of implementing a J check operator using a Scrofulous pulse sequence."""
        rho_mot = q.tensor(rho, q.ket2dm(q.basis(2, 0)))
        operator_mot = q.tensor(operator, q.sigmam())
        Id_rot = q.qeye(self.code.dim)
        check_qubit_z = q.tensor(Id_rot, q.sigmaz())
        max_val = np.max(operator_mot.data.as_scipy())
        h_x_corr = 1 / max_val * (operator_mot + operator_mot.dag())
        h_y_corr = 1 / max_val * ((1j * operator_mot) + (1j * operator_mot).dag())
        theta_list = [1 / 2, 1 / 2, 1 / 2]
        phi_list = [1 / 3 * np.pi, 5 / 3 * np.pi, 1 / 3 * np.pi]

        U_tot = 1
        for theta, phi in zip(theta_list, phi_list):
            H = np.cos(phi) * h_x_corr + np.sin(phi) * h_y_corr
            U = (1j * np.pi * theta * H).expm()
            U_tot = U_tot * U

        rho_corr = U_tot * rho_mot * U_tot.dag()
        es_list, pr_list, ev_list = get_unique_projectors(
            check_qubit_z, rho_corr, return_projector=False
        )
        result_list = []
        for es, pr, ev in zip(es_list, pr_list, ev_list):
            if np.abs(pr) > 1e-9:
                proj_eig = es.dag() * rho_corr * es / pr
                rho_final = proj_eig.ptrace([0])
                # self.code.rot_hinton(rho_final.ptrace([0]))
                proj_dict = {
                    "eigenstate": es,
                    "probability": pr,
                    "eigenvalue": ev,
                    "state": rho_final,
                }
                result_list.append(proj_dict)
        return result_list

    def perform_J_correction(self, action_name: str):
        action_name = "j_check"
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            outcome_list = self.get_previous_outcomes(node)
            if outcome_list == [1.0, 1.0]:
                node.sequence_done = True
            else:
                self.generate_node(
                    state_name=node.state_name,
                    action_name=action_name,
                    sequence_depth=self.sequence_counter + 1,
                    state=rho,
                    parent=node,
                )

        self.sequence_counter += 1

    def perform_m_check(self, action_name: str, delta_m: int) -> None:
        """Performs check operators subsequently on multiple m states"""
        code = self.code
        m_list = np.array([code.m1, code.m2, -code.m1, -code.m2]) + delta_m
        full_u = 1
        for this_m in m_list:
            op_bsb = code.sigma_j_k(code.code_manifold, this_m, 0, -delta_m)
            h_bsb = q.tensor(op_bsb, q.sigmam())
            h_bsb = h_bsb + h_bsb.dag()
            u_bsb = (1j * np.pi / 2 * h_bsb).expm()
            full_u = full_u * u_bsb
        self.perform_check(
            action_name, full_u, apply_method=self.apply_m_check_operator
        )
        m_time = self.time_m * self.time_m_check
        timescale = 1 / self.dephasing_m_time_ratio
        # self.perform_decay(action_name + 'dephasing', m_time*timescale, apply_method=self.apply_dephasing)
        # if self.do_dephasing:
        self.perform_decay(action_name + "decay", m_time, individual_op=True)

    def apply_m_check_operator(self, rho: q.Qobj, operator: q.Qobj) -> list:
        """Simulate Hamiltonian dynamics of implementing a single m state."""
        rho_mot = q.tensor(rho, q.ket2dm(q.basis(2, 0)))
        Id_rot = q.qeye(self.code.dim)
        check_qubit_z = q.tensor(Id_rot, q.sigmaz())
        U_tot = operator

        rho_corr = U_tot * rho_mot * U_tot.dag()
        es_list, pr_list, ev_list = get_unique_projectors(
            check_qubit_z, rho_corr, return_projector=False
        )
        result_list = []
        for es, pr, ev in zip(es_list, pr_list, ev_list):
            if np.abs(pr) > 1e-9:
                proj_eig = es.dag() * rho_corr * es / pr
                rho_final = proj_eig.ptrace([0])
                # self.code.rot_hinton(rho_final.ptrace([0]))
                proj_dict = {
                    "eigenstate": es,
                    "probability": pr,
                    "eigenvalue": ev,
                    "state": rho_final,
                }
                result_list.append(proj_dict)
        return result_list


class ImplementationApprox(ImplementationDynamics):
    """Simulates QEC sequences for approximate codes"""

    def __init__(self, *args, **kwargs):
        super(ImplementationApprox, self).__init__(*args, **kwargs)
        self.J_m_time_ratio = 100
        self.time_j_check = 4.5
        self.time_m_check = 2
        self.time_refresh = 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    imp_op = ImplementationDynamics(
        full_basis=False, do_plot=True, apply_bbr_decay=True
    )
    imp_op.full_sequence_ideal(do_refresh=True, do_final_decay=True, time_decay=0.5)
    imp_op.characterize_results()
    imp_op.characterize_single_cycle()
    no_cycles = 10
    decay_time = 0.1
    do_refresh = True
    for decay_time in [0.1, 0.2]:
        #    for do_refresh in [True, False]:
        for j_time in [0.02, 0.05]:
            time_list = []
            log_op_list = []
            log_op_uncorr_list = []
            imp_op = ImplementationDynamics(full_basis=False)
            final_state_dict = imp_op.basis
            for i in range(no_cycles):
                imp_op = ImplementationDynamics(full_basis=False)
                imp_op.full_sequence_ideal(
                    time_m=j_time,
                    time_decay=decay_time,
                    do_refresh=do_refresh,
                    initial_state_dict=final_state_dict,
                )
                final_state_dict = imp_op.get_state_dict(final_state=True)
                this_time = (i + 1) * (
                    (12 * do_refresh + 8) * imp_op.time_m / imp_op.J_m_time_ratio
                    + decay_time
                )
                time_list.append(this_time)
                print(f"------- Sequence with refreshing {i + 1} iteration")
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

    time_list = []
    log_op_uncorr_list = []
    for this_time in np.linspace(1e-7, 0.5, 100):
        log_op, log_op_uncorr = imp_op.characterize_results(effective_time=this_time)
        time_list.append(this_time)
        log_op_uncorr_list.append(log_op_uncorr)
    plt.figure(2, figsize=(10, 7))
    plt.plot(time_list, 1 - np.array(log_op_uncorr_list), label="uncorrected")
    plt.ylabel("1-(<Z>+1)/2")
    plt.xlabel("time")
    plt.xlim(0, 1)
    plt.legend(loc="best")
    plt.savefig(f"test_img/log_op_times_refresh{do_refresh}_p.pdf")
