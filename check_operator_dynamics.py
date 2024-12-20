import matplotlib.pyplot as plt
import numpy as np

# import qtpy
from bigtree import Node, print_tree, find_attrs

from rot_qec_counter import *


class CheckOperatorDynamics:
    """Class to create and run sequential QEC with AE codes by implementing ideal operators

    code_manifold: J value of code manifold
    buffer: Buffer that sets size of Hilbert space  Jmax=J0+buffer
    cs_m1, cs_m2: m values for codewords
    debug: If True, debugging on the terminal via print is enabled
    full_basis: If True, the full basis set is simulated
    file_prefix: Path, where intermediate figures are stored.
    do_plot: If True, intermediate Hinton plots will be generated
    apply_bbr_decay: If True: black body decay is simulated between applying the ideal operators.
    approx_code: If True, an approximate code is used
    individual_op: If True, it is assumed that all emission and absorption events are resolvable
    """

    def __init__(
        self,
        code_manifold: int = 7,
        buffer: int = 2,
        cs_m1: int = 2,
        cs_m2: int = 5,
        debug: bool = True,
        full_basis: bool = False,
        file_prefix: str = "test_img/",
        do_plot: bool = False,
        apply_bbr_decay: bool = True,
        approx_code: bool = True,
        individual_op: bool = False,
    ):
        self.indvidual_op = individual_op
        self.approx_code = approx_code
        self.apply_bbr_decay = apply_bbr_decay
        self.do_plot = do_plot
        self.file_prefix = file_prefix
        max_j = code_manifold + buffer
        self.code = CounterSymmetricCode(
            max_j, code_manifold, cs_m1, cs_m2, approx_code=approx_code
        )
        self.get_bbr_decay_op()
        self.get_bbr_decay_op_individual_op()
        self.refresh_amplitude0 = [
            [[1.0, 1.0, 1.0, 1.0], 0.005050505050505103],
            [[1.0, 1.0, -1.0, 1.0], 0.0252525252525253],
            [[-1.0, 1.0, -1.0, 1.0], 0.11616161616161616],
            [[1.0, -1.0, -1.0, 1.0], -0.11616161616161616],
            [[1.0, 1.0, 1.0, -1.0], 0.0454545454545455],
            [[-1.0, 1.0, 1.0, -1.0], -0.1464646464646464],
            [[1.0, -1.0, 1.0, -1.0], 0.1464646464646465],
        ]

        self.refresh_amplitude1 = [
            [[1.0, 1.0, 1.0, 1.0], 0.005050505050505103],
            [[1.0, 1.0, -1.0, 1.0], -0.02525252525252523],
            [[-1.0, 1.0, -1.0, 1.0], 0.11616161616161616],
            [[1.0, -1.0, -1.0, 1.0], -0.11616161616161616],
            [[1.0, 1.0, 1.0, -1.0], -0.04545454545454543],
            [[-1.0, 1.0, 1.0, -1.0], -0.1464646464646464],
            [[1.0, -1.0, 1.0, -1.0], 0.1464646464646465],
        ]

        self.debug = debug

        if full_basis:
            self.basis = {
                "0": q.ket2dm(self.code.psi_0),
                "1": q.ket2dm(self.code.psi_1),
                "+": q.ket2dm(self.code.psi_p),
                "-": q.ket2dm(self.code.psi_m),
            }
        else:
            self.basis = {"0": q.ket2dm(self.code.psi_0)}
        self.initial_state_dict = self.basis
        self.set_logical_op_dict()

    def set_logical_op_dict(self) -> None:
        """Generate dictionary for logical operators"""
        op_dict = {
            "0": -self.code.logical_z_operator(),
            "1": self.code.logical_z_operator(),
            "+": self.code.logical_x_operator(),
            "-": self.code.logical_x_operator(),
        }
        self.logical_op_dict = {}
        for state_name in self.basis.keys():
            self.logical_op_dict[state_name] = op_dict[state_name]

    def debug_print(self, my_str: str) -> None:
        if self.debug:
            print(my_str)

    def get_decay_op_dict(self) -> dict:
        """Generate decay operators. Only delta_J=-1"""
        return {
            "pi": self.code.decay_pi_op,
            "sp": self.code.decay_sp_op,
            "sm": self.code.decay_sm_op,
            "pi_p": self.code.decay_pi_op_p,
            "sp_p": self.code.decay_sp_op_p,
            "sm_p": self.code.decay_sm_op_p,
        }

    def get_bbr_decay_op(self) -> None:
        """Creates absorption and emission operators when m states are not resolved"""
        bbr_rate = 1
        code = self.code
        bbr_operator_family = (
            [
                code.get_decay_operator(j, -1, bbr_rate, hc=True, individual_decay=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
            + [
                code.get_decay_operator(j, 0, bbr_rate, hc=True, individual_decay=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
            + [
                code.get_decay_operator(j, 1, bbr_rate, hc=True, individual_decay=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
        )
        self.bbr_op_list = bbr_operator_family

    def get_bbr_decay_op_individual_op(self) -> None:
        """Emission and absorption operators when m states are resolved"""
        bbr_rate = 1
        code = self.code
        bbr_operator_family = (
            [
                code.get_decay_operator(j, -1, bbr_rate, hc=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
            + [
                code.get_decay_operator(j, 0, bbr_rate, hc=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
            + [
                code.get_decay_operator(j, 1, bbr_rate, hc=True)
                for j in range(code.min_l, code.max_l, 1)
            ]
        )
        self.bbr_op_list = bbr_operator_family

        self.bbr_op_list_individual_op = bbr_operator_family

    def apply_bbr(
        self, rho: q.Qobj, final_time: float, individual_op: bool = False
    ) -> q.Qobj:
        """Simulate master equation of black body radiation with free evolution

        rho: initial density matrix
        final_time: Free evolution time
        individual_op: If True, m states are spectroscopically resolvable

        returns final density matrix after free evolution
        """
        if individual_op:
            op_list = self.bbr_op_list_individual_op
        else:
            op_list = self.bbr_op_list
        H = q.identity(self.code.dim)
        num_times = 1000
        times = [0, final_time]
        options = q.solver.Options()
        options["store_final_state"] = True
        options["store_states"] = False
        options["nsteps"] = 1e5
        results = q.mesolve(H * 0, rho, times, op_list, [], options=options)
        return results.final_state

    def apply_operator(self, rho: q.Qobj, operator: q.Qobj) -> q.Qobj:
        """Applies an operator or a ist of operators to a density matrix

        rho: Density matrix to apply the opertator on
        operator: Operator or list of operators"""
        if type(operator) is list:
            rho_op = rho
            for this_op in operator:
                rho_op = this_op * rho_op * this_op.dag()
            return rho_op
        else:
            rho_op = operator * rho * operator.dag()
            return rho_op

    def apply_check_operator(self, rho: q.Qobj, operator: q.Qobj) -> list:
        """Simulates an ideal implementation of a check operator

        rho: Initial density_matrix
        operator: Check operator

        returns a list of dictionaries with keys:
        eigenstate: Eigenstate for outcome
        proability: Probability for outcome
        eigenvalue: Eigenvalue for outcome
        state: Projected state for input density matrix for outcome (not normalized)
        """
        projected_rho_list = []
        decay_op_list = [q.identity(self.code.dim)]

        es_list, pr_list, ev_list = get_unique_projectors(
            operator, rho, return_projector=False
        )
        result_list = []
        for es, pr, ev in zip(es_list, pr_list, ev_list):
            if pr > 1e-9:
                proj_eig = es.dag() * rho * es / pr
                proj_dict = {
                    "eigenstate": es,
                    "probability": pr,
                    "eigenvalue": ev,
                    "state": proj_eig,
                }
                result_list.append(proj_dict)
        return result_list

    def generate_node(
        self,
        state_name: str,
        action_name: str,
        state: q.Qobj,
        sequence_depth: int,
        parent: Node,
        probability: float = 1.0,
        outcome: bool = None,
        sequence_done: bool = False,
    ) -> None:
        """Generates node in self.root_list for saving state and outcomes at each step.
        The state is saved in self.sequence_data. In self.root_list, only the index in self.sequence_data is stored

        state_name: string identifier for the state, either 0,1,+,-
        action_name: name of the check operation
        state: Density matrix
        sequence_dpeth: Momentary sequence depth
        parent: Node of previous check
        probability: Probability for this outcome
        outcome: Outcome of this node
        sequence_done: If True, check sequence is done
        """
        i = len(self.sequence_data)
        a = Node(
            name=state_name + action_name,
            action=action_name,
            state_name=state_name,
            sequence_depth=sequence_depth,
            probability=probability,
            outcome=outcome,
            data_index=i,
            parent=parent,
            sequence_done=sequence_done,
        )

        self.root_list.append(a)
        self.sequence_data.append(state)

    def get_nodes_sequence_depth(self, sequence_depth: int) -> tuple:
        """Retrieves nodes until sequence depth"""
        node_list = find_attrs(
            self.root_node, attr_name="sequence_depth", attr_value=sequence_depth
        )
        return node_list

    def get_nodes_final(self) -> tuple:
        """Retrun final modes of all paths through the tree"""
        node_list = find_attrs(
            self.root_node, attr_name="sequence_done", attr_value=True
        )
        return node_list

    def get_previous_outcomes(self, node: Node) -> list:
        """Returns list of previous outcomes for a given node in the tree"""
        if node.outcome is not None:
            outcome_list = [node.outcome]
        else:
            outcome_list = []
        for this_node in node.ancestors:
            if this_node.outcome is not None:
                outcome_list.append(this_node.outcome)
        return outcome_list

    def get_previous_probabilitites(self, node: Node) -> list:
        """Returns list of previous probabilities for a given node in the tree"""
        if node.probability is not None:
            prob_list = [node.probability]
        else:
            prob_list = []
        for this_node in node.ancestors:
            if this_node.probability is not None:
                prob_list.append(this_node.probability)
        return prob_list

    def plot_sequence_depth(self, sequence_depth: int, filename: str = None):
        """Plots all states that occur in the tree until sequence_depth"""
        if not self.do_plot:
            return
        node_list = self.get_nodes_sequence_depth(sequence_depth)
        for node in node_list:
            state = self.sequence_data[node.data_index]
            self.code.add_state_to_plot(state)
        self.code.plot_all_states(filename, number_of_states=None)

    def perform_decay(
        self,
        action_name: str,
        decay_time: float,
        apply_method: bool = None,
        individual_op: bool = False,
    ):
        """Applies a decay operator and populates the tree

        action_name: String describing the operator
        decay_time: Free evolution time after applying the ideal operator
        apply_method: If None, applies either black body radiation delta_J=+-1 (self.apply_bbr_decay=True)
                             or only spontaneous decay delta_J=-1  (self.apply_bbr_decay=False)
                      If not None, apply_method is directly called for simualting free evolution
        individual_decay: If True, spectroscopically resolvable m levels are assumed
        """
        if apply_method is None:
            if self.apply_bbr_decay:
                apply_method = self.apply_bbr
            else:
                apply_method = self.apply_decay

        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            # rho_decay = self.apply_decay(rho)
            rho_decay = apply_method(
                rho, final_time=decay_time, individual_op=individual_op
            )
            self.generate_node(
                state_name=node.state_name,
                action_name=action_name,
                sequence_depth=self.sequence_counter + 1,
                state=rho_decay,
                parent=node,
            )
        self.sequence_counter += 1

    def apply_decay(
        self, rho: q.Qobj, final_time: float = 1.0, individual_op: bool = False
    ) -> q.Qobj:
        """applies spontaneous decay. Only deltaJ=-1 out of the code manifold

        TODO: m-resolved regime is not implemented"""
        probability = 1 - np.exp(-final_time)
        if individual_op:
            raise RuntimeError(
                "m-resolved BBR in the correcable subspace is not implemented yet"
            )
            decay_op_dict = self.code.get_simple_decay_ops_individual()
        else:
            decay_op_dict = self.get_decay_op_dict()
        rho_list = []
        for decay_name, decay_op in decay_op_dict.items():
            rho_decay = self.apply_operator(rho, decay_op)
            rho_list.append(rho_decay)

        rho_decay_all = 0
        for rho_decay in rho_list:
            rho_decay_all += rho_decay
        rho_decay_all = rho_decay_all / rho_decay_all.norm()
        final_rho = probability * rho_decay_all + (1 - probability) * rho
        return final_rho

    def perform_J_check(self, action_name: str, delta_J: int) -> None:
        """Perform J check operator

        delta_J: Change in J to check for, relative to the code manifold
        """
        check_J = self.code.check_J_operator(J=self.code.code_manifold + delta_J)
        return self.perform_check(action_name, check_J)

    def perform_m_check(self, action_name: str, delta_m: int) -> None:
        """Perform m check operator

        delta_m: Change in m to check for. Checks only in code manifold"""
        check_m = self.code.check_m_operator(delta_m)
        self.perform_check(action_name, check_m)

    def perform_check(
        self, action_name: str, check_op: q.Qobj, apply_method: bool = None
    ) -> None:
        """Generic check function. Is used for J and m checks"""
        if apply_method is None:
            apply_method = self.apply_check_operator
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            out_list = apply_method(rho, check_op)
            for result_dict in out_list:
                rho_proj = result_dict["state"]
                outcome = np.real(result_dict["eigenvalue"])
                probability = result_dict["probability"]
                self.generate_node(
                    state_name=node.state_name,
                    action_name=action_name + str(outcome),
                    sequence_depth=self.sequence_counter + 1,
                    state=rho_proj,
                    outcome=outcome,
                    probability=probability,
                    parent=node,
                )
        self.sequence_counter += 1

    def perform_J_correction(self, action_name: str) -> None:
        """Apply J correction based on previously measured check operators"""
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            delta_j = None
            outcome_list = self.get_previous_outcomes(node)
            if outcome_list[0] == -1.0:
                delta_j = -1
            if outcome_list[1] == -1.0:
                delta_j = +1
            if delta_j is not None:
                J_corr_op = self.code.get_decay_op(
                    self.code.code_manifold - delta_j,
                    delta_j=delta_j,
                    delta_m=0,
                    equal_coupling=True,
                )
                rho_correct = self.apply_operator(rho, J_corr_op)
            else:
                rho_correct = rho
            self.generate_node(
                state_name=node.state_name,
                action_name=action_name,
                sequence_depth=self.sequence_counter + 1,
                state=rho_correct,
                parent=node,
            )
        self.sequence_counter += 1

    def perform_m_correction(self, action_name: str) -> None:
        """Apply m correction based on previously measured check operators"""
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            delta_m = None
            outcome_list = self.get_previous_outcomes(node)
            if outcome_list[0] == -1.0:
                delta_m = 1
            if outcome_list[1] == -1.0:
                delta_m = -1
            if delta_m is not None:
                m_corr_op = self.code.get_decay_op(
                    self.code.code_manifold, 0, delta_m, equal_coupling=True
                )

                rho_correct = self.apply_operator(rho, m_corr_op)
            else:
                rho_correct = rho
            self.generate_node(
                state_name=node.state_name,
                action_name=action_name,
                sequence_depth=self.sequence_counter + 1,
                state=rho_correct,
                parent=node,
            )
        self.sequence_counter += 1

    def find_refreshing_amplitudes(
        self, angle_list: np.ndarray = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ) -> None:
        """Function to numerically find the optimal refreshing amplitudes.
        Only needed to generate the refreshing amplitude list once."""
        max_angle_list = []
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            state_name = node.state_name
            ideal_state = self.basis[state_name]
            s_p_25 = 1j * self.code.sigma_j_k(self.code.code_manifold, -2, 0, 7)
            s_p_52 = 1j * self.code.sigma_j_k(self.code.code_manifold, -5, 0, 7)

            fid_list = []
            # angle_list = np.linspace(0.14,0.16,20) * np.pi
            for rot_angle in angle_list:
                h_25 = s_p_25 + s_p_25.dag()
                h_25 = (-1j * rot_angle * h_25).expm()
                h_52 = s_p_52 + s_p_52.dag()
                h_52 = (-1j * rot_angle * h_52).expm()
                rho_rot = h_52 * h_25 * rho * h_25.dag() * h_52.dag()
                fid = q.fidelity(ideal_state, rho_rot) ** 2
                fid_list.append(fid)
            plt.figure()
            plt.plot(angle_list / np.pi, fid_list)
            outcome_list = self.get_previous_outcomes(node)
            max_angle = angle_list[np.argmax(fid_list)] / np.pi
            print(outcome_list, max_angle)
            plt.savefig(self.file_prefix + "{node.name}-{outcome_list}-angle.pdf")
            max_angle_list.append([outcome_list, max_angle])
        print(max_angle_list)

    def perform_refresh(self, action_name: str) -> None:
        """Apply refresh operations on tree based on previous measurement outcomes"""
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            rho = self.sequence_data[node.data_index]
            outcome_list = self.get_previous_outcomes(node)
            rho_decay = self.apply_refresh(rho, outcome_list)

            self.generate_node(
                state_name=node.state_name,
                action_name=action_name,
                sequence_depth=self.sequence_counter + 1,
                state=rho_decay,
                parent=node,
            )
        self.sequence_counter += 1

    def set_final_nodes_done(self) -> None:
        """Flag that sequence is done at current depth"""
        for node in self.get_nodes_sequence_depth(self.sequence_counter):
            node.sequence_done = True

    def apply_refresh(self, rho: q.Qobj, outcome_list: list) -> q.Qobj:
        """Apply refresh operator on state"""
        rot_angle0 = 0
        rot_angle1 = 0
        for this_outcome_list, this_rot_angle in self.refresh_amplitude0:
            if this_outcome_list == outcome_list:
                rot_angle0 = this_rot_angle * np.pi

        for this_outcome_list, this_rot_angle in self.refresh_amplitude1:
            if this_outcome_list == outcome_list:
                rot_angle1 = this_rot_angle * np.pi

        s_p_25 = 1j * self.code.sigma_j_k(self.code.code_manifold, -2, 0, 7)
        s_p_52 = 1j * self.code.sigma_j_k(self.code.code_manifold, -5, 0, 7)

        fid_list = []
        h_25 = s_p_25 + s_p_25.dag()
        h_25 = (-1j * rot_angle0 * h_25).expm()
        h_52 = s_p_52 + s_p_52.dag()
        h_52 = (-1j * rot_angle1 * h_52).expm()
        rho_rot = h_52 * h_25 * rho * h_25.dag() * h_52.dag()
        return rho_rot

    def get_state_dict(
        self, sequence_depth: bool = None, final_state: bool = False
    ) -> dict:
        """Returns a dictionary of states with given sequence depth or returns all final states"""
        if sequence_depth is None:
            sequence_depth = self.sequence_counter
        if final_state:
            node_iterator = find_attrs(
                self.root_node, attr_name="sequence_done", attr_value=True
            )
        else:
            node_iterator = self.get_nodes_sequence_depth(sequence_depth)
        final_rho_dict = {}
        for key in self.basis.keys():
            final_rho_dict[key] = 0  # q.qzero(self.code.dim)
        for node in node_iterator:
            rho = self.sequence_data[node.data_index]
            prob_list = self.get_previous_probabilitites(node)
            this_prob = np.prod(prob_list)
            state_name = node.state_name
            final_rho_dict[state_name] += rho * this_prob
        return final_rho_dict

    def characterize_single_cycle(self) -> dict:
        """Characterize the logical fidelity of a single QEC cycle"""
        print(
            f"State analysis for delay time: {self.time_decay} and m time scale {self.time_m}"
        )
        # Data format {'+':[['1,1,1,1',prob,log_fidelity,p],['1,1,1,-1'],... ]]}
        result_dict = {}
        for key in self.basis.keys():
            result_dict[key] = []
        for (
            node
        ) in self.get_nodes_final():  # get_nodes_sequence_depth(sequence_length):
            rho = self.sequence_data[node.data_index]
            outcome_list = self.get_previous_outcomes(node)
            prob_list = self.get_previous_probabilitites(node)
            this_prob = np.prod(prob_list)
            state_name = node.state_name
            rho_init = self.basis[state_name]
            physical_fid = q.fidelity(rho, rho_init) ** 2
            log_exp_value = (q.expect(self.logical_op_dict[state_name], rho) + 1) / 2
            print(
                f"|{state_name}> {outcome_list} with prob {this_prob:.5f}: {physical_fid:.3f} {log_exp_value}:.3f "
            )
            result_list = [outcome_list, this_prob, log_exp_value, physical_fid]
            result_dict[state_name].append(result_list)
        return result_dict

    def plot_single_cycle(
        self,
        filename: str = None,
        min_prob: float = 1e-4,
        no_datasets: int = 1,
        idx_data: int = 0,
        plot_prob: bool = True,
        label: str = None,
        fignum: int = 101,
    ):
        """Plot expectation values"""
        plot_width = 1 / (2 * no_datasets)
        plot_offset = plot_width * (idx_data - no_datasets / 2 + 0.5)
        char_dict = self.characterize_single_cycle()
        for state_name, char_list in char_dict.items():
            char_list = list(itertools.chain.from_iterable(char_list))
            prob_list = char_list[1::4]
            idx_list = list(np.argsort(prob_list))
            idx_list.reverse()
            idx_list = [idx for idx in idx_list if prob_list[idx] > min_prob]
            prob_list = [prob_list[idx] for idx in idx_list]
            out_list = char_list[0::4]
            out_str_list = [str([int(o) for o in out_list[idx]]) for idx in idx_list]
            exp_list = char_list[2::4]
            exp_list = [1 - exp_list[idx] for idx in idx_list]
            plt.figure(fignum)
            if plot_prob:
                plt.subplot(2, 1, 1)
                plt.bar(
                    np.arange(len(idx_list)) + plot_offset, prob_list, width=plot_width
                )
                plt.yscale("log")
                plt.xticks([])
                plt.ylabel("Probability")
                plt.subplot(2, 1, 2)
            plt.bar(np.arange(len(idx_list)) + plot_offset, exp_list, width=plot_width)
            plt.xticks(np.arange(len(idx_list)), out_str_list)
            plt.xticks(rotation=70)
            plt.xlabel("Outcomes")
            plt.ylabel("Logical infidelity")
            plt.tight_layout()
            if filename is not None:
                plt.savefig(filename)
            return np.max(exp_list)

    def characterize_results(
        self, sequence_length: int = None, effective_time: float = None
    ):
        if sequence_length is None:
            sequence_length = self.sequence_counter
        if effective_time is None:
            effective_time = self.time_decay
        print(
            f"State analysis for delay time: {self.time_decay} and m time scale {self.time_m}"
        )
        for (
            node
        ) in self.get_nodes_final():  # get_nodes_sequence_depth(sequence_length):
            rho = self.sequence_data[node.data_index]
            outcome_list = self.get_previous_outcomes(node)
            prob_list = self.get_previous_probabilitites(node)
            this_prob = np.prod(prob_list)
            state_name = node.state_name
            rho_init = self.basis[state_name]
            physical_fid = q.fidelity(rho, rho_init) ** 2
            print(
                f"|{state_name}> {outcome_list} with prob {this_prob:.5f}: {physical_fid:.5f} "
            )
        print("#############")
        combined_state_dict = self.get_state_dict(final_state=True)
        for state_name, rho_combined in combined_state_dict.items():
            rho_init = self.basis[state_name]
            physical_fid_combined = q.fidelity(rho_combined, rho_init) ** 2
            log_exp_value = (
                q.expect(self.logical_op_dict[state_name], rho_combined) + 1
            ) / 2
            print(
                f"Physical fidelity of combined state |{state_name}> : {physical_fid_combined:.5f} "
            )
            print(f" (<Z>+1)/2 of combined state |{state_name}> : {log_exp_value:.5f} ")
        print("#############")
        for state_name, rho_initial in self.basis.items():
            rho_decay = self.apply_bbr(rho_initial, effective_time)
            physical_fid_uncorr = q.fidelity(rho_decay, self.basis[state_name]) ** 2
            log_exp_value_uncorr = (
                q.expect(self.logical_op_dict[state_name], rho_decay) + 1
            ) / 2
            print(
                f"Physical fidelity of uncorrected state |{state_name}> : {physical_fid_uncorr:.5f} "
            )
            print(
                f" (<Z>+1)/2 of uncorrected state |{state_name}> : {log_exp_value_uncorr:.5f} "
            )
        print("#############\n")
        return log_exp_value, log_exp_value_uncorr

    def full_sequence_ideal(
        self,
        do_refresh: bool = True,
        time_decay: float = 0.1,
        time_m: float = 0.001,
        initial_state_dict: bool = None,
        do_final_decay: bool = False,
    ):
        """Simulate a full AE error correction sequence with ideal operators

        do_refresh: If True, the refresh operators will be applied
        time_decay: Free evolution time between each operator
        time_m: Additional time for m checks
        initial_state_dict: If None, all basis states will be simulateod
        do_final_decay: If True, a final decay will be performed after the last operator
        """
        if initial_state_dict is None:
            initial_state_dict = self.basis
        else:
            self.initial_state_dict = initial_state_dict
        self.time_decay = time_decay
        self.time_m = time_m
        self.sequence_data = []
        self.root_node = Node(name="Root", outcome=None, probability=1.0)
        self.root_list = []
        self.sequence_counter = 1
        i = len(self.sequence_data)
        action_name = "init"

        for state_name, state in initial_state_dict.items():
            self.generate_node(
                state_name=state_name,
                action_name=action_name,
                sequence_depth=self.sequence_counter,
                state=state,
                parent=self.root_node,
            )

        action_name = "decay"
        self.perform_decay(
            action_name, self.time_decay, individual_op=self.indvidual_op
        )
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "decay.pdf")

        action_name = "checkJ-"
        self.perform_J_check(action_name, -1)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "j-proj.pdf")

        action_name = "checkJ+"
        self.perform_J_check(action_name, +1)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "j+proj.pdf")

        action_name = "correct_J"
        self.perform_J_correction(action_name)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "j-corr.pdf")

        action_name = "checkm_p"
        self.perform_m_check(action_name, +1)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "m+proj.pdf")

        action_name = "checkm_m"
        self.perform_m_check(action_name, -1)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "m-proj.pdf")

        action_name = "correct_m"
        self.perform_m_correction(action_name)
        self.plot_sequence_depth(self.sequence_counter, self.file_prefix + "m-corr.pdf")

        if do_refresh is True:
            action_name = "refresh"
            self.perform_refresh(action_name)
        self.plot_sequence_depth(
            self.sequence_counter, self.file_prefix + "full_corr.pdf"
        )

        if do_final_decay:
            action_name = "final_decay"
            self.perform_decay(
                action_name, self.time_decay, individual_op=self.indvidual_op
            )
            self.plot_sequence_depth(
                self.sequence_counter, self.file_prefix + "final_decay.pdf"
            )

        self.set_final_nodes_done()
        # print(self.get_previous_outcomes(node))
        # print_tree(self.root_node, attr_list=['probability', 'outcome'])


if __name__ == "__main__":
    c_op = CheckOperatorDynamics(
        full_basis=True,
        do_plot=True,
        apply_bbr_decay=False,
        individual_op=False,
        approx_code=False,
    )
    c_op.full_sequence_ideal(do_refresh=True, time_decay=1)
    corr_fid, uncorr_fid = c_op.characterize_results()
