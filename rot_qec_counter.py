"""
Class for molecular counter-symmetric absorption emission QEC code and associated methods.
Author: Brandon Furey and Philipp Schindler - 2024
"""

# IMPORT MODULES
import numpy as np
from numpy import pi
import qutip as q
from qutip import Qobj
import scipy
from scipy.sparse import dia_matrix, csr_matrix, coo_matrix
from sympy.physics.wigner import wigner_3j
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio as imageio
# import itertools

# Define class for counter symmetric code and associated methods
class CounterSymmetricCode:
    """Class for counter symmetric codes

    max_l: Size of Hilbert space in J
    code_manifold: J in which the AE-code is living
    m1, m2: m Values for the code manifold
    min_l: Minimum value of J for the Hilbert space. Can be used when small J are never populated
    debug: Enables terminal print debugging
    approx_code: Initializes an approximate AE code
    """

    def __init__(
        self,
        max_l: int,
        code_manifold: int,
        m1: int,
        m2: int,
        min_l: int = 0,
        debug: bool = False,
        approx_code: bool = False,
    ) -> None:
        self.debug = debug
        self.approx_code = approx_code
        self.states_to_plot = []  # ???
        self.max_l = max_l
        self.min_l = min_l
        self.m1 = m1
        self.m2 = m2
        self.code_manifold = code_manifold
        self.l_m_list = self.get_max_l_list()
        self.dim = len(self.l_m_list)
        self.create_simple_decay_ops()  # ???
        self.create_logical_states()
        self.create_unit_logical_states()
        self.logical_z_op = self.logical_z_operator()
        self.logical_x_op = self.logical_x_operator()
        self.min_code_distance = min([2 * m1, m2 - m1])
        if self.min_code_distance <= 1:
            self.correct_errs = 0
        else:
            self.correct_errs = int((self.min_code_distance - 1) / 2)

    def debug_print(self, my_str: str) -> None:
        if self.debug:
            print(my_str)

    def create_logical_states(self) -> None:
        """Creates logical basis states for the exact or approximate code"""
        if not self.approx_code:
            self.psi_0 = self.get_logical_value_countersymmetric(0)
            self.psi_1 = self.get_logical_value_countersymmetric(1)
        else:
            self.psi_0 = self.get_logcial_value_approx(0)
            self.psi_1 = self.get_logcial_value_approx(1)
        self.psi_p = 1 / np.sqrt(2) * (self.psi_0 + self.psi_1)
        self.psi_m = 1 / np.sqrt(2) * (self.psi_0 - self.psi_1)

    def create_unit_logical_states(self) -> None:
        """Create unit-amplitude (non-normalized) logical states. Works only for exact codes
        i.e. amplitude 1 for each state in logical state. Used for building logical operators.
        TODO: Quite a bit of code duplication. Can possibly be merged with create_logical_states
        """
        self.unit_0 = self.get_unit_logical_value_countersymmetric(0)
        self.unit_1 = self.get_unit_logical_value_countersymmetric(1)
        self.unit_p = self.unit_0 + self.unit_1
        self.unit_m = self.unit_0 - self.unit_1

    def create_simple_decay_ops(self) -> None:
        """Creates decay operators used for simulating ideal sequential QEC in check_opreator_dynamics.py"""
        self.decay_pi_op = self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=0)
        self.decay_sp_op = self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=1)
        self.decay_sm_op = self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=-1)
        self.decay_pi_op_p = self.get_decay_op(J=self.code_manifold, delta_j=1, delta_m=0)
        self.decay_sp_op_p = self.get_decay_op(J=self.code_manifold, delta_j=1, delta_m=1)
        self.decay_sm_op_p = self.get_decay_op(J=self.code_manifold, delta_j=1, delta_m=-1)

    def coupling(self, j: int, m: int, delta_j: int, delta_m: int) -> float:
        """Calculate coupling strength for dipole interaction between linear rotor states.
        This calculates the Slater integral associated with the transition.

        j: starting j
        m: starting m
        delta_j: change in j
        delta_m: change in m

        Returns relative coupling strength
        """
        J0 = j
        J1 = j + delta_j
        m0 = m
        m1 = m + delta_m
        dm = delta_m
        coupling = (
            np.sqrt((2 * J0 + 1) * (2 * J1 + 1) * (3 / (4 * pi)))
            * float(wigner_3j(J0, 1, J1, m0, dm, -m1))
            * float(wigner_3j(J0, 1, J1, 0, 0, 0))
            * (-1) ** m1
        )
        return coupling

    def get_decay_operator(
        self,
        J_0: int,
        dm: int,
        rate: int,
        hc: bool = True,
        individual_decay: bool = False,
    ) -> Qobj:
        """Operator to describe decays like SD or BBR.
        This operator only acts between J_0 and J_0 - 1 manifolds, and can be defined for sigma or pi transitions.

        J_0: upper manifold of pair of connected manifolds
        dm: projection of angular momentum of photon
        rate: transition rate interaction (to be multiplied by the couplings from the Slater integrals)
        hc: Hermitian conjugate (True for BBR, False for spontaneous decay)

        returns decay operator
        """
        # Initializations
        lm_list = self.l_m_list
        orig_ind = []
        dec_ind = []
        row = []
        col = []
        data = []

        # Construction
        if J_0 > 0:  # else, proceed to calculate the operator as normal
            # Identify components in relevant manifolds
            for i, lm_tuple in enumerate(lm_list):
                if lm_tuple[0] == J_0:
                    orig_ind.append(i)
                elif lm_tuple[0] == J_0 - 1:
                    dec_ind.append(i)

            for index_o in orig_ind:
                j_o, m_o = lm_list[index_o]
                # Decays (these include both emissions and absorptions, coupling J_0 and J_0 - 1 by a dm transition - these are unresolved transitions)
                for index_d in dec_ind:
                    j_d, m_d = lm_list[index_d]
                    if m_d == m_o + dm:
                        transition_coupling2_do = rate * (4 * pi / 3) * abs(self.coupling(j_o, m_o, -1, dm)) ** 2
                        if hc:
                            transition_coupling2_up = rate * (4 * pi / 3) * abs(self.coupling(j_d, m_d, 1, -dm)) ** 2
                        else:
                            transition_coupling2_up = 0
                        val_do = np.sqrt(abs(transition_coupling2_do))
                        val_up = np.sqrt(abs(transition_coupling2_up))
                        row.extend([index_d, index_o])
                        col.extend([index_o, index_d])
                        data.extend([val_do, val_up])
        if individual_decay:
            proj_list = []
            for this_data, this_row, this_col in zip(data, row, col):
                row_arr = np.array([this_row])
                col_arr = np.array([this_col])
                data_arr = np.array([this_data])
                proj = csr_matrix((data_arr, (row_arr, col_arr)), shape=(self.dim, self.dim))
                proj_list.append(Qobj(proj))
            return proj_list
        else:
            row_arr = np.array(row)
            col_arr = np.array(col)
            data_arr = np.array(data)
            proj = csr_matrix((data_arr, (row_arr, col_arr)), shape=(self.dim, self.dim))
            return Qobj(proj)

    def get_J_operator(self) -> Qobj:
        """Generates a generic angular momentum J operator"""
        data = []
        for lm_tuple in self.l_m_list:
            data.append(lm_tuple[0])
        data_arr = np.array(data)
        offsets = np.array([0])
        J_op = dia_matrix((data_arr, offsets), shape=(self.dim, self.dim))
        return Qobj(J_op)

    def get_max_l_list(self) -> list:
        """Generates a list of [J,m] with all available J,m combinations"""
        l_m_list = []
        for l in range(self.min_l, self.max_l + 1, 1):
            for m_idx in range(2 * l + 1):
                m = m_idx - l
                l_m_list.append([l, m])
        return l_m_list

    def sigma_j_k(self, j: int, m: int, delta_j: int, delta_m: int) -> Qobj:
        """Generates an operator that couples |j,m> to |j+delta_j, m+delta_m
        The non-zero element of the operator has value 1
        """
        try:
            idx_0 = np.array([self.l_m_list.index([j, m])])
            idx_1 = np.array([self.l_m_list.index([j + delta_j, m + delta_m])])
            data = np.array([1])
            sigma_arr = coo_matrix((data, (idx_1, idx_0)), shape=(self.dim, self.dim))
            sigma_op = Qobj(sigma_arr)
            return sigma_op
            # indexing: [row,column] = [new,original]
        except ValueError:
            return 0

    def get_decay_op(self, J: int, delta_j: int, delta_m: int, equal_coupling: bool = False) -> Qobj:
        """Generates a decay operator acting on all m substates in a single J manifold

        J: original J manifold
        delta_j: difference in J
        delta_m: difference in m
        equal_coupling: if True, all m states will couple the same. Non-zero values are 1.0

        returns the decay operator
        """
        decay_op = 0 * self.sigma_j_k(J, 0, delta_j, delta_m)
        for m in range(-J, +J + 1):
            if equal_coupling:
                value = 1.0
            else:
                value = self.coupling(J, m, delta_j, delta_m)
            decay_op += value * self.sigma_j_k(J, m, delta_j, delta_m)
        return decay_op

    def get_logical_value_countersymmetric(self, r: int) -> Qobj:
        """Generates the logical computational basis state for the exact code

        r: integer that is either 0, or 1 for the two basis states

        returns a ket Qobj
        """
        m1 = self.m1
        m2 = self.m2
        J = self.code_manifold
        idx = []
        data = []
        psi_l_m = q.zero_ket(self.dim)

        for l in range(self.min_l, self.max_l + 1, 1):
            for m in range(-l, l + 1, 1):
                if l == J:
                    if r == 0:
                        if m == -m1:
                            value = np.sqrt(m2 / (m1 + m2))
                            idx.append(self.l_m_list.index([l, m]))
                            data.append(value)
                        elif m == m2:
                            value = np.sqrt(m1 / (m1 + m2))
                            idx.append(self.l_m_list.index([l, m]))
                            data.append(value)
                    if r == 1:
                        if m == -m2:
                            value = np.sqrt(m1 / (m1 + m2))
                            idx.append(self.l_m_list.index([l, m]))
                            data.append(value)
                        elif m == m1:
                            value = np.sqrt(m2 / (m1 + m2))
                            idx.append(self.l_m_list.index([l, m]))
                            data.append(value)
        for i, id in enumerate(idx):
            component = data[i] * q.basis(self.dim, id)
            psi_l_m = psi_l_m + component
        psi_l_m = psi_l_m / psi_l_m.norm()
        return psi_l_m

    def get_logcial_value_approx(self, r: int) -> Qobj:
        """Generates the logical computational basis state for the exact code

        r: integer that is either 0, or 1 for the two basis states

        returns a ket Qobj
        """
        J = self.code_manifold
        value = 1.0
        if r == 0:
            m = 2
        else:
            m = -2
        idx = self.l_m_list.index([J, m])
        proj = [0] * self.dim
        proj[idx] = value
        psi_l_m = Qobj(proj)
        return psi_l_m

    def get_unit_logical_value_countersymmetric(self, r):
        """Unit-amplitude logical states used for building operators.

        r: integer that is either 0, or 1 for the two basis states

        returns a ket Qobj

        TODO: Can probably merged with get_logical_value_approx
        """
        m1 = self.m1
        m2 = self.m2
        J = self.code_manifold
        idx = []
        psi_l_m = q.zero_ket(self.dim)
        for l in range(self.min_l, self.max_l + 1, 1):
            for m in range(-l, l + 1, 1):
                if l == J:
                    if r == 0:
                        if m == -m1:
                            idx.append(self.l_m_list.index([l, m]))
                        elif m == m2:
                            idx.append(self.l_m_list.index([l, m]))
                    if r == 1:
                        if m == -m2:
                            idx.append(self.l_m_list.index([l, m]))
                        elif m == m1:
                            idx.append(self.l_m_list.index([l, m]))
        for id in idx:
            component = q.basis(self.dim, id)
            psi_l_m = psi_l_m + component
        return psi_l_m

    def logical_z_operator(self, unit: bool = True, diag: bool = True) -> Qobj:
        """Generates the logical Z-operator

        unit: If True the unit amplitude state vectors will be used
        diag: If True, only the diagonal matrix is returned. Useful to check only for population overlap.

        Returns the operator as a Qobj
        """
        if unit:
            psi_0 = self.unit_0
            psi_1 = self.unit_1
        else:
            psi_0 = self.psi_0
            psi_1 = self.psi_1
        op = psi_0 * psi_0.dag() - psi_1 * psi_1.dag()
        if diag:
            op_return = q.qdiags(op.diag(), 0)  # pick out only diagonals
        else:
            op_return = op
        return op_return

    def logical_x_operator(self, unit: bool = True, rel_coherences_only: bool = True) -> Qobj:
        """Generates the logical X-operator

        unit: If True the unit amplitude state vectors will be used
        rel_coherences_only: If True, only relative coherences between logical states are evaluated

        TODO: More explanations on relative coherences needed

        Returns the operator as a Qobj
        """
        if self.approx_code:
            rel_coherences_only = False
        if unit:
            psi_0 = self.unit_0
            psi_1 = self.unit_1
        else:
            psi_0 = self.psi_0
            psi_1 = self.psi_1
        if rel_coherences_only:
            idx_list = [
                self.l_m_list.index([self.code_manifold, -self.m1]),
                self.l_m_list.index([self.code_manifold, self.m1]),
                self.l_m_list.index([self.code_manifold, -self.m2]),
                self.l_m_list.index([self.code_manifold, self.m2]),
            ]
            i_arr = np.array([idx_list[0], idx_list[2], idx_list[1], idx_list[3]])
            j_arr = np.array([idx_list[1], idx_list[3], idx_list[0], idx_list[2]])
            data = np.ones((4,))
            op_arr = coo_matrix((data, (i_arr, j_arr)), shape=(self.dim, self.dim))
            op = Qobj(op_arr)
        else:
            op = psi_0 * psi_1.dag() + psi_1 * psi_0.dag()
        return op

    def add_state_to_plot(self, rho: Qobj) -> None:
        """Adds states in the to-plot list for sequential QEC simulations"""
        self.states_to_plot.append(rho)

    def plot_all_states(self, filename: str = None, number_of_states: int = 2):
        """Plots states that are int the to-plot list for sequential QEC simulation

        filename: if None, the plot is not saved
        number_of_states: Only the first number_of_states entries will be plotted"""
        if number_of_states is None:
            number_of_states = len(self.states_to_plot)
        n_rows = int(np.ceil(number_of_states / 2))
        plt.figure(figsize=(12, 4 * n_rows))
        for i in range(number_of_states):
            psi = self.states_to_plot[i]
            plt.subplot(n_rows, 2, i + 1)
            self.rot_hinton(psi, cm_ident=True, cm_ident_color="m")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        self.states_to_plot = []

    def rot_hinton(
        self,
        data: Qobj,
        max_weight=1,
        ax=None,
        cm_ident=False,
        cm_ident_color="red",
        vector_hinton=False,
        vh_amp=False,
        vh_cmap="hsv",
        cmap_shift=0,
        pop_color="white",
        ax_color="k",
        ax_facecolor="gray",
        ax_bkgdcolor="white",
        label_color="k",
        grid_bool=True,
        grid_color="k",
        ax_labels_bool=True,
    ):
        """Draw Hinton diagram for visualizing a weight matrix."""
        if not vector_hinton or not data.isket:
            try:
                if not data.isket:
                    data = (
                        data.diag()
                    )  
            except AttributeError:
                pass
            sq_array = np.zeros((2 * self.max_l + 1, self.max_l + 1), dtype=complex)
            sq_array[:] = np.nan
            list_index = []
            for l, m in self.l_m_list:
                list_index.append([self.max_l + m, l])
            for data_idx, idx_tuple in enumerate(list_index):
                # print(data_idx, idx_tuple)
                sq_array[idx_tuple[0], idx_tuple[1]] = data[data_idx]
            sq_array[np.abs(sq_array) < 1e-10] = 0
            matrix = sq_array
            try:
                if data.isket:
                    matrix = abs(matrix) ** 2
            except AttributeError:
                pass
            ax = ax if ax is not None else plt.gca()
            # if not max_weight:
            #    max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

            ax.patch.set_facecolor(ax_facecolor)
            ax.set_aspect("equal", "box")
            # ax.xaxis.set_major_locator(plt.NullLocator())
            # ax.yaxis.set_major_locator(plt.NullLocator())

            for (x, y), w in np.ndenumerate(matrix):
                if not np.isnan(w):
                    face_color = pop_color
                    edge_color = None
                    size = np.sqrt(abs(w) / max_weight)
                else:
                    size = 1.0
                    face_color = ax_bkgdcolor
                    edge_color = ax_bkgdcolor
                rect = plt.Rectangle(
                    [x - size / 2 - self.max_l, y - size / 2],
                    size,
                    size,
                    facecolor=face_color,
                    edgecolor=edge_color,
                )
                ax.add_patch(rect)

        elif vector_hinton and data.isket:
            cmap = mpl.cm.get_cmap(vh_cmap)
            norm = mpl.colors.Normalize(
                vmin=-pi + np.finfo(float).eps + cmap_shift * 2 * pi,
                vmax=pi + cmap_shift * 2 * pi,
            )
            try:
                if not data.isket:
                    data = data.full.toarray()
            except AttributeError:
                pass
            sq_array = np.zeros((2 * self.max_l + 1, self.max_l + 1), dtype=complex)
            sq_array[:] = np.nan
            list_index = []
            for l, m in self.l_m_list:
                list_index.append([self.max_l + m, l])
            for data_idx, idx_tuple in enumerate(list_index):
                sq_array[idx_tuple[0], idx_tuple[1]] = data[data_idx]
            sq_array[np.abs(sq_array) < 1e-10] = 0
            matrix = sq_array
            ax = ax if ax is not None else plt.gca()
            # if not max_weight:
            #    max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

            ax.patch.set_facecolor(ax_facecolor)
            ax.set_aspect("equal", "box")
            # ax.xaxis.set_major_locator(plt.NullLocator())
            # ax.yaxis.set_major_locator(plt.NullLocator())

            for (x, y), w in np.ndenumerate(matrix):
                if not np.isnan(w):
                    face_color = cmap(norm(np.angle(w)))
                    edge_color = None
                    if not vh_amp:
                        # if vh_amp is False, then hinton plot has rectangles area ~ norm squared of amplitude
                        w_plot = abs(w) ** 2
                    else:
                        # else hinton plot has rectangles ~ norm of amplitude
                        w_plot = abs(w)
                    size = np.sqrt(w_plot / max_weight)
                else:
                    size = 1.0
                    face_color = ax_bkgdcolor
                    edge_color = ax_bkgdcolor
                rect = plt.Rectangle(
                    [x - size / 2 - self.max_l, y - size / 2],
                    size,
                    size,
                    facecolor=face_color,
                    edgecolor=edge_color,
                )
                ax.add_patch(rect)

        ax.set_ylim([-0.5, matrix.shape[1] - 0.5])
        ax.set_xlim([-self.max_l, self.max_l])
        ax.set_yticks(np.arange(0, self.max_l + 1))
        ax.set_yticks(np.arange(0, self.max_l + 2) - 0.5, minor=True)
        ax.set_xticks(np.arange(-self.max_l, self.max_l + 1))
        ax.set_xticks(np.arange(-self.max_l, self.max_l + 2) - 0.5, minor=True)
        ax.grid(grid_bool, which="minor", color=grid_color)
        ax.tick_params(which="minor", bottom=False, left=False)
        if ax_labels_bool:
            ax.set_xlabel("$m$")
            ax.set_ylabel("$J$")
        if cm_ident == True:
            cm_ident_rect = plt.Rectangle(
                [-self.code_manifold - 1 / 2, self.code_manifold - 1 / 2],
                2 * self.code_manifold + 1,
                1,
                fill=None,
                edgecolor=cm_ident_color,
                zorder=10,
            )
            ax.add_patch(cm_ident_rect)
        # ax.autoscale_view()
        # ax.invert_yaxis()
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.tick_params(axis="x", colors=ax_color)
        ax.tick_params(axis="y", colors=ax_color)
        ax.spines["left"].set_color(ax_color)
        ax.spines["bottom"].set_color(ax_color)
        ax.spines["right"].set_color(ax_color)
        ax.spines["top"].set_color(ax_color)
        ax.set_aspect("equal", adjustable="box")

    def get_J_repump_sideband_H(self, rate_param_up: float, rate_param_do: float, zeeman_modes: bool = False) -> Qobj:
        """Hamiltonian for repumping rotational manifold via pi transitions on 2 motional sidebands
        with coherent field in Hilbert space of [molecule,motion,motion].

        This can be extended to include the Zeeman DEC Hamiltonian by taking the tensor product of this with
        two more identity operators on 2D fock spaces.
        Zeeman_modes is an additional parameter which builds the full Hilbert space with two additional
        motional modes for use when also adding in the Zeeman correction.

        rate_param_up: Rabi rate in upwards direction
        rate_param_do: Rabi rate in downwards direction
        zeeman_modes: if True, 4 motional modes are used. If False, 2 motional modes are used
        """

        # Initializations
        repump_up_mol = q.qzero([self.dim])
        repump_do_mol = q.qzero([self.dim])

        # Upward J corrections
        repump_up_mol = sum(
            [
                np.sqrt(4 * pi / 3)
                * self.coupling(self.code_manifold - 1, m_err_up_state, 1, 0)
                * self.sigma_j_k(self.code_manifold - 1, m_err_up_state, 1, 0)
                for m_err_up_state in range(-self.code_manifold + 1, self.code_manifold, 1)
            ]
        )

        # Downward J corrections
        repump_do_mol = sum(
            [
                np.sqrt(4 * pi / 3)
                * self.coupling(self.code_manifold + 1, m_err_do_state, -1, 0)
                * self.sigma_j_k(self.code_manifold + 1, m_err_do_state, -1, 0)
                for m_err_do_state in range(-self.code_manifold - 1, self.code_manifold + 2, 1)
            ]
        )

        # Finalize construction on 2 motional sideband Hilbert space
        if not zeeman_modes:
            repump_up_sb = q.tensor(rate_param_up * repump_up_mol, q.create(2), q.identity(2))
            repump_do_sb = q.tensor(rate_param_do * repump_do_mol, q.identity(2), q.create(2))
            repump_sb = repump_up_sb + repump_do_sb
            repump_sb_tot = (
                repump_sb + repump_sb.dag()
            )  # sigma_j_k only acts one way, and the Hamiltonian needs to be unitary, so add the dagger

        # Finalize construction on 4 motional sideband Hilbert space (full space)
        else:
            repump_up_sb = q.tensor(
                rate_param_up * repump_up_mol,
                q.create(2),
                q.identity(2),
                q.identity(2),
                q.identity(2),
            )
            repump_do_sb = q.tensor(
                rate_param_do * repump_do_mol,
                q.identity(2),
                q.create(2),
                q.identity(2),
                q.identity(2),
            )
            repump_sb = repump_up_sb + repump_do_sb
            repump_sb_tot = (
                repump_sb + repump_sb.dag()
            )  # sigma_j_k only acts one way, and the Hamiltonian needs to be unitary, so add the dagger
        return repump_sb_tot

    def get_m_correction_sideband_H(
        self,
        rate_param_m: float,
        rate_param_p: float,
        repump_modes: bool = True,
    ) -> Qobj:
        """Operator to correct rotational manifolds on a sideband transition via resonant Raman transitions
        using directly input rates for sigma minus and sigma plus fields.

        This scheme is done by frequency-resolved transitions with corresponding sigma_plus or
        sigma_minus transitions and equal coupling is assumed.

        rate_param_m: Rabi rate for the minus m direction
        rate_param_p: Rabi Rate for the plus m direction
        repump_modes: If True: 4 motional modes are used, if False, 2 modes are used

        TODO: repump_modes has different name than in J-correction Hamiltonian
        """

        # Initializations
        m_target_states = [-self.m2, -self.m1, self.m1, self.m2]
        error_p_states = [x - 1 for x in m_target_states]
        error_m_states = [x + 1 for x in m_target_states]
        correct_p_mol = q.qzero([self.dim])
        correct_m_mol = q.qzero([self.dim])

        # Sigma plus corrections on molecule
        correct_p_mol = sum(
            [self.sigma_j_k(self.code_manifold, m_err_p_state, 0, 1) for m_err_p_state in error_p_states]
        )

        # Sigma minus corrections on molecule
        correct_m_mol = sum(
            [self.sigma_j_k(self.code_manifold, m_err_m_state, 0, -1) for m_err_m_state in error_m_states]
        )

        # Finalize construction on 4 motional sideband Hilbert space (full space)
        if repump_modes:
            correct_p_sb = q.tensor(
                rate_param_p * correct_p_mol,
                q.identity(2),
                q.identity(2),
                q.create(2),
                q.identity(2),
            )
            correct_m_sb = q.tensor(
                rate_param_m * correct_m_mol,
                q.identity(2),
                q.identity(2),
                q.identity(2),
                q.create(2),
            )
            correct_sb = correct_p_sb + correct_m_sb
            correct_sb_tot = (
                correct_sb + correct_sb.dag()
            )  # sigma_j_k only acts one way, and the Hamiltonian needs to be unitary, so add the dagger

        # Finalize construction on 2 motional sideband Hilbert space (Zeeman fock modes only in case of perfect repumping)
        else:
            correct_p_sb = q.tensor(rate_param_p * correct_p_mol, q.create(2), q.identity(2))
            correct_m_sb = q.tensor(rate_param_m * correct_m_mol, q.identity(2), q.create(2))
            correct_sb = correct_p_sb + correct_m_sb
            correct_sb_tot = (
                correct_sb + correct_sb.dag()
            )  # sigma_j_k only acts one way, and the Hamiltonian needs to be unitary, so add the dagger

        return correct_sb_tot

    def get_motion_up(self, zeeman_modes: bool = False) -> Qobj:
        """Qubit sigma_z operator on motional mode for up corrections in Hilbert space of [molecule,qubit,qubit].

        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), q.sigmaz(), q.identity(2))
        else:
            op = q.tensor(
                q.identity(self.dim),
                q.sigmaz(),
                q.identity(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_motion_do(self, zeeman_modes: bool = False) -> Qobj:
        """Qubit sigma_z operator on motional mode for down corrections in Hilbert space of [molecule,qubit,qubit].

        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), q.identity(2), q.sigmaz())
        else:
            op = q.tensor(
                q.identity(self.dim),
                q.identity(2),
                q.sigmaz(),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_motion_p(self) -> Qobj:
        """Qubit sigma_z operator on motional mode for sigma plus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit]."""
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            q.sigmaz(),
            q.identity(2),
        )
        return op

    def get_motion_m(self) -> Qobj:
        """Qubit sigma_z operator on motional mode for sigma minus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit]."""
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.sigmaz(),
        )
        return op

    def get_motion_up_num(self, zeeman_modes: bool = False) -> Qobj:
        """Qubit number operator on motional mode for up corrections
        in Hilbert space of [molecule,qubit,qubit].

        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), q.create(2) * q.destroy(2), q.identity(2))
        else:
            op = q.tensor(
                q.identity(self.dim),
                q.create(2) * q.destroy(2),
                q.identity(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_motion_do_num(self, zeeman_modes: bool = False) -> Qobj:
        """Qubit number operator on motional mode for down corrections
        in Hilbert space of [molecule,qubit,qubit].

        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), q.identity(2), q.create(2) * q.destroy(2))
        else:
            op = q.tensor(
                q.identity(self.dim),
                q.identity(2),
                q.create(2) * q.destroy(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_motion_p_num(self) -> Qobj:
        """Qubit number operator on motional mode for sigma plus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit]."""
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            q.create(2) * q.destroy(2),
            q.identity(2),
        )
        return op

    def get_motion_m_num(self) -> Qobj:
        """Qubit number operator on motional mode for sigma minus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit]."""
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.create(2) * q.destroy(2),
        )
        return op

    def get_J_op_sb(self, zeeman_modes: bool = False) -> Qobj:
        """Expectation value for J operator in Hilbert space including motional mode qubits
        (allowing for sideband couplings): [molecule,qubit,qubit].

        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(self.get_J_operator(), q.identity(2), q.identity(2))
        else:
            op = q.tensor(
                self.get_J_operator(),
                q.identity(2),
                q.identity(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_cooling_up(self, rate: float, zeeman_modes: bool = False) -> Qobj:
        """Cooling operator based on destroy operator on motional mode for up corrections
        in Hilbert space of [molecule,qubit,qubit].

        rate: Rabi frequency
        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), np.sqrt(rate) * q.destroy(2), q.identity(2))
        else:
            op = q.tensor(
                q.identity(self.dim),
                np.sqrt(rate) * q.destroy(2),
                q.identity(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_cooling_do(self, rate: float, zeeman_modes: bool = False) -> Qobj:
        """Cooling operator based on destroy operator on motional mode for down corrections
        in Hilbert space of [molecule,qubit,qubit].

        rate: Rabi rate
        zeeman_modes: If True, 4 motional modes are used. If False, 2 modes are used
        """
        if not zeeman_modes:
            op = q.tensor(q.identity(self.dim), q.identity(2), np.sqrt(rate) * q.destroy(2))
        else:
            op = q.tensor(
                q.identity(self.dim),
                q.identity(2),
                np.sqrt(rate) * q.destroy(2),
                q.identity(2),
                q.identity(2),
            )
        return op

    def get_cooling_p(self, rate: float) -> Qobj:
        """Cooling operator based on destroy operator on motional mode for sigma plus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit].

        rate: Rabi rate
        """
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            np.sqrt(rate) * q.destroy(2),
            q.identity(2),
        )
        return op

    def get_cooling_m(self, rate: float) -> Qobj:
        """Cooling operator based on destroy operator on motional mode for sigma minus corrections
        in Hilbert space of [molecule,qubit,qubit,qubit,qubit].

        rate: Rabi rate
        """
        op = q.tensor(
            q.identity(self.dim),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            np.sqrt(rate) * q.destroy(2),
        )
        return op

    def check_J_operator(self, J: int) -> Qobj:
        """Generates the check operator for being in the J manifold

        J: J value of tested manifold
        """
        diag_array = np.zeros(self.dim)
        for j in range(self.min_l, self.max_l + 1):
            for m in range(-j, j + 1, 1):
                idx = self.l_m_list.index([j, m])
                if j == J:
                    value = -1
                else:
                    value = 1
                diag_array.data[idx] = value
        return q.qdiags(diag_array, 0)

    def check_m_operator(self, delta_m: int) -> Qobj:
        """Generates the check opertor for a specific delta_m in the code J manifold

        delta_m: Checked m differece

        """
        J = self.code_manifold
        diag_array = np.ones(self.dim)
        m_list = [-self.m1, -self.m2, self.m1, self.m2]
        for m in range(-J, J + 1, 1):
            idx = self.l_m_list.index([J, m])
            if m - delta_m in m_list:
                value = -1
            else:
                value = 1
            diag_array.data[idx] = value
        return q.qdiags(diag_array, 0)

    def decay_evolution(
        self,
        H: Qobj,
        psi_0: Qobj,
        times: np.ndarray = None,
        c_op: list = None,
        expectation: list = None,
        options: dict = None,
    ):
        """Simulates a master euqation for a given Hamiltonian, subject to black body radiation

        H: Hamiltonian
        psi_0: Initial state
        times: simulation time array. If None, np.linspace(0,1) will be used
        c_op: List of jump operators. If None, spontaneous decay from the code manifold will be assumed
        expectation: List of operators to calculate the expectation values
        options: Options for the solver
        """
        if expectation is None:
            J_op = self.get_J_operator()
            if psi_0.isket:
                fid_op = q.ket2dm(psi_0)
            else:
                fid_op = psi_0
            log_z_op = self.decoding_z_operator()
            expectation = [J_op, fid_op, log_z_op]
        if c_op is None:
            c_op = [
                self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=0),
                self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=-1),
                self.get_decay_op(J=self.code_manifold, delta_j=-1, delta_m=1),
            ]
        if options is None:
            options = q.solver.Options()
            options["store_final_state"] = True
            options["store_states"] = True
            options["nsteps"] = 1e4
        if times is None:
            times = np.linspace(0, 1)
        results = q.mesolve(H, psi_0, times, c_op, expectation, options=options)
        return results

    def simulate_basis(self, basis: list = None):
        """Simualtes decay evolution for multiple states

        basis: List of states. If None, all four basis states will be used

        returns: list of states after the simulation."""
        if basis is None:
            basis = [self.psi_0, self.psi_1, self.psi_p, self.psi_m]
        final_rho_list = []
        for psi in basis:
            H = q.qzero([self.dim])
            results_1 = self.decay_evolution(H, psi)
            final_rho_list.append(results_1.final_state)
        return final_rho_list


def get_unique_projectors(
    operator: Qobj,
    state: Qobj,
    tol: float = 1e-10,
    sparse: bool = True,
    use_qutip_eig: bool = False,
    is_hermitian: bool = False,
    return_projector: bool = False,
) -> list:
    """Calcualtes unique projectors for a state and also the probabiliites

    operator: Projection operator
    state: State to tbe projected
    sparse: If True, the sparse version of qutip eigenstates functions is used
    use_qutip_eig: If True, the qutip internal eigenstates function is used
    is_hermition: Selects correct scipy.linalg eig function. Only active if use_qutip_eig=False
    return_projector: If True, the projector is returned instead of the state. Works only if state is a ket.

    Returns: unique_op_list, probability_list, unique_eigv_list

    TODO: Check what is still needed after the quip upgrade
    """
    if (return_projector is True) and (state.isket):
        state = q.ket2dm(state)

    if use_qutip_eig:
        eigv_list, eigs_list = operator.eigenstates(sparse=sparse)
    else:
        op_array = operator.full()
        if is_hermitian:
            eigv_list, eigs_array_list = scipy.linalg.eigh(op_array)
        else:
            eigv_list, eigs_array_list = scipy.linalg.eig(op_array)
        eigv_list = np.array(eigv_list)
        eigs_list = []
        for eigs_array in eigs_array_list:
            eigs_list.append(q.Qobj(eigs_array))

    unique_idx = 0
    unique_idx_list = []
    this_idx_list = []
    used_idx_list = []
    for idx, eigv in enumerate(eigv_list):
        if idx not in used_idx_list:
            idx_list = list(np.isclose(eigv_list[idx], eigv_list, atol=1e-5))
            similar_indices = [i for i, x in enumerate(idx_list) if x]
            this_idx_list.append(similar_indices)
            used_idx_list += similar_indices

    unique_idx_list = this_idx_list
    unique_op_list = []
    probability_list = []
    unique_eigv_list = []
    for idx_list in unique_idx_list:
        projected_state = 0
        eigv = eigv_list[idx_list[0]]
        unique_eigv_list.append(eigv)
        if state.isket and (return_projector is False):
            for eigstate in [eigs_list[i] for i in idx_list]:
                projected_state += eigstate.dag() * state * eigstate
            probability = projected_state.norm() ** 2
            norm = projected_state.norm()
            if norm < 1e-6:
                norm = 1
            unique_op_list.append(projected_state / norm)
            probability_list.append(probability)
        else:
            for eigstate in [eigs_list[i] for i in idx_list]:
                eig_proj = eigstate * eigstate.dag()
                projected_state += eig_proj.dag() * eig_proj
            # projected_state = projected_state / projected_state.norm()
            if not projected_state.dims == state.dims:
                projected_state.dims = state.dims
            probability = (projected_state * state).tr()
            unique_op_list.append(projected_state)
            probability_list.append(probability)
    return unique_op_list, probability_list, unique_eigv_list
