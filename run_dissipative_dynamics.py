"""
Simulations of toy model system for molecular absorption-emission dissipative error correction using Lindblad master equation formalism within qutip.
Author: Brandon Furey - 2024
"""

# IMPORT MODULES
import numpy as np
import qutip as q
import imageio as imageio
from rot_qec_counter import *
import time

# Simulation script

# Start time
start = time.time()

# DEFINITIONS FOR QEC CODE AND MOLECULAR TRANSITIONS
code_manifold = 7  # J manifold to encode in linear rotor
buffer_up = (
    3  # Additional buffer levels up (at least one for absorption and one more if second-order effects to be studied)
)
buffer_do = code_manifold  # Additional buffer levels down (at least one for absorption and one more if second-order effects to be studied)
max_l = code_manifold + buffer_up  # Maximum J manifold to consider
min_l = code_manifold - buffer_do  # Minimum J manifold to consider
cs_m1 = 2  # m1 value to encode countersymmetric code in
cs_m2 = 5  # m1 value to encode countersymmetric code in

code = CounterSymmetricCode(
    max_l, code_manifold, cs_m1, cs_m2, min_l
)  # instance of counter-symmetric class for [7,2,5] counter-symmetric code embedded in max_l Hilbert space

bbr_rate = 1  # generic rate for code manifold linewidth to be = 1
code_linewidth = bbr_rate  # generic rate related to BBR rate by Eq. 43
rco = True  # relative coherences only in X operator (build as described in paper)

# ENVIRONMENT DISSIPATION OPERATORS - BBR only

# Starting angular momentum for building decay operator family
start_op_l = 1  # initialize
if min_l == 0:
    start_op_l = 1
else:
    start_op_l = min_l

# Build error operator family
bbr_operator_family = (
    [code.get_decay_operator(j, -1, bbr_rate, hc=True) for j in range(start_op_l, max_l + 1, 1)]
    + [code.get_decay_operator(j, 0, bbr_rate, hc=True) for j in range(start_op_l, max_l + 1, 1)]
    + [code.get_decay_operator(j, 1, bbr_rate, hc=True) for j in range(start_op_l, max_l + 1, 1)]
)

dissipation_error_family = bbr_operator_family
sum_dissipation_error_family = sum(dissipation_error_family)

######################################### SIMULATIONS ####################################################

# DO NOTHING #############################################################################################

H = q.identity(code.dim)  # Basic Hamiltonian (does nothing)

# REPUMPING ONLY #########################################################################################

# SIMULATION 1: ERROR STATE, REPUMP J, NO BBR OR SD ######################################################

# Create repumping Hamiltonian
jcode_i = []
jcode_up_i = []
jcode_do_i = []

repump_factor_list = [10, 100, 1000]
repump_rate_list = [repump_factor * code_linewidth for repump_factor in repump_factor_list]
H_J_repump_sideband_list = [code.get_J_repump_sideband_H(repump_rate, repump_rate) for repump_rate in repump_rate_list]

# Motional mode dissipation operators
cool_factor = 2
cool_up_rate_list = [cool_factor * repump_rate for repump_rate in repump_rate_list]
cool_do_rate_list = cool_up_rate_list
cooling_up_list = [code.get_cooling_up(cool_up_rate) for cool_up_rate in cool_up_rate_list]
cooling_do_list = [code.get_cooling_do(cool_do_rate) for cool_do_rate in cool_do_rate_list]

# Create dissipation operator families and repumping coupling Hamiltonian

# tensor product of all operators in dissipation error family with identities on motional mode qubits (to allow driving sidebands)
dissipation_error_sb_family = [
    q.tensor(operator, q.identity(2), q.identity(2)) for operator in dissipation_error_family
]
# family of cooling operators
cooling_sb_family_list = [[cooling_up_list[i], cooling_do_list[i]] for i in range(len(repump_factor_list))]
# enlarge dissipation operator family to include cooling modeled as the operator defined above on motional mode qubit
dissipation_error_sb_and_cooling_family_list = [
    dissipation_error_sb_family + cooling_sb_family for cooling_sb_family in cooling_sb_family_list
]

# parameters for simulation
max_time_1 = 0.05
num_times_1 = 1000
times_1 = np.linspace(0, max_time_1, num_times_1)
options_1 = {
    "store_final_state": True,
    "store_states": True,
    "nsteps": 1e9,
    "max_step": 1e-4,
    "progress_bar": True,
    "atol": 1e-8,
}
filename_sim_1a = "qobj_files//results_1a"
filename_sim_1b = "qobj_files//results_1b"
filename_sim_1c = "qobj_files//results_1c"
filename_sim_1d = "qobj_files//results_1d"

# states for simulation
psi_0_sb = q.tensor(code.psi_0, q.basis(2, 0), q.basis(2, 0))
psi_0_sb_err_a = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, -1, 0) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_0,
    q.basis(2, 0),
    q.basis(2, 0),
)
psi_0_sb_err_b = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 1, 0) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_0,
    q.basis(2, 0),
    q.basis(2, 0),
)
psi_p_sb = q.tensor(code.psi_p, q.basis(2, 0), q.basis(2, 0))
psi_p_sb_err_c = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, -1, 0) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_p,
    q.basis(2, 0),
    q.basis(2, 0),
)
psi_p_sb_err_d = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 1, 0) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_p,
    q.basis(2, 0),
    q.basis(2, 0),
)

# simulations
rf_idx = 2  # repump factor index to run simulations on
results = q.mesolve(
    H_J_repump_sideband_list[rf_idx],
    psi_0_sb_err_a,
    times_1,
    cooling_sb_family_list[rf_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.ket2dm(psi_0_sb),
    ],
    options=options_1,
)
q.qsave(results, filename_sim_1a)  # save simulation results
print("simulation 1a complete")
results = q.mesolve(
    H_J_repump_sideband_list[rf_idx],
    psi_0_sb_err_b,
    times_1,
    cooling_sb_family_list[rf_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.ket2dm(psi_0_sb),
    ],
    options=options_1,
)
q.qsave(results, filename_sim_1b)  # save simulation results
print("simulation 1b complete")
results = q.mesolve(
    H_J_repump_sideband_list[rf_idx],
    psi_p_sb_err_c,
    times_1,
    cooling_sb_family_list[rf_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.ket2dm(psi_p_sb),
    ],
    options=options_1,
)
q.qsave(results, filename_sim_1c)  # save simulation results
print("simulation 1c complete")
results = q.mesolve(
    H_J_repump_sideband_list[rf_idx],
    psi_p_sb_err_d,
    times_1,
    cooling_sb_family_list[rf_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.ket2dm(psi_p_sb),
    ],
    options=options_1,
)
q.qsave(results, filename_sim_1d)  # save simulation results
print("simulation 1d complete")

# SIMULATION 2: LOGICAL STATE, REPUMP J, BBR DECAYS BUT NO SD ####################################################

# parameters for simulation
max_time_2 = 2.0
num_times_2 = 1000
times_2 = np.linspace(0, max_time_2, num_times_2)
options_2 = {
    "store_final_state": True,
    "store_states": True,
    "nsteps": 1e9,
    "max_step": 1e-4,
    "progress_bar": True,
    "atol": 1e-8,
}
psi_0_sb = q.tensor(code.psi_0, q.basis(2, 0), q.basis(2, 0))
psi_p_sb = q.tensor(code.psi_p, q.basis(2, 0), q.basis(2, 0))
filename_sim_2a = "qobj_files//results_2a"
filename_sim_2b = "qobj_files//results_2b"
filename_sim_2c = "qobj_files//results_2c"
filename_sim_2d = "qobj_files//results_2d"

# simulations
results_list = [
    q.mesolve(
        H_J_repump_sideband_list[i],
        psi_0_sb,
        times_2,
        dissipation_error_sb_and_cooling_family_list[i],
        [
            code.get_J_op_sb(),
            code.get_motion_up(),
            code.get_motion_do(),
            code.get_motion_up_num(),
            code.get_motion_do_num(),
            q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
            q.tensor(
                code.logical_x_operator(rel_coherences_only=rco),
                q.identity(2),
                q.identity(2),
            ),
            q.ket2dm(psi_0_sb),
        ],
        options=options_2,
    )
    for i in range(len(repump_factor_list))
]
q.qsave(results_list, filename_sim_2a)  # save simulation results
print("simulation list 2a complete")

results = q.mesolve(
    H * 0,
    code.psi_0,
    times_2,
    dissipation_error_family,
    [
        code.get_J_operator(),
        code.logical_z_operator(),
        code.logical_x_operator(rel_coherences_only=rco),
        q.ket2dm(code.psi_0),
    ],
    options=options_2,
)
q.qsave(results, filename_sim_2b)  # save simulation results
print("simulation 2b complete")

results_list = [
    q.mesolve(
        H_J_repump_sideband_list[i],
        psi_p_sb,
        times_2,
        dissipation_error_sb_and_cooling_family_list[i],
        [
            code.get_J_op_sb(),
            code.get_motion_up(),
            code.get_motion_do(),
            code.get_motion_up_num(),
            code.get_motion_do_num(),
            q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
            q.tensor(
                code.logical_x_operator(rel_coherences_only=rco),
                q.identity(2),
                q.identity(2),
            ),
            q.ket2dm(psi_p_sb),
        ],
        options=options_2,
    )
    for i in range(len(repump_factor_list))
]
q.qsave(results_list, filename_sim_2c)  # save simulation results
print("simulation list 2c complete")

results = q.mesolve(
    H * 0,
    code.psi_p,
    times_2,
    dissipation_error_family,
    [
        code.get_J_operator(),
        code.logical_z_operator(),
        code.logical_x_operator(rel_coherences_only=rco),
        q.ket2dm(code.psi_p),
    ],
    options=options_2,
)
q.qsave(results, filename_sim_2d)  # save simulation results
print("simulation 2d complete")

# ZEEMAN CORRECTION #########################################################################################

# SIMULATION 3: LOGICAL STATE, REPUMP J, ZDEC, BBR DECAYS BUT NO SD ################################################

# Create J repumping Hamiltonian
repump_factor_elem = 2  # element of repump factor/rate list to do simulations on
H_J_repump_sideband_zm = code.get_J_repump_sideband_H(
    repump_rate_list[repump_factor_elem],
    repump_rate_list[repump_factor_elem],
    zeeman_modes=True,
)

# Create Zeeman correction Hamiltonian
zdec_factor_list = [
    10,
    20,
    100,
]  # Zeeman DEC relative rate to J repumping DEC rates (this is how much SLOWER, not faster)
zdec_rate_list = [repump_rate_list[repump_factor_elem] / zdec_factor for zdec_factor in zdec_factor_list]
H_Z_DEC_sideband_list_zm = [code.get_m_correction_sideband_H(zdec_rate, zdec_rate) for zdec_rate in zdec_rate_list]

# Create total DEC Hamiltonian
H_DEC_list = [H_J_repump_sideband_zm + H_Z_DEC_sideband_list_zm[i] for i in range(len(zdec_factor_list))]

# Motional mode dissipation operators for J repumping
cooling_factor_elem = repump_factor_elem
cooling_up_zm = code.get_cooling_up(cool_up_rate_list[cooling_factor_elem], zeeman_modes=True)
cooling_do_zm = code.get_cooling_do(cool_do_rate_list[cooling_factor_elem], zeeman_modes=True)

# Motional mode dissipation operators for m correction
cool_zm_factor = cool_factor
cool_p_rate_list = [cool_zm_factor * zdec_rate for zdec_rate in zdec_rate_list]
cool_m_rate_list = cool_p_rate_list
cooling_p_list_zm = [code.get_cooling_p(cool_p_rate) for cool_p_rate in cool_p_rate_list]
cooling_m_list_zm = [code.get_cooling_m(cool_m_rate) for cool_m_rate in cool_m_rate_list]

# Create dissipation operator families

# tensor product of all operators in dissipation error family with identities on motional mode qubits (to allow driving sidebands)
dissipation_error_sb_family_zm = [
    q.tensor(operator, q.identity(2), q.identity(2), q.identity(2), q.identity(2))
    for operator in dissipation_error_family
]

# family of cooling operators
cooling_sb_family_list_zm = [
    [cooling_up_zm, cooling_do_zm, cooling_p_list_zm[i], cooling_m_list_zm[i]] for i in range(len(zdec_factor_list))
]

# enlarge dissipation operator family to include cooling modeled as the operator defined above on motional mode qubit
dissipation_error_sb_and_cooling_family_list_zm = [
    dissipation_error_sb_family_zm + cooling_sb_family_zm for cooling_sb_family_zm in cooling_sb_family_list_zm
]

# parameters for simulation
max_time_3 = 2.0
num_times_3 = 50
times_3 = np.linspace(0, max_time_3, num_times_3)
options_3 = {
    "store_final_state": True,
    "store_states": True,
    "nsteps": 1e9,
    "max_step": 1e-4,
    "progress_bar": True,
    "atol": 1e-8,
}
psi_0_sb2 = q.tensor(code.psi_0, q.basis(2, 0), q.basis(2, 0), q.basis(2, 0), q.basis(2, 0))
psi_p_sb2 = q.tensor(code.psi_p, q.basis(2, 0), q.basis(2, 0), q.basis(2, 0), q.basis(2, 0))
filename_sim_3a = "qobj_files//results_3a"
filename_sim_3b = "qobj_files//results_3b"
filename_sim_3c = "qobj_files//results_3c"
filename_sim_3d = "qobj_files//results_3d"

# simulations
zdec_idx = 2  # choose which rate for Z DEC simulations
results = q.mesolve(
    H_DEC_list[zdec_idx],
    psi_0_sb2,
    times_3,
    dissipation_error_sb_and_cooling_family_list_zm[zdec_idx],
    [
        code.get_J_op_sb(zeeman_modes=True),
        code.get_motion_up(zeeman_modes=True),
        code.get_motion_do(zeeman_modes=True),
        code.get_motion_up_num(zeeman_modes=True),
        code.get_motion_do_num(zeeman_modes=True),
        code.get_motion_p(),
        code.get_motion_m(),
        code.get_motion_p_num(),
        code.get_motion_m_num(),
        q.tensor(
            code.logical_z_operator(),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.identity(2),
        ),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_0_sb2),
    ],
    options=options_3,
)
q.qsave(results, filename_sim_3a)  # save simulation results
print("simulation 3a complete")

results = q.mesolve(
    H * 0,
    code.psi_0,
    times_3,
    dissipation_error_family,
    [
        code.get_J_operator(),
        code.logical_z_operator(),
        code.logical_x_operator(rel_coherences_only=rco),
        q.ket2dm(code.psi_0),
    ],
    options=options_3,
)
q.qsave(results, filename_sim_3b)  # save simulation results
print("simulation 3b complete")

results = q.mesolve(
    H_DEC_list[zdec_idx],
    psi_p_sb2,
    times_3,
    dissipation_error_sb_and_cooling_family_list_zm[zdec_idx],
    [
        code.get_J_op_sb(zeeman_modes=True),
        code.get_motion_up(zeeman_modes=True),
        code.get_motion_do(zeeman_modes=True),
        code.get_motion_up_num(zeeman_modes=True),
        code.get_motion_do_num(zeeman_modes=True),
        code.get_motion_p(),
        code.get_motion_m(),
        code.get_motion_p_num(),
        code.get_motion_m_num(),
        q.tensor(
            code.logical_z_operator(),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.identity(2),
        ),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_p_sb2),
    ],
    options=options_3,
)
q.qsave(results, filename_sim_3c)  # save simulation results
print("simulation 3c complete")

results = q.mesolve(
    H * 0,
    code.psi_p,
    times_3,
    dissipation_error_family,
    [
        code.get_J_operator(),
        code.logical_z_operator(),
        code.logical_x_operator(rel_coherences_only=rco),
        q.ket2dm(code.psi_p),
    ],
    options=options_3,
)
q.qsave(results, filename_sim_3d)  # save simulation results
print("simulation 3d complete")


# SIMULATION 4: ERROR STATE AFTER PERFECT REPUMP J THEN DO ZDEC, NO BBR OR SD ################################################

# Create Zeeman correction Hamiltonian w/o repumping cooling modes
H_Z_DEC_sideband_list_zm_dr = [
    code.get_m_correction_sideband_H(zdec_rate, zdec_rate, repump_modes=False) for zdec_rate in zdec_rate_list
]

# Create dissipation operator families
# family of cooling operators
cooling_p_list_zm_dr = [code.get_cooling_up(cool_p_rate) for cool_p_rate in cool_p_rate_list]
cooling_m_list_zm_dr = [code.get_cooling_do(cool_m_rate) for cool_m_rate in cool_m_rate_list]
cooling_sb_family_list_zm_dr = [
    [cooling_p_list_zm_dr[i], cooling_m_list_zm_dr[i]] for i in range(len(zdec_factor_list))
]

# states for simulations
psi_0_sb = q.tensor(code.psi_0, q.basis(2, 0), q.basis(2, 0))
psi_0_sb_err_p = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 0, -1) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_0,
    q.basis(2, 0),
    q.basis(2, 0),
)
psi_0_sb_err_m = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 0, 1) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_0,
    q.basis(2, 0),
    q.basis(2, 0),
)

psi_p_sb = q.tensor(code.psi_p, q.basis(2, 0), q.basis(2, 0))
psi_p_sb_err_p = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 0, -1) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_p,
    q.basis(2, 0),
    q.basis(2, 0),
)
psi_p_sb_err_m = q.tensor(
    sum([code.sigma_j_k(code.code_manifold, m, 0, 1) for m in range(-code.code_manifold, code.code_manifold + 1, 1)])
    * code.psi_p,
    q.basis(2, 0),
    q.basis(2, 0),
)

# parameters for simulations
max_time_4 = 0.05
num_times_4 = 1000
times_4 = np.linspace(0, max_time_4, num_times_4)
options_4 = {
    "store_final_state": True,
    "store_states": True,
    "nsteps": 1e9,
    "max_step": 1e-4,
    "progress_bar": True,
    "atol": 1e-8,
}
filename_sim_4a = "qobj_files//results_4a"
filename_sim_4b = "qobj_files//results_4b"
filename_sim_4c = "qobj_files//results_4c"
filename_sim_4d = "qobj_files//results_4d"

# simulation
zdec_perfect_repump_rel_rate_idx = 0
results = q.mesolve(
    H_Z_DEC_sideband_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    psi_0_sb_err_p,
    times_4,
    cooling_sb_family_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_0_sb),
    ],
    options=options_4,
)
q.qsave(results, filename_sim_4a)  # save simulation results
print("simulation 4a complete")

results = q.mesolve(
    H_Z_DEC_sideband_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    psi_0_sb_err_m,
    times_4,
    cooling_sb_family_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_0_sb),
    ],
    options=options_4,
)
q.qsave(results, filename_sim_4b)  # save simulation results
print("simulation 4b complete")

results = q.mesolve(
    H_Z_DEC_sideband_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    psi_p_sb_err_p,
    times_4,
    cooling_sb_family_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_p_sb),
    ],
    options=options_4,
)
q.qsave(results, filename_sim_4c)  # save simulation results
print("simulation 4c complete")

results = q.mesolve(
    H_Z_DEC_sideband_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    psi_p_sb_err_m,
    times_4,
    cooling_sb_family_list_zm_dr[zdec_perfect_repump_rel_rate_idx],
    [
        code.get_J_op_sb(),
        code.get_motion_up(),
        code.get_motion_do(),
        code.get_motion_up_num(),
        code.get_motion_do_num(),
        q.tensor(code.logical_z_operator(), q.identity(2), q.identity(2)),
        q.tensor(
            code.logical_x_operator(rel_coherences_only=rco),
            q.identity(2),
            q.identity(2),
        ),
        q.ket2dm(psi_p_sb),
    ],
    options=options_4,
)
q.qsave(results, filename_sim_4d)  # save simulation results
print("simulation 4d complete")

# End time
end = time.time()
elapsed = end - start
print("Elapsed time = {} seconds".format(elapsed))
