import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

"""Generates figure 5 from pre-saved data"""

appr_str = "approx"
exact_str = "exact"

color_list = [
    "#1F77B4",
    "#FF983C",
]

flist = [
    [exact_str, 0.002, 0.05, True, True],
    [appr_str, 0.002, 0.05, 7, False],
    # [appr_str,0.01,0.05,7,False],
    # [exact_str,0.01,0.05,True,True],
]

# plt.figure(figsize=(10,6))
cm = 1 / 2.54
fig1, axs = plt.subplots(1, 1, figsize=(8.60 * cm, 8.75 * cm))
font = {"family": "Arial", "weight": "normal", "size": 9}
plt.rc("font", **font)
COLOR = "k"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR
plt.rcParams["xtick.color"] = COLOR
plt.rcParams["ytick.color"] = COLOR
plt.rc("axes", edgecolor=COLOR)

emarker = "-"
amarker = "-"
aalpha = 1
for type_str, m_time, decay_time, do_refresh, do_dephasing in flist:
    if type_str == appr_str:
        marker = "-"
    else:
        marker = "-"
    fname = f"sequence_data/{type_str}_cycles_ind_{m_time}_{decay_time}_{do_refresh}_{do_dephasing}.txt"
    dat = np.loadtxt(fname, dtype=np.complex_)
    time_list = dat[0, :]
    op_log = dat[1, :]
    op_log_uncorr = dat[2, :]
    fname0 = f"sequence_data_0/{type_str}_cycles_ind_{m_time}_{decay_time}_{do_refresh}_{do_dephasing}.txt"
    dat0 = np.loadtxt(fname0, dtype=np.complex_)
    time_list0 = dat0[0, :]
    op_log0 = dat0[1, :]

    if type_str == "exact":
        color = color_list[0]
        marker = emarker
        # emarker += '-'
    else:
        color = color_list[1]
        marker = amarker
        # amarker += '-'
    axs.plot(
        time_list, op_log, marker + "-", color=color, label=f"$\\rho_+$ {type_str}"
    )
    axs.plot(time_list0, op_log0, marker, color=color, label=f"$\\rho_0$ {type_str} ")
    axs.fill_between(
        time_list, op_log, op_log0, color=color, alpha=0.2
    )  # logical fidelity range for full DEC
axs.plot(time_list, op_log_uncorr, "-", color="#a43530", label=f"do nothing")

axs.minorticks_on()

# Put a legend below current axis
lines, labels = axs.get_legend_handles_labels()
fig1.legend(
    lines, labels, loc="lower center", fancybox=True, shadow=False, frameon=1, ncol=2
)
axs.set_xlabel("Time ($\\Gamma_{\\text{C}}^{-1}$)")
axs.set_ylabel("Logical fidelity of $|\\overline{+}\\rangle$")
axs.set_xlim([0, 2])
# FINALIZE FIGURE AND SAVE
fig1.patch.set_alpha(1)
fig1.tight_layout()
# fig1.subplots_adjust(bottom=0.25)  # create space for legend
fig1.subplots_adjust(bottom=0.40)  # create space for legend

fig1.savefig(
    "images/paper/log_fid_sequence.pdf", facecolor=fig1.get_facecolor(), dpi=300
)

plt.show()
