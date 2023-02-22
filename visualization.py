import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
from QGrain.statistics import logarithmic
from QGrain.models import ArtificialDataset, SSUResult, EMMAResult, UDMResult
from QGrain.distributions import Normal
from QGrain.utils import get_image_by_proportions
from numpy import ndarray


# a column = 3.14 inchs
# a page = 6.69 inchs

def summarize(components: ndarray, q=0.01):
    mean = np.mean(components, axis=0)
    upper = np.quantile(components, q=1 - q, axis=0)
    lower = np.quantile(components, q=q, axis=0)
    return mean, lower, upper


def plot_summary(axes: plt.Axes, classes: ndarray, components: ndarray, xlabel="Grain size ($\mu m$)",
                 ylabel="Frequency ($\%$)", title=None):
    n_samples, n_components, n_classes = components.shape
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    light_colors = ["#A8D2F0", "#FFC999", "#AFE9AF", "#EFA9AA"]
    mean, lower, upper = summarize(components, q=0.01)
    for i in range(n_components):
        axes.plot(classes, mean[i] * 100, color=colors[i], zorder=-10 + i)
        axes.fill_between(classes, lower[i] * 100, upper[i] * 100, color=light_colors[i], lw=0.02, zorder=-20 + i)
    axes.set_xscale("log")
    axes.set_xlim(0.2, 2000)
    axes.set_ylim(0.0, 12.5)
    axes.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    axes.set_yticks([2, 6, 10], ["2", "6", "10"])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)


def plot_proportions(axes: plt.Axes, proportions: ndarray, xlabel="Sample index", ylabel="Proportion ($\%$)",
                     title=None, cmap="tab10"):
    n_samples, _, _ = proportions.shape
    image = get_image_by_proportions(proportions[:, 0, :], resolution=100)
    axes.imshow(image, cmap=cmap, vmin=0, vmax=9, aspect="auto", extent=(0.0, n_samples, 100.0, 0.0))
    axes.set_xlim(0, n_samples)
    axes.set_ylim(0.0, 100.0)
    axes.set_xticks([i * 100 for i in range(6)])
    axes.set_yticks([20, 40, 60, 80], ["20", "40", "60", "80"])
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)


with open("./dump/all_results.dump", "rb") as f:
    check_point = pickle.load(f)
    dataset = check_point["dataset"]  # type: ArtificialDataset
    all_statistics = check_point["all_statistics"]  # type: list[dict]
    pc_values = check_point["pc_values"]  # type: ndarray
    pcs = check_point["pcs"]  # type: ndarray
    pc_ratios = check_point["pc_ratios"]  # type: list[float]
    emma_result = check_point["emma_result"]  # type: EMMAResult
    udm_result = check_point["udm_result"]  # type: UDMResult
    ssu_results = check_point["ssu_results"]  # type: list[SSUResult]
    ages = check_point["ages"]  # type: ndarray

plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["mathtext.fontset"] = "dejavusans"

classes = np.expand_dims(np.expand_dims(dataset.classes_phi, axis=0), axis=0).repeat(
    dataset.n_samples, axis=0).repeat(dataset.n_components, axis=1)
interval = np.abs((dataset.classes_phi[0] - dataset.classes_phi[-1]) / (dataset.n_classes - 1))
proportions_true, components_true, (m_true, std_true, s_true, k_true) = Normal.interpret(
    dataset.parameters, classes, interval)
proportions_udm, components_udm, (m_udm, std_udm, s_udm, k_udm) = Normal.interpret(
    udm_result.parameters[-1], classes, interval)
proportions_emma = np.expand_dims(emma_result.proportions, axis=1)
components_emma = np.expand_dims(emma_result.end_members, axis=0).repeat(dataset.n_samples, axis=0)
m_emma = np.expand_dims(np.array([logarithmic(dataset.classes_phi, end_member)["mean"] for
                                  end_member in emma_result.end_members]), axis=0).repeat(dataset.n_samples, axis=0)
std_emma = np.expand_dims(np.array([logarithmic(dataset.classes_phi, end_member)["std"] for
                                    end_member in emma_result.end_members]), axis=0).repeat(dataset.n_samples, axis=0)
proportions_ssu = np.array([[[component.proportion for component in result]] for result in ssu_results])
components_ssu = np.array([[component.distribution for component in result] for result in ssu_results])
m_ssu = np.array([[component.moments["mean"] for component in result] for result in ssu_results])
std_ssu = np.array([[component.moments["std"] for component in result] for result in ssu_results])
s_ssu = np.array([[component.moments["skewness"] for component in result] for result in ssu_results])
k_ssu = np.array([[component.moments["kurtosis"] for component in result] for result in ssu_results])

plt.figure(figsize=(6.6, 3.3))
cmap = plt.get_cmap("tab10")
# distributions of components
axes_1 = plt.subplot(2, 4, 1)
plot_summary(axes_1, dataset.classes, components_true, title="True")
axes_2 = plt.subplot(2, 4, 2)
plot_summary(axes_2, dataset.classes, components_emma, title="EMMA")
axes_3 = plt.subplot(2, 4, 3)
plot_summary(axes_3, dataset.classes, components_ssu, title="SSU")
axes_4 = plt.subplot(2, 4, 4)
plot_summary(axes_4, dataset.classes, components_udm, title="UDM")

# proportions of components
axes_5 = plt.subplot(2, 4, 5)
plot_proportions(axes_5, proportions_true, title="True")
axes_6 = plt.subplot(2, 4, 6)
plot_proportions(axes_6, proportions_emma, title="EMMA")
axes_7 = plt.subplot(2, 4, 7)
plot_proportions(axes_7, proportions_ssu, title="SSU")
axes_8 = plt.subplot(2, 4, 8)
plot_proportions(axes_8, proportions_udm, title="UDM")

plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06, f"{string.ascii_uppercase[n]}", transform=ax.transAxes, size=10, weight="bold")
plt.savefig("./figures/Distributions and Proportions.png", dpi=1200.0)
# plt.savefig("./figures/Distributions and Proportions.tif", dpi=600.0, pil_kwargs={"compression": "tiff_lzw"})
plt.savefig("./figures/Distributions and Proportions.eps")

plt.figure(figsize=(6.6, 5.0))
cmap = plt.get_cmap("tab10")
plt.subplot(3, 4, 1)
plt.plot(m_true[:, 0], m_emma[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(m_true[:, 0], m_ssu[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(m_true[:, 0], m_udm[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 10.05, 10.35
ticks = [10.1, 10.2, 10.3]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\phi$)")
plt.ylabel("Estimated ($\phi$)")
plt.title("$Mz_1$")

plt.subplot(3, 4, 2)
plt.plot(m_true[:, 1], m_emma[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(m_true[:, 1], m_ssu[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(m_true[:, 1], m_udm[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 7.75, 8.30
ticks = [7.8, 8.0, 8.2]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\phi$)")
plt.ylabel("Estimated ($\phi$)")
plt.title("$Mz_2$")

plt.subplot(3, 4, 3)
plt.plot(m_true[:, 2], m_emma[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(m_true[:, 2], m_ssu[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(m_true[:, 2], m_udm[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 5.35, 6.65
ticks = [5.4, 5.8, 6.2, 6.6]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\phi$)")
plt.ylabel("Estimated ($\phi$)")
plt.title("$Mz_3$")

plt.subplot(3, 4, 4)
plt.plot(m_true[:, 3], m_emma[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(m_true[:, 3], m_ssu[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(m_true[:, 3], m_udm[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 2.35, 4.65
ticks = [2.5, 3.5, 4.5]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\phi$)")
plt.ylabel("Estimated ($\phi$)")
plt.title("$Mz_4$")

plt.subplot(3, 4, 5)
plt.plot(std_true[:, 0], std_emma[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(std_true[:, 0], std_ssu[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(std_true[:, 0], std_udm[:, 0], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0.525, 0.7
ticks = [0.56, 0.62, 0.68]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True")
plt.ylabel("Estimated")
plt.title("$So_1$")

plt.subplot(3, 4, 6)
plt.plot(std_true[:, 1], std_emma[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(std_true[:, 1], std_ssu[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(std_true[:, 1], std_udm[:, 1], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0.675, 0.875
ticks = [0.68, 0.76, 0.84]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True")
plt.ylabel("Estimated")
plt.title("$So_2$")

plt.subplot(3, 4, 7)
plt.plot(std_true[:, 2], std_emma[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(std_true[:, 2], std_ssu[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(std_true[:, 2], std_udm[:, 2], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0.695, 0.745
ticks = [0.7, 0.72, 0.74]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True")
plt.ylabel("Estimated")
plt.title("$So_3$")

plt.subplot(3, 4, 8)
plt.plot(std_true[:, 3], std_emma[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(std_true[:, 3], std_ssu[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(std_true[:, 3], std_udm[:, 3], ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0.585, 0.715
ticks = [0.6, 0.65, 0.7]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True")
plt.ylabel("Estimated")
plt.title("$So_4$")

plt.subplot(3, 4, 9)
plt.plot(proportions_true[:, 0, 0] * 100, proportions_emma[:, 0, 0] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(proportions_true[:, 0, 0] * 100, proportions_ssu[:, 0, 0] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(proportions_true[:, 0, 0] * 100, proportions_udm[:, 0, 0] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0, 16
ticks = [5, 10, 15]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\%$)")
plt.ylabel("Estimated ($\%$)")
plt.title("$p_1$")

plt.subplot(3, 4, 10)
plt.plot(proportions_true[:, 0, 1] * 100, proportions_emma[:, 0, 1] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(proportions_true[:, 0, 1] * 100, proportions_ssu[:, 0, 1] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(proportions_true[:, 0, 1] * 100, proportions_udm[:, 0, 1] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0, 38
ticks = [10, 20, 30]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\%$)")
plt.ylabel("Estimated ($\%$)")
plt.title("$p_2$")

plt.subplot(3, 4, 11)
plt.plot(proportions_true[:, 0, 2] * 100, proportions_emma[:, 0, 2] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(proportions_true[:, 0, 2] * 100, proportions_ssu[:, 0, 2] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(proportions_true[:, 0, 2] * 100, proportions_udm[:, 0, 2] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 5.5, 77.5
ticks = [20, 40, 60]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\%$)")
plt.ylabel("Estimated ($\%$)")
plt.title("$p_3$")

plt.subplot(3, 4, 12)
plt.plot(proportions_true[:, 0, 3] * 100, proportions_emma[:, 0, 3] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(0), marker=".")
plt.plot(proportions_true[:, 0, 3] * 100, proportions_ssu[:, 0, 3] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(1), marker=".")
plt.plot(proportions_true[:, 0, 3] * 100, proportions_udm[:, 0, 3] * 100,
         ms=4.0, mew=0.0, linewidth=0.0, mfc=cmap(2), marker=".")
plt.plot([-1e6, 1e6], [-1e6, 1e6], c="gray", ls="--")
ranges = 0.0, 85
ticks = [20, 40, 60, 80]
plt.xlim(ranges)
plt.ylim(ranges)
plt.xticks(ticks)
plt.yticks(ticks)
plt.gca().set_aspect(1.0)
plt.xlabel("True ($\%$)")
plt.ylabel("Estimated ($\%$)")
plt.title("$p_4$")

plt.tight_layout()

for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")
plt.savefig("./figures/True - Estimated.png", dpi=1200.0)
plt.savefig("./figures/True - Estimated.eps")

plt.show()
