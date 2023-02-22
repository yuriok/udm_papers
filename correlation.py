import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from QGrain.statistics import logarithmic
from QGrain.models import ArtificialDataset, SSUResult, EMMAResult, UDMResult
from QGrain.distributions import Normal
from numpy import ndarray
from scipy.stats import pearsonr

df = pd.read_excel("./sheets/Series.xlsx", sheet_name=0, index_col=0, header=0)
series_1 = df.iloc[:, 1].to_numpy()
series_2 = df.iloc[:, 2].to_numpy()
series_3 = df.iloc[:, 3].to_numpy()

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

names_and_series = [
    ("Signal 1", series_1),
    ("Signal 2", series_2),
    ("Signal 3", series_3),
    ("Mz", np.array([statistics["logarithmic"]["mean"]
     for statistics in all_statistics])),
    ("So", np.array([statistics["logarithmic"]["std"]
     for statistics in all_statistics])),
    ("Skewness", np.array([statistics["logarithmic"]
     ["skewness"] for statistics in all_statistics])),
    ("Kurtosis", np.array([statistics["logarithmic"]["kurtosis"] for statistics in all_statistics]))]

names = ["<5 μm", "5-16 μm", "16-32 μm", "32-63 μm", ">63 μm"]
keys = [np.less(dataset.classes, 5),
        np.all([np.greater_equal(dataset.classes, 5),
               np.less(dataset.classes, 16)], axis=0),
        np.all([np.greater_equal(dataset.classes, 16),
               np.less(dataset.classes, 32)], axis=0),
        np.all([np.greater_equal(dataset.classes, 32),
               np.less(dataset.classes, 63)], axis=0),
        np.greater(dataset.classes, 63)]
for name, key in zip(names, keys):
    series = []
    for sample in dataset:
        value = np.sum(sample.distribution[key])
        series.append(value)
    names_and_series.append((name, np.array(series)))

for i in range(10):
    # name = f"PC{i+1} ({pc_ratios[i]:.2%})"
    name = f"PC{i + 1}"
    names_and_series.append((name, pc_values[:, i]))

classes = np.expand_dims(np.expand_dims(dataset.classes_phi, axis=0), axis=0).repeat(
    dataset.n_samples, axis=0).repeat(dataset.n_components, axis=1)
interval = np.abs(
    (dataset.classes_phi[0] - dataset.classes_phi[-1]) / (dataset.n_classes - 1))
proportions_true, components_true, (m_true, std_true, s_true, k_true) = Normal.interpret(
    dataset.parameters, classes, interval)
proportions_udm, components_udm, (m_udm, std_udm, s_udm, k_udm) = Normal.interpret(
    udm_result.parameters[-1], classes, interval)
proportions_emma = np.expand_dims(emma_result.proportions, axis=1)
components_emma = np.expand_dims(
    emma_result.end_members, axis=0).repeat(dataset.n_samples, axis=0)
m_emma = np.expand_dims(np.array([logarithmic(dataset.classes_phi, end_member)["mean"] for end_member
                                  in emma_result.end_members]), axis=0).repeat(dataset.n_samples, axis=0)
std_emma = np.expand_dims(np.array([logarithmic(dataset.classes_phi, end_member)["std"] for end_member
                                    in emma_result.end_members]), axis=0).repeat(dataset.n_samples, axis=0)
proportions_ssu = np.array(
    [[[component.proportion for component in result]] for result in ssu_results])
components_ssu = np.array(
    [[component.distribution for component in result] for result in ssu_results])
m_ssu = np.array([[component.moments["mean"]
                 for component in result] for result in ssu_results])
std_ssu = np.array([[component.moments["std"]
                   for component in result] for result in ssu_results])
s_ssu = np.array([[component.moments["skewness"]
                 for component in result] for result in ssu_results])
k_ssu = np.array([[component.moments["kurtosis"]
                 for component in result] for result in ssu_results])

for i in range(dataset.n_components):
    name = f"C{i + 1} - True"
    names_and_series.append((name, proportions_true[:, 0, i]))
for i in range(emma_result.n_members):
    name = f"EM{i + 1}"
    names_and_series.append((name, proportions_emma[:, 0, i]))
for i in range(udm_result.n_components):
    name = f"C{i + 1} - SSU"
    names_and_series.append((name, proportions_ssu[:, 0, i]))
for i in range(udm_result.n_components):
    name = f"C{i + 1} - UDM"
    names_and_series.append((name, proportions_udm[:, 0, i]))

for i in range(dataset.n_components):
    name = f"Mz{i + 1} - True"
    names_and_series.append((name, m_true[:, i]))
for i in range(udm_result.n_components):
    name = f"Mz{i + 1} - SSU"
    names_and_series.append((name, m_ssu[:, i]))
for i in range(udm_result.n_components):
    name = f"Mz{i + 1} - UDM"
    names_and_series.append((name, m_udm[:, i]))

for i in range(dataset.n_components):
    name = f"So{i + 1} - True"
    names_and_series.append((name, std_true[:, i]))
for i in range(udm_result.n_components):
    name = f"So{i + 1} - SSU"
    names_and_series.append((name, std_ssu[:, i]))
for i in range(udm_result.n_components):
    name = f"So{i + 1} - UDM"
    names_and_series.append((name, std_udm[:, i]))

names = [name for name, series in names_and_series]
all_series = [series for name, series in names_and_series]
n_series = len(names_and_series)
r_matrix = np.zeros((n_series, n_series))
p_matrix = np.zeros((n_series, n_series))
for i in range(n_series):
    for j in range(n_series):
        r, p = pearsonr(names_and_series[i][1], names_and_series[j][1])
        r_matrix[i, j] = r
        p_matrix[i, j] = p
writer = pd.ExcelWriter("./sheets/Correlation_New.xlsx")
df = pd.DataFrame(r_matrix, index=names, columns=names)
df.to_excel(writer, sheet_name="R Matrix")
df = pd.DataFrame(p_matrix, index=names, columns=names)
df.to_excel(writer, sheet_name="p Matrix")
df = pd.DataFrame(np.array(all_series).T, columns=names)
df.to_excel(writer, sheet_name="Series")
writer.save()

brief_indexes = [0, 1, 2, 3, 9, 11, 12, 13, 26,
                 27, 28, 29, 34, 35, 36, 37, 46, 47, 48, 49]
brief_names = []
for index in brief_indexes:
    name: str = names_and_series[index][0]
    if name.endswith(" - UDM"):
        name = name[:-6]
    if name[-1] in ("1", "2", "3", "4"):
        name = f"{name[:-1]}$_{name[-1]}$"
    if name == "Mz":
        name = "Mz"
    if name[-2:] == "μm":
        name = f"{name[:-2]} $\mu m$"
    brief_names.append(name)
brief_series = [names_and_series[index][1] for index in brief_indexes]

r_matrix = np.full((len(brief_indexes), len(brief_indexes)), fill_value=np.nan)
p_matrix = np.full((len(brief_indexes), len(brief_indexes)), fill_value=np.nan)
for i in range(len(brief_indexes)):
    for j in range(len(brief_indexes)):
        r, p = pearsonr(brief_series[i], brief_series[j])
        r_matrix[i, j] = r
        p_matrix[i, j] = p
mask = np.ones_like(r_matrix)
mask[np.tril_indices_from(mask)] = False

plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["mathtext.fontset"] = "dejavusans"

plt.figure(figsize=(6.6, 5))
plt.gca().set_aspect(1.0)
plt.gca().tick_params(axis="x", which="both", top=False, bottom=False)
plt.gca().tick_params(axis="y", which="both", left=False, right=False)
axes = seaborn.heatmap(
    r_matrix,
    vmin=-1.0, vmax=1.0,
    cmap="coolwarm",
    mask=mask,
    xticklabels=brief_names,
    yticklabels=brief_names)
cbar = axes.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
x_offset = 0.5
y_offset = 0.5
for i in range(len(brief_indexes)):
    for j in range(len(brief_indexes)):
        p = p_matrix[i, j]
        r = abs(r_matrix[i, j])
        if not mask[i, j]:
            continue
        if p < 0.01:
            plt.text(i + x_offset, j + y_offset, f"{r:.2f}",
                     horizontalalignment="center", verticalalignment="center",
                     fontdict={"fontsize": 6})
        elif p < 0.05:
            pass
plt.tight_layout()

plt.savefig("./figures/Correlation.png", dpi=1200.0)
plt.savefig("./figures/Correlation.eps")
plt.show()
