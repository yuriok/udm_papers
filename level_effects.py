import os
import pickle
import string
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
from QGrain.models import ArtificialDataset, UDMResult, KernelType
from QGrain.metrics import loss_numpy
from QGrain.udm import try_udm
from scipy.stats import pearsonr


os.makedirs("./dump/level_effects", exist_ok=True)
with open("./dump/dataset.dump", "rb") as f:
    dataset: ArtificialDataset = pickle.load(f)
levels = np.linspace(-1.0, 5.0, 301)
parameters = np.mean(dataset.parameters, axis=0)[:-1, :].astype(np.float32)
signal_df = pd.read_excel("./sheets/Series.xlsx", sheet_name=0, header=0, index_col=0)
series_1 = signal_df["Series 1"].to_numpy()
series_2 = signal_df["Series 2"].to_numpy()
series_3 = signal_df["Series 3"].to_numpy()

perform = False
if perform:
    results = []
    for i, level in enumerate(levels):
        result = try_udm(dataset, KernelType.Normal, 4, x0=parameters, device="cuda", pretrain_epochs=400,
                         min_epochs=2000, max_epochs=3000, precision=10, learning_rate=5e-3, betas=(0.4, 0.1),
                         constraint_level=level, need_history=False)
        with open(f"./dump/level_effects/{i:03} (level={level:.2f}).udm", "wb") as f:
            pickle.dump(result, f)
        print(f"The task (level={level:.2f}) was finished.")
        results.append(result)
else:
    results: List[UDMResult] = []
    for filename in sorted(os.listdir("./dump/level_effects")):
        with open(os.path.join("./dump/level_effects", filename), "rb") as f:
            result = pickle.load(f)
            results.append(result)


plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["mathtext.fontset"] = "dejavusans"

plt.figure(figsize=(4.4, 5.8))
cmap = plt.get_cmap("tab10")

plt.subplot(3, 2, 1)
for sample in dataset[::5]:
    plt.plot(sample.classes, sample.distribution * 100, color="gray", linewidth=0.2)
plt.xscale("log")
plt.xlim(dataset.classes[0], dataset.classes[-1])
plt.ylim(0.0, 10)
plt.xticks([1e-1, 1e0, 1e1, 1e2, 1e3], ["0.1", "1", "10", "100", "1000"])
plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"])
plt.xlabel("Grain size ($\mu m$)")
plt.ylabel("Frequency ($\%$)")
plt.title("GSDs")

plt.subplot(3, 2, 2)
loss_function = loss_numpy("lmse")
losses = [result.loss("lmse") for result in results]
plt.plot(levels, losses, color="black", label="Overall")
for i in range(dataset.n_components):
    component_distances = [loss_function(result.components[:, i, :], dataset.components[:, i, :], None)
                           for result in results]
    plt.plot(levels, component_distances, color=cmap(i), label=f"C{i+1}")
plt.xlim(0, 4)
plt.xlabel("Constraint level")
plt.ylabel("LMSE")
plt.title("Distribution")
plt.legend(loc="upper left", prop={"size": 6})

# for i, title in enumerate(("Mean", "Sorting coefficient", "Abundance"))
plt.subplot(3, 2, 3)
for i in range(dataset.n_components):
    losses = [np.mean(np.abs(result.parameters[-1, :, 0, i] - dataset.parameters[:, 0, i])) for result in results]
    plt.plot(levels, losses, color=cmap(i))
plt.yscale("log")
plt.xlim(0, 4)
plt.ylim(0.002, 0.6)
plt.yticks([1e-2, 1e-1], ["0.01", "0.1"])
plt.xlabel("Constraint level")
plt.ylabel("MAE ($\phi$)")
plt.title("Mean")

plt.subplot(3, 2, 4)
for i in range(dataset.n_components):
    losses = [np.mean(np.abs(result.parameters[-1, :, 1, i] - dataset.parameters[:, 1, i])) for result in results]
    plt.plot(levels, losses, color=cmap(i))
plt.yscale("log")
plt.xlim(0, 4)
plt.ylim(0.002, 0.2)
plt.yticks([1e-2, 1e-1], ["0.01", "0.1"])
plt.xlabel("Constraint level")
plt.ylabel("MAE ($\phi$)")
plt.title("Sorting coefficient")

plt.subplot(3, 2, 5)
for i in range(dataset.n_components):
    losses = [np.mean(np.abs(result.proportions[:, 0, i] - dataset.proportions[:, 0, i])) * 100.0 for result in results]
    plt.plot(levels, losses, color=cmap(i))
plt.yscale("log")
plt.xlim(0, 4)
plt.ylim(0.02, 3)
plt.yticks([1e-1, 1], ["0.1", "1"])
plt.xlabel("Constraint level")
plt.ylabel("MAE ($\%$)")
plt.title("Proportion")

plt.subplot(3, 2, 6)


def get_series(index: int, p: ndarray):
    if index == 0:
        return p[-1, :, 0, 2]
    elif index == 1:
        return p[-1, :, 0, 3]
    elif index == 2:
        return p[-1, :, 2, 3]


for i, series in enumerate([series_1, series_2, series_3]):
    r2 = [pearsonr(series, get_series(i, result.parameters))[0]**2 for result in results]
    plt.plot(levels, r2, color=cmap(i), label=f"Signal {i+1}")
plt.xlim(0, 4)
plt.ylim(0.0, 1.05)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"])
plt.xlabel("Constraint level")
plt.ylabel("$R^2$")
plt.legend(loc="lower left", prop={"size": 6})
plt.title("Signal")

plt.tight_layout()
for n, ax in enumerate(plt.gcf().axes):
    ax.text(-0.15, 1.06,
            f"{string.ascii_uppercase[n]}",
            transform=ax.transAxes,
            size=10, weight="bold")

plt.savefig("./figures/Level Effects.png", dpi=1200.0)
plt.savefig("./figures/Level Effects.eps")
# plt.savefig("./figures/Level Effects.tif", dpi=1200.0, pil_kwargs={"compression": "tiff_lzw"})
plt.show()
