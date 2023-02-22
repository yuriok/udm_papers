import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from QGrain.models import KernelType, ArtificialDataset, EMMAResult
from QGrain.emma import try_emma
from QGrain.io import *

datasets = ["D1", "D2"]
labels = ["Ideal", "Actual"]
colors = ["#126bae", "#a61b29"]

analysis_required = False
for name in datasets:
    if not analysis_required:
        break
    dataset: ArtificialDataset = load_dataset(f"./sheets/{name}.xlsx", name, sheet_index=1)
    os.makedirs(f"./dump/{name}", exist_ok=True)

    for i in range(1, 11):
        emma_result = try_emma(dataset, KernelType.Normal, i, device="cpu", loss="lmse",
                               pretrain_epochs=0, min_epochs=2000, max_epochs=10000,
                               precision=20.0, learning_rate=5e-3, betas=(0.8, 0.5),
                               need_history=False)
        with open(f"./dump/{name}/result_{i}.emma", "wb") as f:
            pickle.dump(emma_result, f)

plt.style.use(["science", "no-latex"])
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["mathtext.fontset"] = "dejavusans"

plt.figure(figsize=(3.14, 2.5))



for name, label, color in zip(datasets, labels, colors):
    mse_list = []
    for i in range(1, 11):
        with open(f"./dump/{name}/result_{i}.emma", "rb") as f:
            emma_result: EMMAResult = pickle.load(f)
        mse = emma_result.loss("angular")
        mse_list.append(mse)
    plt.plot(range(1, 11), mse_list, label=label, color=color, marker=".", mfc=color, mew=0.0, ms=8.0, zorder=10)

plt.vlines(4, 0.0, 1.0, colors=["#c0c0c0"], linewidth=10.0, zorder=-1)
plt.vlines(4, 0.0, 1.0, colors=["#e0e0e0"], linewidth=50.0, zorder=-2)
plt.xlim(0, 11)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.tick_params(axis="x", which="minor", top=False, bottom=False)
plt.ylim(0, 0.22)
plt.yticks([0.0, 0.04, 0.08, 0.12, 0.16, 0.2])
plt.xlabel("Number of end members")
plt.ylabel("Angular deviation ($\degree$)")
plt.legend(loc="upper right", prop={"size": 8})

plt.tight_layout()

plt.savefig("./figures/Ideal vs Actual.png", dpi=1200.0)
plt.savefig("./figures/Ideal vs Actual.eps")
plt.show()
