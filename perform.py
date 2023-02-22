if __name__ == "__main__":
    from typing import *
    import pickle

    from numpy import ndarray
    from QGrain.statistics import all_statistics
    from QGrain.models import DistributionType, KernelType, ArtificialDataset, SSUResult, EMMAResult, UDMResult
    from QGrain.ssu import try_dataset
    from QGrain.emma import try_emma
    from QGrain.udm import try_udm
    from QGrain.io import *
    from sklearn.decomposition import PCA
    from multiprocessing import freeze_support

    freeze_support()

    with open("./dump/dataset.dump", "rb") as f:
        dataset: ArtificialDataset = pickle.load(f)
    x0 = np.mean(dataset.parameters, axis=0)

    options = dict(x0=x0.astype(np.float64), loss="lmse", optimizer="L-BFGS-B", try_global=False, need_history=True)
    ssu_results, failed_tasks = try_dataset(dataset, DistributionType.Normal, 4, n_processes=8, options=options)
    assert len(ssu_results) == dataset.n_samples
    assert len(failed_tasks) == 0

    all_sample_statistics = []
    for sample in dataset:
        all_sample_statistics.append(all_statistics(sample.classes, sample.classes_phi, sample.distribution))

    pca = PCA(n_components=10)
    pc_values = pca.fit_transform(dataset.distributions)
    pcs = pca.components_
    pc_ratios = pca.explained_variance_ratio_

    emma_result = try_emma(dataset, KernelType.Normal, 4, x0=x0[:-1, :].astype(np.float32), device="cuda", loss="lmse",
                           pretrain_epochs=400, min_epochs=2000, max_epochs=3000, precision=10.0, learning_rate=5e-3,
                           betas=(0.4, 0.1), update_end_members=True, need_history=True)

    udm_result = try_udm(dataset, KernelType.Normal, 4, x0=x0[:-1, :].astype(np.float32), device="cuda",
                         pretrain_epochs=400, min_epochs=2000, max_epochs=3000, precision=10.0, learning_rate=5e-3,
                         betas=(0.4, 0.1), constraint_level=2.0, need_history=True)

    check_point = {
        "dataset": dataset,
        "all_statistics": all_sample_statistics,
        "pc_values": pc_values,
        "pcs": pcs,
        "pc_ratios": pc_ratios,
        "emma_result": emma_result,
        "udm_result": udm_result,
        "ssu_results": ssu_results,
        "ages": np.linspace(4.5, 144.0, 300)}

    save_artificial_dataset(dataset, "./sheets/Artificial Dataset.xlsx")
    save_pca(dataset.dataset, "./sheets/PCA.xlsx")
    save_statistics(dataset.dataset, "./sheets/Statistics.xlsx")
    save_emma(emma_result, "./sheets/EMMA.xlsx")
    save_ssu(ssu_results, "./sheets/SSU.xlsx")
    save_udm(udm_result, "./sheets/UDM.xlsx")

    with open("./dump/all_results.dump", "wb") as f:
        pickle.dump(check_point, f)
    with open("./dump/result.emma", "wb") as f:
        pickle.dump(emma_result, f)
    with open("./dump/result.udm", "wb") as f:
        pickle.dump(udm_result, f)
    with open("./dump/result.ssu", "wb") as f:
        pickle.dump(ssu_results, f)
