import numpy as np

from src.data.dataset import IODDataset
from src.metrics.final.base import FinalMetric


class Forgetting(FinalMetric[float]):
    def result(self, dataset: IODDataset, label_last_seen_hist: np.array, label_occurrences: np.array, **metric_results):
        forgetting = []

        # Represents how much the model has forgotten about task j after being trained on task i [Wang et al. 2021]
        # Within this metric, we calculate the forgetting by calculating the change in metrics from when last trained
        # on data from task j.
        # We do not consider forwards transfer, and as a result stop our iteration early.
        # Lower tracked metric is better: higher forgetting is better / negative means we are forgetting
        # Higher tracked metric is better: lower forgetting is better / positive means we are forgetting
        label_occs = np.array(label_last_seen_hist)

        for metric_idx, res in enumerate(metric_results.values()):
            for cls_idx, cls in enumerate(res):
                split_order = np.argsort(label_occs[:, cls_idx])
                k_min = label_occs[split_order[0], cls_idx]

                f_c = 0
                # The mean of this continual metric on this class for the split most recently containing the class
                acap_kmin = np.nanmean(res[cls_idx, split_order[0], :])
                for split_idx in split_order:
                    k = label_occs[split_idx, cls_idx]
                    # The mean of this continual metric across all eval splits, measured on this class
                    acap_k = np.nanmean(res[cls_idx, split_idx, :])

                    num = k - k_min
                    denom = sum([label_occs[j, cls_idx] - k_min for j in split_order])
                    f_c += num / denom * (acap_kmin - acap_k)
                forgetting.append(f_c)

        return forgetting, np.nanmean(forgetting)
