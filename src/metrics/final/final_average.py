import numpy as np

from src.data.dataset import IODDataset
from src.metrics.final.base import FinalMetric


class FinalAverage(FinalMetric[float]):
    def result(self, dataset: IODDataset, label_last_seen_hist: np.array, label_occurrences: np.array, **metric_results):
        out_res = []
        split_weights = [len(s) for s in dataset.test_splits]
        for metric_idx, res in enumerate(metric_results.values()):
            final_res = res[:, -1, :]
            masked_final_res = np.ma.masked_array(final_res, np.isnan(final_res))
            average = np.ma.average(masked_final_res, axis=1, weights=split_weights)

            out_res.append(average.filled(np.nan))

        out_res = np.array(out_res)
        return out_res, np.nanmean(out_res)
