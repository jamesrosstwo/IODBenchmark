from typing import List

import numpy as np

from src.data.dataset import IODDataset
from src.metrics.final.base import FinalMetric


class ContinualAverage(FinalMetric[float]):
    def result(self, dataset: IODDataset, label_last_seen_hist: np.array, label_occurrences: np.array, **metric_results):
        out_res = []
        for metric_idx, res in enumerate(metric_results.values()):
            metric_res = []
            class_occurrences = []
            for cls_idx, cls in enumerate(res):
                f = 0
                n = 0
                for i in range(len(cls)):
                    for j in range(i + 1):
                        if np.isnan(cls[i][j]):
                            continue
                        f += cls[i][j]
                        n += 1

                if n == 0:
                    metric_res.append(np.nan)
                else:
                    metric_res.append(f / n)
                class_occurrences.append(n)

            total = 0
            for m, c in zip(metric_res, class_occurrences):
                if c == 0 or np.isnan(m):
                    continue
                total += m * c

            out_res.append((metric_res, total / sum(class_occurrences)))

        return out_res, np.nanmean(out_res)
