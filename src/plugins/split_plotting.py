import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import jensenshannon
import torch.distributed as dist

from data.dataset import IODDataset, IODSplit
from experiments.experiment import IODExperiment
from plugins.base import IODPlugin


def get_label_occurrence(split: IODSplit, num_classes: int):
    all_labels = []
    for _, t in split._datapoints:
        all_labels += t["labels"].tolist()

    if len(all_labels) == 0:
        return np.zeros(num_classes), 1

    img_class_counts = np.bincount(np.array(all_labels).astype(np.int64), minlength=num_classes).astype(np.float64)
    return img_class_counts, np.sum(img_class_counts)


def _get_all_labels(dataset: IODDataset, split: IODSplit):
    all_labels = []
    for t in split.targets:
        all_labels += t["labels"].tolist()

    img_class_counts = np.bincount(np.array(all_labels), minlength=dataset.num_classes).astype(np.float64)
    return img_class_counts, np.sum(img_class_counts)


class SplitPlotting(IODPlugin):
    def __init__(self, name, exp: IODExperiment, show_plots=False):
        super().__init__(name, exp)
        self._out_dir = None
        self._show_plots = show_plots

    def _write_heatmaps(self, dataset):
        x_axis_cfg = dict(
            tickmode='array',
            tickvals=list(range(dataset.num_classes)),
            ticktext=list(dataset.class_names.values()),
        )

        train_dists = np.zeros((2, dataset.n_splits, dataset.num_classes))
        eval_dists = np.zeros((2, dataset.n_splits, dataset.num_classes))
        agg_dists = np.zeros((2, dataset.n_splits, dataset.num_classes))

        train_class_sizes = []
        train_aspect_ratios = []
        bins = []

        for split_idx, (train_split, test_split) in enumerate(zip(dataset.train_splits, dataset.test_splits)):
            train_hist, train_div = _get_all_labels(dataset, train_split)
            eval_hist, eval_div = _get_all_labels(dataset, test_split)
            agg_hist = train_hist + eval_hist
            agg_div = train_div + eval_div

            train_dists[0, split_idx, :] = train_hist
            eval_dists[0, split_idx, :] = eval_hist
            agg_dists[0, split_idx, :] = agg_hist

            train_dists[1, split_idx, :] = train_hist / train_div
            eval_dists[1, split_idx, :] = eval_hist / eval_div
            agg_dists[1, split_idx, :] = agg_hist / agg_div

            class_sizes = [[] for _ in range(dataset.num_classes)]
            aspect_ratios = [[] for _ in range(dataset.num_classes)]

            for t in train_split.targets:
                for box, label in zip(t["boxes"], t["labels"]):
                    w = box[2] - box[0]
                    h = box[3] - box[1]

                    class_sizes[label].append(int(np.sqrt(w * h).item()))

                    if h == 0 or w == 0:
                        continue
                    ratio = (w / h).item().as_integer_ratio()
                    reduced_ratio = max(ratio) // min(ratio)
                    if ratio[1] > ratio[0]:
                        reduced_ratio = -reduced_ratio
                    # Large negative values: Tall and narrow
                    # Large positive values: Short and wide
                    aspect_ratios[label].append(reduced_ratio)

            bound = int(10e10)
            min_sz = min([min(x) if len(x) > 0 else -bound for x in class_sizes])
            max_sz = max([max(x) if len(x) > 0 else bound for x in class_sizes])
            min_ar = min([min(x) if len(x) > 0 else -bound for x in aspect_ratios])
            max_ar = max([max(x) if len(x) > 0 else bound for x in aspect_ratios])
            sz_bins = list(np.arange(min_sz, max_sz, (max_sz - min_sz) / 10)) + [max_sz]
            ar_bins = list(np.arange(min_ar, max_ar, (max_ar - min_ar) / 10)) + [max_ar]
            train_class_sizes.append((np.stack([np.histogram(x, bins=sz_bins)[0] for x in class_sizes]).T, sz_bins))
            train_aspect_ratios.append((np.stack([np.histogram(x, bins=ar_bins)[0] for x in aspect_ratios]).T, ar_bins))

        all_data = {'train_raw': train_dists[0], 'train_norm': train_dists[1], 'eval_raw': eval_dists[0],
                    'eval_norm': eval_dists[1], 'agg_raw': agg_dists[0], 'agg_norm': agg_dists[1]}

        train_js_dist_mx = np.zeros((dataset.n_splits, dataset.n_splits))
        for i, norm_dist in enumerate(all_data["train_norm"]):
            for j, match_dist in enumerate(all_data["train_norm"]):
                train_js_dist_mx[i, j] = jensenshannon(norm_dist, match_dist)

        train_js_fig = px.imshow(train_js_dist_mx, text_auto=True)

        train_js_fig.update_layout(  # customize font and legend orientation & position
            title="Jensen-Shannon Distance of Normalized Class Distributions Between Train Splits",
            xaxis_title="Split Index",
            yaxis_title="Split Index",
        )

        train_js_fig.write_html(self._out_dir / "train_js_mx.html")
        train_js_fig.write_image(self._out_dir / "train_js_mx.png")

        eval_js_dist_mx = np.zeros((dataset.n_splits, dataset.n_splits))
        for i, norm_dist in enumerate(all_data["train_norm"]):
            for j, match_dist in enumerate(all_data["train_norm"]):
                eval_js_dist_mx[i, j] = jensenshannon(norm_dist, match_dist)

        eval_js_fig = px.imshow(eval_js_dist_mx, text_auto=True)

        eval_js_fig.update_layout(  # customize font and legend orientation & position
            title="Jensen-Shannon Distance of Normalized Class Distributions Between Eval Splits",
            xaxis_title="Split Index",
            yaxis_title="Split Index",
        )

        eval_js_fig.write_html(self._out_dir / "eval_kl_mx.html")
        eval_js_fig.write_image(self._out_dir / "eval_kl_mx.png")

        all_js_dist_mx = np.zeros((dataset.n_splits, dataset.n_splits))
        for i, norm_dist in enumerate(all_data["agg_norm"]):
            for j, match_dist in enumerate(all_data["agg_norm"]):
                all_js_dist_mx[i, j] = jensenshannon(norm_dist, match_dist)

        all_kl_fig = px.imshow(all_js_dist_mx, text_auto=True)

        all_kl_fig.update_layout(  # customize font and legend orientation & position
            title="Jensen-Shannon Distance of Normalized Class Distributions Between All Splits",
            xaxis_title="Split Index",
            yaxis_title="Split Index",
        )
        all_kl_fig.write_html(self._out_dir / "all_kl_mx.html")
        all_kl_fig.write_image(self._out_dir / "all_kl_mx.png")

        for name, data in all_data.items():
            name = dataset.name + "_" + name
            data = np.log(np.array(data))
            fig = px.imshow(data, text_auto=True)

            fig.update_layout(  # customize font and legend orientation & position
                title="Log Class Distributions for " + name,
                xaxis_title="Class Index",
                yaxis_title="Split Index",
                xaxis=x_axis_cfg
            )
            if self._show_plots:
                fig.show()
            fig.write_html(self._out_dir / (name + ".html"))
            fig.write_image(self._out_dir / (name + ".png"))

        for index, ((class_sizes, size_bins), (aspect_ratios, aspect_bins)) in enumerate(
                zip(train_class_sizes, train_aspect_ratios)):
            class_size_fig = px.imshow(class_sizes, text_auto=True)
            sz_yaxis = dict(
                tickmode='array',
                tickvals=[b - 0.5 for b in range(len(size_bins))],
                ticktext=[str(b)[:5] for b in size_bins],
            )
            class_size_fig.update_layout(  # customize font and legend orientation & position
                title="Box sizes of instances in split {0}".format(index),
                xaxis_title="Class Index",
                yaxis_title="Size Bin",
                xaxis=x_axis_cfg,
                yaxis=sz_yaxis
            )

            class_sz_name = "box_sizes_split{0}".format(index)
            if self._show_plots:
                class_size_fig.show()
            class_size_fig.write_html(self._out_dir / (class_sz_name + ".html"))
            class_size_fig.write_image(self._out_dir / (class_sz_name + ".png"))

            ar_yaxis = dict(
                tickmode='array',
                tickvals=[b - 0.5 for b in range(len(aspect_bins))],
                ticktext=[str(b)[:5] for b in aspect_bins],
            )
            aspect_ratio_fig = px.imshow(aspect_ratios, text_auto=True)
            aspect_ratio_fig.update_layout(  # customize font and legend orientation & position
                title="Aspect Ratios of instances in split {0}".format(index),
                xaxis_title="Class Index",
                yaxis_title="Aspect Ratio Bin",
                xaxis=x_axis_cfg,
                yaxis=ar_yaxis
            )

            aspect_ratio_name = "ratios_split{0}".format(index)
            if self._show_plots:
                aspect_ratio_fig.show()
            aspect_ratio_fig.write_html(self._out_dir / (aspect_ratio_name + ".html"))
            aspect_ratio_fig.write_image(self._out_dir / (aspect_ratio_name + ".png"))

    def _write_seen_classes(self, dataset):
        history = []
        seen_train_classes = [0 for _ in range(dataset.num_classes)]

        mod = 5
        for split in dataset.train_splits:
            for idx, t in enumerate(split.targets):
                for label in t["labels"].tolist():
                    seen_train_classes[label] += 1
                if idx % mod == 0:
                    history.append(seen_train_classes.copy())

        n_classes_seen = [sum(map(lambda x: x > 0, d)) for d in history]
        classes_seen_df = pd.DataFrame({"n_class": n_classes_seen})
        classes_seen_fig = px.line(classes_seen_df, y="n_class")
        classes_seen_fig.update_layout(title='Number of Classes Seen Over Time',
                                       xaxis_title='Time step',
                                       yaxis_title='Number of Classes Seen')

        if self._show_plots:
            classes_seen_fig.show()
        classes_seen_fig.write_html(self._out_dir / "num_classes_seen.html")
        classes_seen_fig.write_image(self._out_dir / "num_classes_seen.png")

        # Now we are indexing by class rather than by index in our dataset
        history = np.array(history).T[1:, :]

        class_instances_fig = go.Figure()
        for c, x in enumerate(history):
            class_instances_fig.add_trace(
                go.Scatter(x=list(range(0, len(x) * mod, mod)), y=x, name=dataset.class_names[c + 1]))

        class_instances_fig.update_layout(title='Number of class instance occurrences over time',
                                          xaxis_title='Time step',
                                          yaxis_title='Number of Occurrences')
        if self._show_plots:
            class_instances_fig.show()
        class_instances_fig.write_html(self._out_dir / "class_instances.html")
        class_instances_fig.write_image(self._out_dir / "class_instances.png")

    def before_dataset(self, dataset: IODDataset):
        self._out_dir = self._exp.out_path / "SplitPlotting"
        self._out_dir.mkdir(exist_ok=True)
        self._out_dir /= dataset.name
        self._out_dir.mkdir(exist_ok=dist.get_world_size() > 1)

        self._write_heatmaps(dataset)
        self._write_seen_classes(dataset)
