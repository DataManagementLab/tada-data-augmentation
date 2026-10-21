from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
rc("font", **{"family": "serif"})

SPLITS = [0.01, 0.05, 0.1, 0.2, 0.5]
color_dict = {"ctgan": 0, "tvae": 1, "mixup": 2, "copula_gan": 3, "ctabganplus": 4, "great": 5, "realtabformer": 6, "tabpfn": 7, "topo": 8, 'Original data':9}

def read_results_file(paths, time_file=None, plot_data=False, create_table=False):
    dataset_dicts = []
    for path in paths:
        file = open(path, "r")
        content = file.readlines()
        line_skip = 8
        if time_file is not None:
            time_f = open(time_file, "r")
            time_content = time_f.readlines()
            line_skip = 7
        dataset_dict = {}
        for i in range(0, len(content), line_skip):
            model_name = content[i].split(" ")[1].removesuffix("\n")

            downstream_name = content[i + 1].split(" ")[1].removesuffix("\n")
            dataset_name = content[i + 2].split(" ")[1].removesuffix("\n")
            sample_size = float(content[i + 3].split(" ")[2].removesuffix("\n"))
            org_loss = float(content[i + 4].split(" ")[2].removesuffix("\n"))
            syn_loss = float(content[i + 5].split(" ")[2].removesuffix("\n"))

            # if dataset_name in ["california_housing", "microsoft", "year"]:
            #    continue

            # if model_name in ["great"]:
            #    continue

            if dataset_name not in dataset_dict.keys():
                dataset_dict[dataset_name] = {}

            if model_name not in dataset_dict[dataset_name].keys():
                dataset_dict[dataset_name][model_name] = {
                    "size": [],
                    "org": [],
                    "syn": [],
                    "time": [],
                }

            dataset_dict[dataset_name][model_name]["size"].append(sample_size)
            dataset_dict[dataset_name][model_name]["org"].append(org_loss)
            dataset_dict[dataset_name][model_name]["syn"].append(syn_loss)

            if time_file is None:
                training_time = float(content[i + 6].split(" ")[1].removesuffix("\n"))
                dataset_dict[dataset_name][model_name]["time"].append(training_time)

        if time_file is not None:
            for i in range(0, len(time_content), 5):
                model_name = time_content[i].split(" ")[1].removesuffix("\n")
                dataset_name = time_content[i + 1].split(" ")[1].removesuffix("\n")
                sample_size = float(
                    time_content[i + 2].split(" ")[2].removesuffix("\n")
                )
                time = float(time_content[i + 3].split(" ")[1].removesuffix("\n"))

                # if dataset_name in ["california_housing", "microsoft", "year"]:
                #    continue
                if model_name in ["great", "topo"]:
                    continue

                dataset_dict[dataset_name][model_name]["time"].append(time)

        dataset_dicts.append(dataset_dict)

    if plot_data:
        for dataset in dataset_dicts[0]:
            x_1 = []
            x_2 = []
            y = SPLITS
            time = []
            time_001 = []
            time_05 = []
            for dataset_dict in dataset_dicts:
                x_1_p = []
                x_2_p = []
                for model in dataset_dict[dataset]:
                    org_data = dataset_dict[dataset][model]["org"]
                    syn_data = dataset_dict[dataset][model]["syn"]
                    time_001.append(float(dataset_dict[dataset][model]["time"][0]))
                    time_05.append(float(dataset_dict[dataset][model]["time"][4]))
                    time.append(float(np.average(dataset_dict[dataset][model]["time"])))
                    if len(org_data) != 7:
                        pad = 7 - len(org_data)
                        org_data = np.pad(
                            np.array(org_data), (0, pad), constant_values=0
                        ).tolist()
                        syn_data = np.pad(
                            np.array(syn_data), (0, pad), constant_values=0
                        ).tolist()
                    x_1_p.append(org_data)
                    x_2_p.append(syn_data)
                x_1.append(x_1_p)
                print(len(x_1_p))
                x_2.append(x_2_p)

            x_1 = np.array(x_1)[0]
            x_2 = np.array(x_2).mean(axis=0)
            x_3 = np.concatenate(
                (np.array(x_2), np.array([np.mean(np.array(x_1), axis=0)])), axis=0
            )
            labels = list(dataset_dict[dataset].keys())
            markers = ["s", "^", "s", "p", "h", "*", "X", "v", "<", ">"]
            colors = plt.get_cmap("Set3")  # ["r", "g", "b", "c", "m", "y", "brown"]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4.8), sharey=True)

            m = np.array([np.mean(np.array(x_1), axis=0)])[0][:5]

            x = np.logspace(
                np.log10(1),
                np.log10(np.max([np.max(time_001) * 1.5, np.max(time_05)]) * 1.5),
                100,
            )
            y = np.linspace(
                np.min([np.min(x_2[:, 0]), np.min(x_2[:, 4])]) * 0.8 if np.min([np.min(x_2[:, 0]), np.min(x_2[:, 4])]) > 0.1 else np.max([np.max(x_2[:, 0]), np.max(x_2[:, 4])]) * (-0.09),
                np.max([np.max(x_2[:, 0]), np.max(x_2[:, 4])]) * 1.2,
                100,
            )
            if dataset == "aloi":
                y = np.linspace(0, 0.5, 100)

            def add_banded_background(ax, floor, higher_is_better=True, n_bands=10,
                                      light=0.5, dark=0.4, alpha=0.7, res=(400, 600)):
                """Call AFTER xscale/xlim/ylim are set.
                floor: baseline y-value; the bad side of it is solid red."""
                from matplotlib.colors import to_rgb
                ymin, ymax = ax.get_ylim()
                xmin, xmax = ax.get_xlim()
                ny, nx = res

                y_frac = (np.arange(ny) + 0.5) / ny
                y_data = ymin + y_frac * (ymax - ymin)

                # discrete boxes: 0 = bottom band, n_bands-1 = top band
                band = np.minimum((y_frac * n_bands).astype(int), n_bands - 1)
                score = band / (n_bands - 1)
                if not higher_is_better:
                    score = 1 - score
                base = plt.get_cmap("RdYlGn")(score)[:, :3]  # (ny, 3)

                # bad side of the baseline is solid red
                bad = y_data < floor if higher_is_better else y_data > floor
                base[bad] = to_rgb("#d73027")

                # x: light tint (left) -> darker shade (right), same hue
                left = base * (1 - light) + light  # mix with white
                right = base * dark  # mix with black
                t = np.linspace(0, 1, nx)[None, :, None]
                img = left[:, None, :] * (1 - t) + right[:, None, :] * t

                ax.imshow(img, origin="lower", aspect="auto", extent=(0, 1, 0, 1),
                          transform=ax.transAxes, zorder=-1, alpha=alpha)
                ax.set_xlim(xmin, xmax)  # imshow can reset limits
                ax.set_ylim(ymin, ymax)

            higher_is_better = dataset not in [
                "california_housing", "microsoft", "year", "helena", "aloi"
            ]

            for ax in axes:
                ax.set_xscale("log")
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())

            add_banded_background(axes[0], floor=m[0], higher_is_better=higher_is_better)
            add_banded_background(axes[1], floor=m[4], higher_is_better=higher_is_better)

            # axes[0].set_xscale('log')

            axes[0].axhline(
                m[0],
                label="1% original data",
                marker=None,
                color="black",
            )

            axes[1].axhline(
                m[4],
                label="50 % original data",
                marker=None,
                color="black",
                linestyle="dashed",
            )

            for i in range(len(labels)):
                # plt.scatter(time[i], np.mean(x_2[i][:5]), label=labels[i], s=50, marker=markers[i])
                if i == labels.index("topo"):
                    marker_size = 14
                    edgecolor = "white"

                    axes[0].annotate(
                        "Ours",
                        xy=(time_001[i], np.mean(x_2[i][0])),  # arrow tip: the topo marker
                        xytext=(-40, -8),  # text position, in points relative to the tip
                        textcoords="offset points",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        #bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=1),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                        shrinkA=0, shrinkB=8),  # shrinkB keeps the arrowhead off the marker
                        zorder=10,
                    )

                    axes[1].annotate(
                        "Ours",
                        xy=(time_05[i], np.mean(x_2[i][4])),  # arrow tip: the topo marker
                        xytext=(-40, -8),  # text position, in points relative to the tip
                        textcoords="offset points",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        #bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=1),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                        shrinkA=0, shrinkB=8),  # shrinkB keeps the arrowhead off the marker
                        zorder=10,
                    )

                else:
                    marker_size = 8
                    edgecolor = "black"

                c = colors(color_dict[labels[i]])

                if np.mean(x_2[i][0]) > 0:
                    axes[0].plot(
                        time_001[i],
                        np.mean(x_2[i][0]),
                        linestyle="None",
                        marker="o",
                        ms=marker_size,
                        markeredgecolor=edgecolor,
                        color=c,
                    )
                if np.mean(x_2[i][4]) > 0:
                    axes[1].plot(
                        time_05[i],
                        np.mean(x_2[i][4]),
                        linestyle="None",
                        marker="^",
                        ms=marker_size,
                        markeredgecolor=edgecolor,
                        color=c,
                    )

            # plt.axis((-np.max(time)*0.1, np.max(time)*1.1, np.max(x_2[:, :5]), np.min(x_2[:, :5])))
            # plt.legend(ncol=1)
            # plt.legend(loc='best', bbox_to_anchor=(1.0, 1.0))

            y_scale_name = (
                "RMSE (Lower is better) ←"
                if dataset
                in ["california_housing", "microsoft", "year", "helena", "aloi"]
                else "Accuracy (Higher is better) →"
            )

            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D

            handles, _ = plt.gca().get_legend_handles_labels()
            h = []
            for i in range(len(labels)):
                h.append(mpatches.Patch(color=colors(color_dict[labels[i]]), label=labels[i]))

            h.append(
                Line2D(
                    [0],
                    [0],
                    label="",
                    marker="o",
                    markersize=10,
                    markeredgecolor="w",
                    markerfacecolor="w",
                    linestyle="",
                )
            )
            """h.append(
                Line2D(
                    [0],
                    [0],
                    label="1% synthetic data",
                    marker="o",
                    markersize=10,
                    markeredgecolor="k",
                    markerfacecolor="w",
                    linestyle="",
                )
            )
            h.append(
                Line2D(
                    [0],
                    [0],
                    label="50% synthetic data",
                    marker="^",
                    markersize=10,
                    markeredgecolor="k",
                    markerfacecolor="w",
                    linestyle="",
                )
            )
            h.append(
                Line2D(
                    [0],
                    [0],
                    label="",
                    marker="o",
                    markersize=10,
                    markeredgecolor="w",
                    markerfacecolor="w",
                    linestyle="",
                )
            )"""
            h_, _ = axes[0].get_legend_handles_labels()
            h.extend(h_)
            h.extend(handles)

            fig.add_artist(plt.legend(handles=h, loc="best", bbox_to_anchor=(1.0, 1.0)))

            if dataset == "aloi":
                axes[0].set_ylim(-0.01, 0.4)
                axes[1].set_ylim(-0.01, 0.4)

            # axes[0].set_xlabel("Time (s)")
            axes[0].set_title("99% synthetic data", fontweight="bold")
            axes[1].set_title("50% synthetic data", fontweight="bold")
            axes[0].set_ylabel(y_scale_name)
            axes[0].set_xscale("log")
            # axes[1].set_xlabel("Time (s)")
            # axes[1].set_ylabel(y_scale_name)
            axes[1].set_xscale("log")
            fig.align_ylabels(axes)
            fig.suptitle(dataset.title(), fontsize=16, fontweight="bold")
            fig.supxlabel("Time in seconds (Lower is better) ←")
            plt.tight_layout()
            plt.savefig("time_plot_" + dataset + ".pdf")
            fig.show()

            labels.append("Original data")

            fig = plt.figure(figsize=(8, 3))  # (12, 6))
            yy = []
            l = []
            for i in range(len(np.array(x_3).T)):
                yy.append(y)
                l.append(labels)

            br = np.arange(len(SPLITS))

            bar_cmap = plt.get_cmap("Set3")
            rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

            for i in range(len(np.array(x_3))):
                # plt.scatter(y, np.array(x_3)[i][:5], label=labels[i])
                # plt.plot(y, np.array(x_3)[i][:5])

                plt.bar(
                    br,
                    np.array(x_3)[i][:5],
                    width=0.08,
                    edgecolor="grey",
                    label=labels[i],
                    color=bar_cmap(color_dict[labels[i]]),
                )
                br = [x + 0.08 for x in br]

            plt.xticks([r + (0.2) for r in range(len(SPLITS))], SPLITS)

            if dataset == "aloi":
                plt.ylim(0, 0.5)

            fig.add_artist(plt.legend(loc="best", bbox_to_anchor=(1.0, 1.0)))
            plt.xlabel("Percentage data portion")
            plt.ylabel(y_scale_name)
            plt.title(dataset.title(), fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig("bar_plot_" + dataset + ".pdf")
            fig.show()

    if create_table:
        comp_table = None
        dataset_dict = dataset_dicts[1]
        for dataset in dataset_dicts[1]:
            columns = ["model"]
            portions = SPLITS
            for portion in portions:
                columns.append(dataset + "_" + str(portion))

            df = None
            for model in dataset_dict[dataset]:
                syn_data = dataset_dict[dataset][model]["syn"]
                if len(syn_data) != 7:
                    pad = 7 - len(syn_data)
                    # org_data = np.pad(np.array(org_data), (0, pad), constant_values=0).tolist()
                    syn_data = np.pad(
                        np.array(syn_data), (0, pad), constant_values=np.NAN
                    ).tolist()

                syn_data.insert(0, model)
                if df is None:
                    df = pd.DataFrame([syn_data], columns=columns)
                else:
                    d = pd.DataFrame([syn_data], columns=columns)
                    df = pd.concat([df, d], axis="rows")

            df.set_index("model", inplace=True)

            if comp_table is None:
                comp_table = df
            else:
                comp_table = pd.concat([comp_table, df], axis="columns")
        comp_table.fillna(-1, inplace=True)
        comp_table.to_csv("./" + downstream_name + "_table.csv")


if __name__ == "__main__":
    read_results_file(
        ["../results/Other models/results_rmse_new.txt"], time_file=None, plot_data=True
    )
