import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import alphas

mpl.use("pdf")

plt.rc("font", family="serif", serif="Times")
plt.rc("text", usetex=True)
plt.rc("ytick", labelsize=8)
plt.rc("axes", labelsize=8)

# width as measured in inkscape
width = 3.487
height = width / 1.618


title_size = 7
alpha_value = 0.625
fig, ax = plt.subplots(figsize=(width, height))
plt.suptitle(
    "Performance Data for Increasing Percentage of Normal Data",
    fontsize=7,
)

ax.set_ylim([0.9, 1.0])
ax.set_xlim([0, 100])
number_of_data_points = 19
x = [
    i / (number_of_data_points + 1) * 100
    for i in range(1, number_of_data_points + 1)
]

normal_f1_scores = [
    0.9592998174677471,
    0.976570386569457,
    0.9789024434132019,
    0.9811042140947295,
    0.9859021671730422,
    0.9881088652974543,
    0.987197618706962,
    0.9878705189913539,
    0.9886963213056301,
    0.9900800997335185,
    0.9910966685415415,
    0.9916523145507721,
    0.9900413051935633,
    0.9895399894280029,
    0.9935832230205804,
    0.9939214135083267,
    0.9942693185462325,
    0.9937717435254884,
    0.9953915111384211,
]
attack_f1_scores = [
    0.9978272276191824,
    0.9974724038298305,
    0.9963306134761813,
    0.9952719101122852,
    0.9953436487852546,
    0.9948484853488744,
    0.9931333296974009,
    0.9919349117203426,
    0.9907251488215454,
    0.9900426607164817,
    0.9890635622432111,
    0.9873332371433318,
    0.9817840358406738,
    0.9749769470935243,
    0.9807372471677676,
    0.9753381680348484,
    0.9673321016453384,
    0.9418353518500562,
    0.9130767830452164,
]
accuracy_scores = [
    0.9958750000000001,
    0.9954375000000001,
    0.9937500000000001,
    0.9924375,
    0.993,
    0.9928125000000001,
    0.9910625,
    0.9903124999999999,
    0.9898125,
    0.9900625,
    0.9901875,
    0.9899374999999999,
    0.987125,
    0.9852500000000001,
    0.9903749999999999,
    0.99025,
    0.99025,
    0.9887499999999999,
    0.99125,
]

ax.plot(
    x,
    normal_f1_scores,
    "-bo",
)

ax.plot(
    x,
    attack_f1_scores,
    "-ro",
)
ax.plot(
    x,
    accuracy_scores,
    "-go",
)


plt.xticks(
    x,
    fontsize=title_size - 1,
)

plt.yticks(
    fontsize=title_size - 1,
)


plt.ylabel("Scores", fontsize=title_size - 1)
plt.xlabel("Percentage of Normal Data", fontsize=title_size - 1)


box = ax.get_position()
ax.set_position(
    [
        box.x0 - box.width * 0.00,
        box.y0 + box.height * 0.175,
        box.width * 1.1,
        box.height * 0.85,
    ]
)

colors = {
    "Normal F1 Score": "b",
    "Attack F1 Score": "r",
    "Accuracy": "g",
}

labels = list(colors.keys())
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[label], alpha=alpha_value)
    for label in labels
]
plt.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    fontsize=title_size - 1,
    handletextpad=0.2,
    handlelength=1.0,
    columnspacing=1.0,
)

fig.savefig("plot.pdf")
