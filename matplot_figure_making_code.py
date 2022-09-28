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
    0.987098127627215,
    0.9835027136562979,
    0.9864920036646826,
    0.9920597393155013,
    0.9910670127490269,
    0.9940599283935535,
    0.9918087392805752,
    0.9931354564674233,
    0.9937649580205941,
    0.9957538481777792,
    0.9948663423149521,
    0.9957681192983825,
    0.9968921069847587,
    0.9973481531714224,
    0.9968087582127152,
    0.9960675324691113,
    0.998102334257965,
    0.9973614432315095,
    0.9988798057121345,
]
attack_f1_scores = [
    0.9992728761170179,
    0.9981935140269573,
    0.9976481095806958,
    0.998049943304532,
    0.9969857095961947,
    0.9975097952169518,
    0.9957022408226285,
    0.9952296283536561,
    0.994856315665577,
    0.9957242815456796,
    0.9937572756692867,
    0.9938539418440719,
    0.9943662114927146,
    0.9935741620096394,
    0.9906283617943483,
    0.9847332819348502,
    0.9883841664337458,
    0.9758676302459417,
    0.9784505305563396,
]
accuracy_scores = [
    0.998625,
    0.9967499999999999,
    0.9959999999999999,
    0.996875,
    0.9955000000000002,
    0.9964999999999998,
    0.9943750000000001,
    0.994375,
    0.9943749999999998,
    0.9957499999999998,
    0.994375,
    0.9949999999999999,
    0.9960000000000001,
    0.9962499999999999,
    0.9952500000000001,
    0.9937500000000001,
    0.9967499999999999,
    0.9952499999999999,
    0.997875,
]

ax.plot(
    x,
    normal_f1_scores,
    marker="o",
    color="#6060FF",
    linestyle="solid",
)

ax.plot(
    x,
    attack_f1_scores,
    marker="o",
    color="#FF6060",
    linestyle="solid",
)
ax.plot(
    x,
    accuracy_scores,
    marker="o",
    color="#60B060",
    linestyle="solid",
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
    "Average Normal F1 Score": "b",
    "Average Attack F1 Score": "r",
    "Average Accuracy": "g",
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
