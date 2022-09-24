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
    0.9852851660593597,
    0.9849635540583662,
    0.9818202246761075,
    0.9922255310734874,
    0.9900192215388672,
    0.9921071402714935,
    0.9931364929227364,
    0.9939849721379158,
    0.9950632785978082,
    0.9967693949064163,
    0.9972272252912283,
    0.9947957081265596,
    0.9965521637200233,
    0.9967703195264142,
    0.9969024599408798,
    0.9974943950468015,
    0.9977844998457253,
    0.9980643821505805,
    0.9989489559590503,
]
attack_f1_scores = [
    0.9992083763468743,
    0.9983302701059404,
    0.9967533563528509,
    0.9979433233584883,
    0.9968784712927139,
    0.9965966944556952,
    0.9963461699994139,
    0.9960593134834064,
    0.9958509753408414,
    0.9967256589211703,
    0.9967246685355564,
    0.992143465377603,
    0.9934989680781274,
    0.992502851153551,
    0.9896777322356591,
    0.9899896942383182,
    0.9876599509903802,
    0.9817448117851866,
    0.9783697308550107,
]
accuracy_scores = [
    0.9984999999999999,
    0.9969999999999999,
    0.9945,
    0.9967499999999999,
    0.9952500000000001,
    0.9952500000000002,
    0.9952500000000001,
    0.9952500000000001,
    0.9955,
    0.9967499999999999,
    0.9969999999999999,
    0.99375,
    0.9955,
    0.9955,
    0.9952499999999999,
    0.9960000000000001,
    0.9962500000000001,
    0.9964999999999999,
    0.998,
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
