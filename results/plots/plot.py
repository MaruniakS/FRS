import matplotlib.pyplot as plt
import numpy as np

precision = [0.92, 0.91, 0.86]
recall = [1, 0.94, 0.77]
f1_score = [0.96, 0.93, 0.81]

labels = ['Direct', 'Indirect', 'Outage']
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
bars1 = ax.bar(x - width, precision, width, label='Precision', color='gold')
bars2 = ax.bar(x, recall, width, label='Recall', color='orangered')
bars3 = ax.bar(x + width, f1_score, width, label='F1-score', color='deeppink')

# Add value labels on top
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0, 1.05)
plt.show()
