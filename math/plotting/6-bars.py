#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']


fig, ax = plt.subplots()
bar_width = 0.5
x_pos = np.arange(len(fruit[0]))

for i, row in enumerate(fruit):
    ax.bar(x_pos, row, width=bar_width, bottom=np.sum(fruit[:i], axis=0), color=colors[i], label=['apples', 'bananas', 'oranges', 'peaches'][i])

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_ylim([0, 80])
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(x_pos)
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])
ax.legend()

plt.show()
