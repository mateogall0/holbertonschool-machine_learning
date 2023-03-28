#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axes = plt.subplots(nrows=3, ncols=2)

axes[0, 0].plot(y0, color='red')

axes[0, 1].scatter(x1, y1, c='magenta')
axes[0, 1].set_xlabel('Height (in)')
axes[0, 1].set_ylabel('Weight (lbs)')
axes[0, 1].set_title("Men's Height vs Weight")

axes[1, 0].plot(x2, np.log(y2))
axes[1, 0].set_xlabel('Time (years)')
axes[1, 0].set_ylabel('Fraction Remaining')
axes[1, 0].set_title("Exponential decay of C-14")

axes[1, 1].set_xlabel('Time (years)')
axes[1, 1].set_ylabel('Fraction Remaining')
axes[1, 1].set_title('Exponential Decay of Radioactive Elements')
axes[1, 1].plot(x3, y31, color='red', label='C-14', linestyle=':')
axes[1, 1].plot(x3, y32, c='green', label='Ra-226')
axes[1, 1].legend()


axes[2, 1].remove()
axes[2, 0].remove()
ax5 = fig.add_subplot(3, 1, 3)

ax5.set_xlabel('Grades')
ax5.set_ylabel('Number of Students')
ax5.set_title('Project A')
ax5.hist(student_grades, bins=10, edgecolor='black')
ax5.set_xlim(0)
ax5.set_ylim(0, 30)

fig.suptitle('All in One')
fig.tight_layout()
plt.show()
