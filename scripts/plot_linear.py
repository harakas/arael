#!/usr/bin/env python3
"""Generate plot for linear regression example."""

import matplotlib.pyplot as plt
import numpy as np

# Data from linear_demo.rs
data = [
    (-0.15640527, -0.09394677),
    (-0.14665490, -0.09246022),
    (-0.13697288, -0.09540069),
    (-0.12694226, -0.12290291),
    (-0.11715084, -0.07633987),
    (-0.10758017, -0.09448499),
    (-0.09716778, -0.21283103),
    (-0.08797396, -0.09011850),
    (-0.07798443, -0.20681172),
    (-0.06789591, -0.20985495),
    (-0.05861036, -0.09025026),
    (-0.04905056, -0.08905702),
    (-0.03925969, -0.09053987),
    (-0.02946681, -0.08921083),
    (-0.01946522, -0.08783635),
    (-0.00987463, -0.08561212),
    (-0.00014984, -0.08510941),
    ( 0.00974805, -0.08513614),
    ( 0.01940540, -0.08678824),
    ( 0.02935162, -0.08533194),
    ( 0.03921752, -0.08541373),
]

x = np.array([d[0] for d in data])
y = np.array([d[1] for d in data])

# Fit coefficients from the demo
lin_a, lin_b = 0.17499836, -0.09714453
rob_a, rob_b = 0.05899305, -0.08699960

x_line = np.array([x.min() - 0.01, x.max() + 0.01])

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(x, y, c='#2196F3', s=30, zorder=3, label='Data')

ax.plot(x_line, lin_a * x_line + lin_b, c='#FF9800', linewidth=1.5, label='Linear regression')
ax.plot(x_line, rob_a * x_line + rob_b, '-', c='#4CAF50', linewidth=2, label='Robust fit')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Robust Linear Regression with Error Suppression')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/linear_regression.png', dpi=150)
print("Saved docs/linear_regression.png")
