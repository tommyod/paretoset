# ==========================================================================================
# =============================== EXAMPLE 1: HOTELS ========================================
# ==========================================================================================

from skyline import skyline
import pandas as pd

hotels = pd.DataFrame({"price": [50, 53, 62, 87, 83, 39, 60, 44], "distance_to_beach": [13, 21, 19, 13, 5, 22, 22, 25]})
mask = skyline(hotels, sense=["min", "min"])
skyline_hotels = hotels[mask]


# ==================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(5.5, 3))

plt.title("Skyline hotels (Pareto-efficient hotels)")

plt.scatter(hotels["price"], hotels["distance_to_beach"], zorder=10, label="All hotels", s=50, alpha=0.8)

plt.scatter(
    skyline_hotels["price"], skyline_hotels["distance_to_beach"], zorder=5, label="Skyline hotels", s=150, alpha=1
)


plt.legend()
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xlabel("Price")
plt.ylabel("Distance to the beach")
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.tight_layout()
plt.savefig("example_hotels.png", dpi=100)
plt.show()


# ==========================================================================================
# =============================== EXAMPLE 2: SALESPEOPLE ===================================
# ==========================================================================================

from skyline import skyline
import pandas as pd

salespeople = pd.DataFrame(
    {
        "salary": [94, 107, 67, 87, 153, 62, 43, 115, 78, 77, 119, 127],
        "sales": [123, 72, 80, 40, 64, 104, 106, 135, 61, 81, 162, 60],
        "department": ["c", "c", "c", "b", "b", "a", "a", "c", "b", "a", "b", "a"],
    }
)
mask = skyline(salespeople, sense=["min", "max", "diff"])
top_performers = salespeople[mask]


# ==================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3))
# plt.suptitle("Salespeople eligible for a raise (high sales, low salary)", y=1.00)

salespeople_by_dept = salespeople.groupby("department")
performers_by_dept = top_performers.groupby("department")

subplot = 1
for (group_salespeople, group_performers) in zip(salespeople_by_dept, performers_by_dept):
    plt.subplot(1, 3, subplot)

    department, group_salespeople = group_salespeople
    _, group_performers = group_performers

    plt.title(f"Department '{department}'")
    plt.scatter(
        group_salespeople["salary"], group_salespeople["sales"], zorder=10, label="Salespeople", s=50, alpha=0.8
    )

    plt.scatter(group_performers["salary"], group_performers["sales"], zorder=4, label="Top performers", s=150, alpha=1)

    plt.xlabel("Salary")
    if subplot == 1:
        plt.ylabel("Sales")
        plt.legend()
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.subplots_adjust(top=0.5)
    subplot += 1

plt.tight_layout()
plt.savefig("example_salespeople.png", dpi=100)
plt.show()


# ==========================================================================================
# =============================== EXAMPLE 2: SALESPEOPLE ===================================
# ==========================================================================================

from skyline import skyline
import numpy as np
from collections import namedtuple

np.random.seed(42)

# Create Solution objects holding the problem solution and objective values
Solution = namedtuple("Solution", ["solution", "objective_values"])
solutions = [Solution(solution=object, objective_values=np.random.randn(2)) for _ in range(999)]

# Create an array of shape (solutions, objectives) and compute the non-dominated set
objective_values_array = np.vstack([s.objective_values for s in solutions])
mask = skyline(objective_values_array, sense=[min, min])

# Filter the list of solutions, keeping only the non-dominated solutions
efficient_solutions = [solution for (solution, m) in zip(solutions, mask) if m]


# ==================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(5.5, 3))

plt.title("Objective value space and efficient solutions")

plt.scatter(objective_values_array[:, 0], objective_values_array[:, 1], zorder=10, label="Solutions", s=10, alpha=0.8)

plt.scatter(
    objective_values_array[mask, 0],
    objective_values_array[mask, 1],
    zorder=5,
    label="Efficient solutions",
    s=50,
    alpha=1,
)


plt.legend(loc="upper right").set_zorder(50)
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.tight_layout()
plt.savefig("example_optimization.png", dpi=100)
plt.show()
