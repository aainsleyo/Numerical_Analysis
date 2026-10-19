# Import Libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


# Define the exact solution and Picard iteration function.
def exact_solution(t):
    return np.sqrt(4 + 2*t) - 1


# Define the Picard iteration function.
def picard(t1, N):
    # Regular grid and initial guess.
    t = np.linspace(0, t1, N)
    u = np.ones(N)
    u_exact = exact_solution(t)

    # Compute the initial error.
    error = np.max(np.abs(u - u_exact))

    while True:
        # Picard iteration.
        u_new = 1 + cumulative_trapezoid(
            1 / (1 + u), t, initial=0
        )

        new_error = np.max(np.abs(u_new - u_exact))

        # Return the minimum error when it stops decreasing.
        if new_error >= error:
            return error

        u = u_new
        error = new_error


# Parameters
t1 = 1.0
N = 16

N_values = []
errors = []

# Start the timer and run Picard iteration until one hour has elapsed.
start = time.perf_counter()

# Increase N until approximately one hour has elapsed.
while time.perf_counter() - start < 3600:
    try:
        e_N = picard(t1, N)
    except MemoryError:
        print("Stopped: insufficient memory for the next grid.")
        break

    N_values.append(N)
    errors.append(e_N)

    print(f"N = {N:10d}, e_N = {e_N:.6e}")

    N *= 2


# Plot minimum error versus number of grid points.
plt.loglog(N_values, errors, "o-")
plt.xlabel("Number of grid points, N")
plt.ylabel(r"Minimum error, $e_N$")
plt.title("Grid-Based Picard Iteration")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.show()