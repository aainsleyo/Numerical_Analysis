import numpy as np
import matplotlib.pyplot as plt


def finite_difference(u, h, method='central'):
    """Compute the derivative using periodic neighbors."""
    if method == 'central':
        return (np.roll(u, -1) - np.roll(u, 1)) / (2 * h)
    elif method == 'forward':
        return (np.roll(u, -1) - u) / h
    elif method == 'backward':
        return (u - np.roll(u, 1)) / h

# Define function and its exact derivative.
u = lambda x: np.exp(np.sin(x))
u_prime = lambda x: np.exp(np.sin(x)) * np.cos(x)

# Choose the interval and grid sizes.
a, b = -np.pi, np.pi
methods = ['central', 'forward', 'backward']
grid_sizes = [2**k for k in range(4, 30)]

# Show overlapping curves.
styles = {
    'central': 'o-',
    'forward': 's--',
    'backward': 'x:'
}

plt.figure(figsize=(9, 6))

for method in methods:
    results = []

    print(f"\nMethod: {method}")
    print(f"{'N':>12} {'h':>15} {'Maximum abs error':>20}")

    for N in grid_sizes:
        h = (b - a) / N
        x = a + np.arange(N) * h

        values = u(x)
        approximation = finite_difference(values, h, method)
        error = np.max(np.abs(approximation - u_prime(x)))

        results.append((N, h, error))
        print(f"{N:12d} {h:15.6e} {error:20.8e}")

        # Release the large arrays before the next grid.
        del x, values, approximation

    # Find the smallest error among the tested grids.
    best = min(results, key=lambda row: row[2])
    N_best, h_best, error_best = best

    # Fit the slope where truncation error dominates.
    N_array = np.array([row[0] for row in results])
    h_array = np.array([row[1] for row in results])
    error_array = np.array([row[2] for row in results])

    mask = (
        (N_array >= 64)
        & (N_array <= 4096)
        & (error_array > 0)
    )

    slope = np.polyfit(
        np.log(h_array[mask]),
        np.log(error_array[mask]),
        1
    )[0]

    print(f"\nBest tested N: {N_best}")
    print(f"Best tested h: {h_best:.8e}")
    print(f"Minimum maximum absolute error: {error_best:.8e}")
    print(f"Truncation-region slope versus h: {slope:.4f}")
    print(f"Truncation-region slope versus N: {-slope:.4f}")

    if error_best == results[-1][2]:
        print("Minimum occurs at the largest tested grid.")
        print("Further refinement is needed to locate the turn upward.")

    # Plot on one figure.
    plt.loglog(
        N_array, error_array, styles[method],
        label=method.capitalize(),
        markersize=5,
        markerfacecolor='none'
    )

plt.xlabel('Number of grid points, N')
plt.ylabel('Maximum absolute error')
plt.title('Finite-difference error on a periodic grid')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('finite_difference_error.png', dpi=300)
plt.show()