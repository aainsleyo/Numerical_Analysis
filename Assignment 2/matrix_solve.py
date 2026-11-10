# Code sourced from Moe 
# The convergence tests were written by me with assistance from Claude & VScode when needed.
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from pathlib import Path

# Solve the PDE using an implicit-explicit finite difference method.
def solve_pde(u_init, mu, n, dt, tf):
    h = 2.0 / n
    x = np.linspace(-1, 1, n + 1)
    size = n - 1
    r = dt / (h ** 2)  # common term in all coefficients in M

    # INITIALIZE: M = matrix of coefficients
    main_diag = (1.0 + 2.0 * r) * np.ones(size)
    off_diag = -r * np.ones(size - 1)
    M = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)

    # Find M = PLU (Done ONCE outside the loop)
    P, L, U = scipy.linalg.lu(M)

    # Apply BCs u(-1) = u(1) = 0
    u_init = u_init.copy()
    u_init[0] = 0.0
    u_init[-1] = 0.0
    u_int = u_init[1:n].copy()

    steps = int(round(tf / dt))
    if not np.isclose(steps * dt, tf, rtol=1e-12, atol=1e-14):
        raise ValueError("Choose dt so that tf / dt is an integer.")

    save_interval = max(1, steps // 100)  # Save about 100 snapshots

    snap_time = [0.0]
    solutions = [u_init.copy()]

    # Time loop
    for step in range(steps):
        # F[j] = u[j,t] + dt * mu * sin(u[j,t])
        b = u_int + dt * mu * np.sin(u_int)

        # Solve Ly = P.T b, Ux = y, since M = PLU
        y = scipy.linalg.solve_triangular(
            L, P.T @ b, lower=True, unit_diagonal=True
        )
        u_int = scipy.linalg.solve_triangular(U, y, lower=False)

        if (step + 1) % save_interval == 0 or (step + 1) == steps:
            u_full = np.zeros(n + 1)
            u_full[1:n] = u_int
            solutions.append(u_full.copy())
            snap_time.append((step + 1) * dt)

    return x, snap_time, solutions


def time_convergence():
    """Halve dt on a fixed spatial grid and check for first-order convergence."""
    n = 200
    mu = 1.0
    tf = 0.5
    dt_values = [0.05 / 2**k for k in range(7)]

    x = np.linspace(-1, 1, n + 1)
    u_init = np.cos((np.pi / 2) * x)

    final_solutions = []

    for dt in dt_values:
        _, _, history = solve_pde(u_init, mu, n, dt, tf)
        final_solutions.append(history[-1])

    # Compare each solution with the one using half the time step.
    differences = []

    for k in range(len(dt_values) - 1):
        difference = np.max(
            np.abs(final_solutions[k] - final_solutions[k + 1])
        )
        differences.append(difference)

    print("\nTime convergence: expected order = 1")
    print("dt              Difference       Order")

    for k in range(len(differences)):
        if k == 0:
            print(f"{dt_values[k]:.7f}       {differences[k]:.6e}       -")
        else:
            order = np.log2(differences[k - 1] / differences[k])
            print(
                f"{dt_values[k]:.7f}       "
                f"{differences[k]:.6e}       {order:.4f}"
            )

    return np.array(dt_values[:-1]), np.array(differences)


def space_convergence():
    """Halve h at fixed dt and check for second-order convergence."""
    dt = 0.0005
    mu = 1.0
    tf = 0.5
    n_values = [10, 20, 40, 80, 160]

    final_solutions = []

    for n in n_values:
        x = np.linspace(-1, 1, n + 1)
        u_init = np.cos((np.pi / 2) * x)

        _, _, history = solve_pde(u_init, mu, n, dt, tf)
        final_solutions.append(history[-1])

    differences = []

    for k in range(len(n_values) - 1):
        coarse = final_solutions[k]

        # Every second point on the finer grid matches a coarse-grid point.
        fine = final_solutions[k + 1][::2]

        difference = np.max(np.abs(coarse - fine))
        differences.append(difference)

    h_values = 2.0 / np.array(n_values[:-1])

    print("\nSpace convergence: expected order = 2")
    print("h               Difference       Order")

    for k in range(len(differences)):
        if k == 0:
            print(f"{h_values[k]:.7f}       {differences[k]:.6e}       -")
        else:
            order = np.log2(differences[k - 1] / differences[k])
            print(
                f"{h_values[k]:.7f}       "
                f"{differences[k]:.6e}       {order:.4f}"
            )

    return h_values, np.array(differences)


# Test run with some IC and also plotting intermediate time steps
if __name__ == "__main__":
    n = 400
    dt = 0.005
    tf = 100.0
    mu = np.pi**2 / 4 + 0.005

    x = np.linspace(-1, 1, n + 1)

    # Enter ICs
    u_init = np.cos((np.pi / 2) * x)
    print(u_init)

    x_grid, snap_times, history = solve_pde(u_init, mu, n, dt, tf)

    # Create the output folder if it does not already exist.
    Path("Numerical Analysis").mkdir(exist_ok=True)

    # Used Claude to help make plots look nicer
    plt.figure(figsize=(8, 5))

    # for t_val, u_sol in zip(snap_times, history):
    #     plt.plot(x_grid, u_sol, label=f't = {t_val:.2f}')

    mid = n // 2  # index of x = 0
    u_mid = [u_sol[mid] for u_sol in history]

    plt.semilogy(snap_times, u_mid, 'o-', label='u(0, t)')
    plt.xlabel("t")
    plt.ylabel("u(0, t)")
    plt.title("Solution at x = 0 vs time (IMEX finite difference)")
    plt.grid(True)
    plt.legend()
    plt.savefig("Numerical Analysis/Heat_Eq_Solns.pdf")
    plt.show()

    # plt.xlabel("x")
    # plt.ylabel("u(x, t)")
    # plt.title("Heat Equation Solutions with Implicit-Explicit Finite Difference methods")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("Numerical Analysis/Heat_Eq_Solns.pdf")
    # plt.show()

    # Run convergence tests.
    dt_values, time_differences = time_convergence()
    h_values, space_differences = space_convergence()

    # Plot time convergence.
    plt.figure(figsize=(8, 5))
    plt.loglog(
        dt_values, time_differences, 'o-', label="Solution difference"
    )
    plt.loglog(
        dt_values,
        time_differences[0] * (dt_values / dt_values[0]),
        '--',
        label="First-order reference"
    )
    plt.xlabel("Time step, delta")
    plt.ylabel("Maximum absolute difference")
    plt.title("Time Convergence (IMEX finite difference)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Numerical Analysis/Time_Convergence.pdf")
    plt.show()

    # Plot space convergence.
    plt.figure(figsize=(8, 5))
    plt.loglog(
        h_values, space_differences, 'o-', label="Solution difference"
    )
    plt.loglog(
        h_values,
        space_differences[0] * (h_values / h_values[0])**2,
        '--',
        label="Second-order reference"
    )
    plt.xlabel("Grid spacing, h")
    plt.ylabel("Maximum absolute difference")
    plt.title("Space Convergence (IMEX finite difference)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Numerical Analysis/Space_Convergence.pdf")
    plt.show()
    