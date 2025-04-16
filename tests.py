import pytest

import numpy as np
import matplotlib.pyplot as plt
import scipy

from main import Kernel, Theorem9SolverR2, eq_spaced_grid, BrownianMotion, VectorisedRandomVortices, Theorem9SolverHalfPlane


def chorin_cutoff(r, delta):
        return np.where(r > delta, 1, r / delta)

def test_kernel_cutoff():
    # plot a vector field for a chorin kernel!
    # (technically not a unit test - sue me)
    kerny = Kernel(chorin_cutoff, 0.1)
    points = eq_spaced_grid([-0.2, 0.2], [-0.2, 0.2], 400)
    kernel_vals = kerny.k_R2(x=points[:,0], y=points[:,1])

    # plot hidden so they don't keep popping up!
    # plt.quiver(points[:, 0], points[:, 1], kernel_vals[:, 0], kernel_vals[:, 1])
    # plt.show()

def test_brownian_motion():
    bm = BrownianMotion(std_dev=1, value=np.array([0.0, 3.0]))
    xs = []
    ys = []
    for i in range(8000):
        bm.step(0.001)
        x,y = bm.value
        xs += [x]
        ys += [y]

    # plot hidden so they don't keep popping up!
    # plt.plot(xs, ys)
    # plt.show()

def test_velocity_recovery():
    random_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-1, 1], [-1, 1]],
        strength=lambda x: np.zeros(len(x)),
        viscosity=0.1,
        x_grid_size=2,
        y_grid_size=2,
        estimate_expectation=True,
        num_simulations=1
    )

    solver = Theorem9SolverR2(
        kernel=Kernel(chorin_cutoff, 0.1),
        timestep=0.1,
        random_vortices=random_vortices
    )

    points = eq_spaced_grid([-1, 1], [-1, 1], 1000)
    velocity_field = solver.u(points)

    # plot hidden so they don't keep popping up!
    # plt.quiver(points[:, 0], points[:, 1], velocity_field[:, 0], velocity_field[:, 1])
    # plt.show()

def test_two_vortex_motion():
    # In the inviscous case, two vortices with opposite strength must move in parallel
    random_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-1, 1], [-1, 1]],
        strength=lambda x: np.zeros(len(x)) + (x[...,0] > 0) - (x[...,0] < 0),
        viscosity=0,
        x_grid_size=2,
        y_grid_size=1,
        estimate_expectation=True,
        num_simulations=1,
        store_path=True
    )
    solver = Theorem9SolverR2(
        kernel=Kernel(chorin_cutoff, 0.1),
        timestep=0.1,
        random_vortices=random_vortices
    )

    for _ in range(500):
        solver.step()

    path1 = np.array(solver.state.vortices.path[:, 0])
    path2 = np.array(solver.state.vortices.path[:, 1])      

    # asssert the vortices move in parallel
    np.testing.assert_equal(path1[:,1], path2[:,1])
    
    # assert the vortices move with speed (1/4pi)
    d = solver.state.vortices.initial_xs[1][0] - solver.state.vortices.initial_xs[0][0]
    np.testing.assert_almost_equal(path1[-1][1], -solver.time * 1 / (d * np.pi), 1)

def test_single_sum_vortex_initialisation():
    # test the vortices are initialised in a grid, and that the dxdy term is as expected
    random_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-5, 5], [-2, 1]],
        strength=lambda x: np.zeros(len(np.array(x))) + 1,
        viscosity=0,
        x_grid_size=3,
        y_grid_size=3
    )

    np.testing.assert_equal(random_vortices.n, 9)
    np.testing.assert_equal(random_vortices.area_elements[0], 30/9)

def test_double_sum_initialisation():
    # test the vortices are initialised in a grid, and that the dxdy term is as expected
    num_simulations = 10
    random_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-1, 1], [-1, 1]],
        strength=lambda x: np.zeros(len(np.array(x))) + 1,
        viscosity=0.0,
        x_grid_size=3,
        y_grid_size=3,
        num_simulations=10,
        estimate_expectation=True
    )

    np.testing.assert_equal(random_vortices.n, 90)
    np.testing.assert_equal(random_vortices.area_elements[0], 4/9)
    np.testing.assert_equal(len(random_vortices.xs), num_simulations * 9)

def test_brownian_motion_independence():
    # test the brownian motions are shared for double sum, and independent for single version
    single_sum_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-1, 1], [-1, 1]],
        strength=lambda x: np.zeros(len(np.array(x))) + 1,
        viscosity=0.0,
        x_grid_size=3,
        y_grid_size=3,
    )

    double_sum_vortices =  VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=[[-1, 1], [-1, 1]],
        strength=lambda x: np.zeros(len(np.array(x))) + 1,
        viscosity=0.0,
        x_grid_size=3,
        y_grid_size=3,
        estimate_expectation=True,
        num_simulations=3
    )

    last_step = 0
    for i in range(single_sum_vortices.n):
        # this is only a rough test to see if two brownian consecutive motions take the same step
        step = single_sum_vortices.brownian_motions[i].step(timestep = 1)
        if (last_step == step).all():
            assert "Brownian motions copied in single sum solver!"
        last_step = step

    for i in range(double_sum_vortices.num_simulations):
        last_step = 0
        for j in range(double_sum_vortices.n // double_sum_vortices.num_simulations):
            step = double_sum_vortices.brownian_motions[i * double_sum_vortices.num_simulations + j].step(timestep=1)
            # this is only a rough test to see if two brownian consecutive motions take the same step
            if i != 0 and (last_step != step).any():
                assert "Brownian motions not copied in double sum solver!"
            last_step = step

def test_plot_simulations_seperately():
    # we expect each simulation to look sensible on it's own
    domain = np.array([[-1, 1], [-1, 1]])
    random_vortices = VectorisedRandomVortices.gen_rectangular_grid(
        parameter_domain=domain,
        strength=lambda x: np.zeros(len(np.array(x))) + 1,
        viscosity=0.001,
        x_grid_size=3,
        y_grid_size=3,
        estimate_expectation=True,
        num_simulations=2
    )
    solver = Theorem9SolverR2(
        kernel=Kernel(chorin_cutoff, 0.2),
        random_vortices=random_vortices,
        timestep=0.1,
    )

    solver.run_simulation(iterations=30)

    fig, ax = plt.subplots(2)

    num_vector_points = 1600
    plot_domain = 1.5 * domain
    grid_points = eq_spaced_grid(plot_domain[0], plot_domain[1], num_vector_points)

    for i in range(2):
        ax[i].set_xlim(*plot_domain[0])
        ax[i].set_ylim(*plot_domain[1]) 

        # TODO: find a nice way to fix this unit test - need different states for each simulation
        # field = solver.calculate_u(x=grid_points, vortices=random_vortices.vortices[i], kernel=solver.kernel)
        # quiver = ax[i].quiver(grid_points[:,0], grid_points[:, 1], field[:,0], field[:, 1])
    # plt.show()

def test_k_half_plane():
    x = [0, 1]
    y = [2, 4]

    # test the formula for k_half_plane matches that from Choir 2022 Half Plane!
    np.testing.assert_almost_equal(
        Kernel(cutoff_function=chorin_cutoff, cutoff_radius=0.1).k_half_plane(x=x, y=y).ravel(),
        np.array([
            1 / (2*np.pi) * ((y[1] - x[1]) / ((y[0]-x[0])**2 + (y[1] - x[1])**2) 
                           - (y[1] + x[1]) / ((y[0]-x[0])**2 + (y[1] + x[1])**2)),
            1 / (2*np.pi) * ((y[0] - x[0]) / ((y[0]-x[0])**2 + (y[1] + x[1])**2) 
                           - (y[0] - x[0]) / ((y[0]-x[0])**2 + (y[1] - x[1])**2))
        ]),
        10
    )

def test_k_half_plane_integrals():
    # the numerical integral of the x-component and the y-component (dy) should match their closed forms!
    y1 = 3
    x = [5, 6]
    k = Kernel(cutoff_function=chorin_cutoff, cutoff_radius=0.1)

    for i in [0, 1]:
        np.testing.assert_almost_equal(
            scipy.integrate.quad_vec(
                f=lambda y: k.k_half_plane(x, [y1, y])[i], 
                a=0,
                b=10,
                limit=10000)[0] + 
            scipy.integrate.quad_vec(
                f=lambda y: k.k_half_plane(x, [y1, y])[i], 
                a=10,
                b=1e9,
                limit=10000)[0],
            k.integrate_k_half_plane_dy_2(x=x, y1=y1)[i],
            5
        )
