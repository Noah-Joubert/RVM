from __future__ import annotations
from typing import Callable, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import scipy


def eq_spaced_grid(x_range, y_range, N):
    # Generates a grid of equally spaced points
    sqrt_num_vortices = int(np.sqrt(N))  # as two-dimensional grid
    return rectangular_grid(x_range=x_range, y_range=y_range, x_grid_size=sqrt_num_vortices, y_grid_size=sqrt_num_vortices)

def rectangular_grid(x_range, y_range, x_grid_size, y_grid_size):
    # Generates a rectangular grid of equally spaced points
    x_grid_size = int(x_grid_size)
    y_grid_size = int(y_grid_size)
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    x_points = np.linspace(x_range[0] + x_span / (x_grid_size + 1), x_range[1] - x_span / (x_grid_size + 1), x_grid_size)
    y_points = np.linspace(y_range[0] + y_span / (y_grid_size + 1), y_range[1] - y_span / (y_grid_size + 1), y_grid_size)
    X, Y = np.meshgrid(x_points, y_points)
    return np.column_stack([X.ravel(), Y.ravel()])      

def reflect_in_half_plane(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.stack([
        x[..., 0], -x[..., 1]
    ], axis=-1)

def build_simulations(xs: np.ndarray, strength: Callable[[np.ndarray], float], area_element: float, viscosity: float, estimate_expectation: bool, num_simulations: int):
    # two ways of implementing the RVM
    # 1. Negletct the expectation by providing each vortex with an independent brownian motion
    # 2. Calculate the expectation, by simultaneously running several simulations  n
    vortices = []
    strengths = np.array(strength(xs))
    if estimate_expectation:
        for _ in range(num_simulations):
            simulation = []
            # each vortex in a simulation should share a brownian motion
            bm = BrownianMotion()
            for i, posn in enumerate(xs):
                simulation.append(
                    Vortex(
                        x=posn, 
                        strength=strengths[i], 
                        brownian_motion=bm,
                        area_element=area_element,
                        viscosity=viscosity
                    )
                )
            vortices.append(simulation)

    # or a single simulation with independent brownian motions  
    else:
        # if neglecting the expectation, each vortex should have its own brownian motion
        s = []
        for i, posn in enumerate(xs):
            s.append(
                Vortex(
                    x=posn, 
                    strength=strengths[i], 
                    brownian_motion=BrownianMotion(),
                    area_element=area_element,
                    viscosity=viscosity)
                )
        vortices = s

    vortices = np.array(vortices)
    # we treat the case single sum case as a double sum case with 1 simulation
    return np.atleast_2d(vortices)  


@dataclass
class BrownianMotion:
    # keep track of vortex ids for seeding randomness
    _counter: int = field(default=0, init=False, repr=False)  # Class-level counter
    id: int = field(init=False)  # Instance ID

    # our implementation of Brownian Motion
    std_dev: float = field(default_factory=lambda: 1)
    value: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    last_step: np.ndarray = field(init=False, default_factory=lambda: np.array([0.0, 0.0]))
    time: float = field(init=False, default_factory=lambda: 0)
    rng: np.random.Generator = field(init=False, default_factory=np.random.default_rng)  # Independent RNG

    def __post_init__(self):
        type(self)._counter += 1  # Increment counter
        self.id = self._counter  # Assign unique ID
        self.rng = np.random.default_rng(self.id)

    def step(self, timestep: float) -> np.ndarray:
        # evaluate our brownian motion after a timestep
        dB = self.rng.normal(loc=0.0, scale=self.std_dev * np.sqrt(timestep), size=self.value.shape)
        self.value += dB
        self.time += timestep
        self.last_step = dB
        return dB
    

@dataclass
class Kernel:
    # A kernel is defined by a cutoff_function and a cutoff_radius.
    # this is to avoid the singularity in the regular Biot-Savarat kernel
    cutoff_function: Callable[[float, float], float]
    cutoff_radius: float
    T: Callable = lambda x: x  # conformal map
    dTdX: Callable = lambda x: np.stack(
        [
            np.full_like(x[..., 0], 1),
            np.full_like(x[..., 0], 0),
        ], axis=-1
    )  # y derivative
    dTdY: Callable = lambda x: np.stack(
        [
            np.full_like(x[..., 0], 0),
            np.full_like(x[..., 0], 1),
        ], axis=-1
    )  # x derivative

    def k_R2(self, x: np.ndarray, y: np.ndarray, eps: float = 0.00001) -> np.ndarray:
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        
        # broadcast to matrices
        n, m = len(x), len(y)
        matrix_x = np.tile(x, (m, 1, 1))
        matrix_y = np.tile(y, (n, 1, 1)).transpose(1, 0, 2)
        
        # evaluate the kernel
        z = matrix_x - matrix_y
        r = np.linalg.norm(z, axis=-1)
        r = np.maximum(r, eps)  # avoid division by zero error
        transformed_points = np.stack((-z[..., 1], z[..., 0]), axis=-1)
        result = transformed_points * (
            self.cutoff_function(r, self.cutoff_radius) / (2 * np.pi * r * r)
        )[...,np.newaxis]

        # reduce dimension
        return np.squeeze(result)
    
    def k_half_plane(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # k_half_plane = R2(x, y) - R2(x, reflected(y)) = (y,0) / 2*pi*r^2
        # TODO: validate that this formula is correct 
        return self.k_R2(x, y) - self.k_R2(reflect_in_half_plane(x), y)
    
    def k_indicator_half_plane(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # K.k_indicator_half_plane is K_D from the Qian 2022 Montecarlo paper
        # K_D(x, y) := K_half_plane(x, y) I{y > 0}
        # NOTE: here the inequality is made inclusive to calculate the derivative in the Thm9 Boundary Term
        result = self.k_half_plane(x=x, y=y)
        factors = (np.atleast_2d(y)[...,1] <= 0)
        if result.ndim == 3:
            factors = factors[..., np.newaxis, np.newaxis]
        else:
            factors = factors[..., np.newaxis]
        return result * factors

    def integrate_k_half_plane_dy_2(self, x: np.ndarray, y1: float) -> np.ndarray:
        # returns the integral of K_half_plane(x,(y1, y2))dy_2 from (0, \infty)
        original_x = np.array(x)
        x = np.atleast_2d(x)
        result = np.column_stack([
            np.zeros(shape=len(x)),
            1 / (2*np.pi) * (np.arctan2(y1 - x[:, 0], x[:, 1]) - np.arctan2(y1 - x[:, 0], -x[:, 1]))
        ])
        if original_x.ndim == 1:
            return result[0]
        else:
            return result

    def conformal_k(self, x: np.ndarray, y: np.nadarray, eps: float = 0.1) -> np.ndarray:
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        
        # broadcast to matrices
        n, m = len(x), len(y)
        matrix_x = np.tile(x, (m, 1, 1))
        matrix_y = np.tile(y, (n, 1, 1)).transpose(1, 0, 2)

        norm_1 = np.maximum(
            np.linalg.norm(self.T(matrix_x)-self.T(matrix_y), axis=-1),
            eps
        ) 
        norm_2 = np.maximum(
            np.linalg.norm(self.T(matrix_x)-reflect_in_half_plane(self.T(matrix_y)), axis=-1),
            eps
        )
        k_minus = (self.T(matrix_y) - self.T(matrix_x)) /(2*np.pi*norm_1*norm_1)[..., np.newaxis] 
        k_plus = (self.T(matrix_y) - reflect_in_half_plane(self.T(matrix_x))) /(2*np.pi*norm_2*norm_2)[..., np.newaxis] 

        d = self.dTdY(matrix_y)
        k = k_minus - k_plus
        K_1 = d[..., 0] * k[..., 0] + d[..., 1] * k[..., 1]
        d = self.dTdX(matrix_y)
        k = -k_minus + k_plus
        K_2 = d[..., 0] * k[..., 0] + d[..., 1] * k[..., 1]

        result = np.squeeze(
            np.stack(
                [K_1, K_2], axis=-1
            )
        )

        # remove singularities from x==y
        mask = np.isnan(result)
        result[mask] = 0
        return result

@dataclass
class State:
    # An instance of State stores a copy of the simulation at a point in time.
    # Its purpose is to hold the data to recover the velocity field at some point in the past.
    # It is needed because different subclassess of RandomVortexSovler will require different parameters to find the velocity.
    # For instance, the boundary vorticty is required for velocity calculations in Theorem8.
    # It is here implemented as a dynamic dataclass
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
    
    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
    
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


@dataclass
class VectorisedRandomVortices:
    # all of these properties are lists, where one entry is one vortex
    # each list has size (num_simulations * num_vortices_per_simulation)
    domain: np.ndarray
    parameter_domain: np.ndarray
    mapping: Callable  # map from the parameter domain to the euclidian domain
    initial_xs: np.array[np.ndarray] 
    area_elements: np.array[float]
    next_delta_x: np.array[np.ndarray] = field(init=False)
    xs: np.array[np.ndarray] = field(init=False)
    strengths: np.array[float] = field(init=False)
    brownian_motions: np.array[BrownianMotion] = field(init=False)
    factors: np.array[float] = field(init=False)  # area_element * strength 

    # theorem 2pt4 stuff
    left_domain: np.array[float] = field(init=False)
    force_sums: np.array[float] = field(init=False)
    boundary_sums: np.array[float] = field(init=False)

    # other properties
    estimate_expectation: bool
    num_simulations: int 
    strength: Callable[[np.ndarray], np.ndarray]
    viscosity: float
    n: int = field(init=False)  # total number of vortices
    n_per_sim: int = field(init=False)  # number of vortices per sim

    # debugging properties
    store_path: bool = field(default_factory = lambda: False)
    path: np.ndarray = field(init=False)
 
    def __post_init__(self):
        self.n = len(self.initial_xs)
        self.xs = np.copy(self.initial_xs)
        self.next_delta_x = np.zeros(self.n)
        self.strengths = self.strength(self.initial_xs)
        self.factors = self.strengths * self.area_elements
        self.left_domain = np.full(self.n, False)
        self.force_sums = np.full(self.n, 0.0)
        self.boundary_sums = np.full(self.n, 0.0)

        self.n_per_sim = self.n // self.num_simulations
        # check each simulation contains the same number of vortices
        np.testing.assert_equal(self.n_per_sim, self.n / self.num_simulations)

        brownian_motions = []
        if not self.estimate_expectation:
            for _ in range(self.n):
                brownian_motions.append(
                    BrownianMotion()
                )
        else:
            for _ in range(self.num_simulations):
                bm = BrownianMotion()
                for _ in range(self.n_per_sim):
                    brownian_motions.append(bm)
        self.brownian_motions = np.array(brownian_motions)

        self.path = np.array([np.copy(self.xs)])

    @classmethod
    def gen_rectangular_grid(
        cls,
        parameter_domain: np.ndarray,
        strength: Callable[[np.ndarray], float],
        viscosity: float,
        x_grid_size: int,
        y_grid_size: int,
        estimate_expectation: bool = False,
        num_simulations: int = 1,
        store_path: bool = False,
        euclidian_domain: np.ndarray = None,
        mapping: Callable = lambda x: x
    ) -> VectorisedRandomVortices:
        n = int(x_grid_size * y_grid_size * num_simulations)

        # generate the parameter values
        xs = np.tile(
            rectangular_grid(x_range=parameter_domain[0], y_range=parameter_domain[1], x_grid_size=x_grid_size, y_grid_size=y_grid_size),
            (num_simulations, 1)
        )
        xs = mapping(xs)  # map the parameter values to Rsquared

        if euclidian_domain is None:
            euclidian_domain = parameter_domain # for when the params are not euclidian (x, y)

        area_element = (euclidian_domain[0][1] - euclidian_domain[0][0]) * (euclidian_domain[1][1] - euclidian_domain[1][0]) / (x_grid_size * y_grid_size)
        
        area_elements = np.full((n), area_element)

        return cls(
            domain=euclidian_domain,
            parameter_domain=parameter_domain,
            mapping=mapping,
            initial_xs=xs, 
            area_elements=area_elements, 
            estimate_expectation=estimate_expectation,
            num_simulations=num_simulations,
            strength=strength,
            viscosity=viscosity,
            store_path=store_path
        )

    def prepare_step(self, velocity_field: np.ndarray, timestep: float):
        assert len(velocity_field) == self.n

        velocity_term = velocity_field * timestep 
        # TODO: vectorise this
        viscosity_term = np.sqrt(2*self.viscosity) * np.array([bm.last_step for bm in self.brownian_motions])
        delta_x = velocity_term + viscosity_term
        delta_x = np.squeeze(delta_x)
        self.next_delta_x = delta_x  # NOTE: this might need to be in an np.copy()

    def step(self):
        if self.store_path:
            self.path = np.concatenate(
            [
                self.path,
                [np.copy(self.xs)]
            ]
            )
        self.xs += self.next_delta_x
   
    def __add__(self, vortices: VectorisedRandomVortices) -> VectorisedRandomVortices:
        # merge two sets of vortices
        # NOTE: merging looses the property that vortices are ordered by their simulations
        # mergine two sets of vortices means they become UNORDERED
        assert self.num_simulations == vortices.num_simulations, "To merge two sets of vortices, they must have the same number of simulations."
        assert self.estimate_expectation == vortices.estimate_expectation, "To merge two sets of vortices, we can't estimate expectations if the other doesn't."

        # expand the domain
        x_domain = [
            np.min([self.domain[0][0], vortices.domain[0][0]]),
            np.max([self.domain[0][1], vortices.domain[0][1]]),
        ]
        y_domain = [
            np.min([self.domain[1][0], vortices.domain[1][0]]),
            np.max([self.domain[1][1], vortices.domain[1][1]]),
        ]
        self.domain = np.array([x_domain, y_domain])

        self.initial_xs = np.concatenate([self.initial_xs, vortices.initial_xs])
        self.area_elements = np.concatenate([self.area_elements, vortices.area_elements])
        self.xs = np.concatenate([self.xs, vortices.xs])
        self.strengths = np.concatenate([self.strengths, vortices.strengths])
        self.brownian_motions = np.concatenate([self.brownian_motions, vortices.brownian_motions])
        self.factors = np.concatenate([self.factors, vortices.factors])

        self.left_domain = np.concatenate([self.left_domain, vortices.left_domain])
        self.force_sums = np.concatenate([self.force_sums, vortices.force_sums])
        self.boundary_sums = np.concatenate([self.boundary_sums, vortices.boundary_sums])

        self.n += vortices.n

        return self


@dataclass
class VectorisedPhatomVortices(VectorisedRandomVortices):
    # PhantomVortices are used specifically to calculate the boundary term in Thm9 from Qian 2022 Half Plane
    # To calulate the $\xi^2$ derivative, vortices come in pairs. One spawned at the boundary, an one spawned just from it
    # Although this class inherits from VectorisedRandomVortices, most of the functionality of the parent class is not needed

    def __init__(
        self,
        region: np.ndarray,
        num_simulations: int,
        estimate_expectation: bool,
        theta: Callable[[np.ndarray], np.ndarray],
        new_vortex_rate: int,
        eps: float = 0.1
    ):
        # build the x-coordinates for generating new vortices
        self.new_vortex_rate = new_vortex_rate
        self.eps = eps
        self.range = region
        self.rng = np.random.default_rng()  # Independent RNG
        self.length_element = (self.range[1]-self.range[0]) / self.new_vortex_rate

        # build the generating xs
        x1s = np.column_stack([
            np.linspace(*self.range, self.new_vortex_rate),
            np.zeros(self.new_vortex_rate)
        ])
        x2s = np.column_stack([
            np.linspace(*self.range, self.new_vortex_rate),
            np.zeros(self.new_vortex_rate) - self.eps
        ])
        xs = []
        for i in range(self.new_vortex_rate):
            xs.append(x1s[i])
            xs.append(x2s[i])
        self.generating_xs = np.array(xs) + [self.rng.uniform(-self.length_element, self.length_element), 0]

        super().__init__(
            domain = np.array([region, [0, 0]]),
            initial_xs= self.generating_xs,
            area_elements=np.full((2*self.new_vortex_rate), self.length_element),
            num_simulations=num_simulations,
            estimate_expectation=estimate_expectation,
            strength = theta,
            viscosity = 0
        )
        
    def generate(self, theta: Callable[[np.ndarray], np.ndarray]):
        new_vortices = VectorisedPhatomVortices(
            region=self.range,
            num_simulations=self.num_simulations,
            estimate_expectation=self.estimate_expectation,
            theta=theta,
            eps=self.eps,
            new_vortex_rate=self.new_vortex_rate
        )

        self += new_vortices


@dataclass
class Plot:
    # Encapsulates a single plot of the fluid
    grid_size: int
    state: State

    fig: plt.figure
    ax: plt.axes
    gif: bool

    def __post_init__(self):
        self.parameter_domain = self.state.parameter_domain
        self.euclidian_domain = self.state.vortices.domain
        self.parametric_mapping = self.state.parametric_mapping
        self.use_quiver = self.state.use_quiver
        self.use_velocity_cmap = self.state.use_velocity_cmap
        self.use_vorticity_cmap = self.state.use_vorticity_cmap
        self.use_streamplot = self.state.use_streamplot
        self.use_scatter = self.state.use_scatter
        self.in_domain = self.state.in_domain

        self.ax.set_xlim(self.euclidian_domain[0][0], self.euclidian_domain[0][1])
        self.ax.set_ylim(self.euclidian_domain[1][0], self.euclidian_domain[1][1])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        # initialise the plots
        # TODO: have streamplot_X/Y be in euclidian coordinates
        self.streamplot_X, self.streamplot_Y = np.meshgrid(np.linspace(*self.euclidian_domain[0], self.grid_size), np.linspace(*self.euclidian_domain[1], self.grid_size))
        self.stream_plot_grid = np.stack(
            [
                self.streamplot_X.ravel(),
                self.streamplot_Y.ravel()
            ], axis=-1
        )

        X, Y = np.meshgrid(np.linspace(*self.parameter_domain[0], self.grid_size), np.linspace(*self.parameter_domain[1], self.grid_size))
        cmap_grid = np.stack([X, Y], axis=-1)
        cmap_grid = self.parametric_mapping(cmap_grid)
        self.grid = cmap_grid.reshape((self.grid_size**2, 2))
        self.scales_grid = self.grid

        self.cmap_x = cmap_grid[..., 0]
        self.cmap_y = cmap_grid[..., 1]

        magnitude = np.zeros(self.grid_size ** 2).reshape((self.grid_size, self.grid_size))
        
        if self.use_velocity_cmap:
            self.velocity_cmap = self.ax.pcolormesh(self.cmap_x, self.cmap_y, magnitude, shading='auto', cmap='viridis', alpha=0.6)
        if self.use_vorticity_cmap:
            self.vorticity_cmap = self.ax.pcolormesh(self.cmap_x, self.cmap_y, magnitude, shading='auto', cmap='viridis', alpha=0.6)
        if self.use_quiver:
            self.quiver = self.ax.quiver(self.grid[:,0], self.grid[:, 1], np.zeros(self.grid_size ** 2), np.zeros(self.grid_size ** 2))
        if self.use_scatter:
            if self.gif:
                self.scatter = self.ax.scatter(np.zeros(10), np.zeros(10), marker='x', color='red', alpha=0.6, s=15) 
            else:
                self.scatter = self.ax.scatter(np.zeros(10), np.zeros(10), marker='x', color='red', alpha=0.4, s=7) 
        if self.use_streamplot:
            stream_plot_field = np.zeros((self.grid_size, self.grid_size, 2))
            self.streamplot = self.ax.streamplot(self.streamplot_X, self.streamplot_Y, stream_plot_field[..., 0], stream_plot_field[..., 1], color='black', linewidth=0.8)
    
    def set_scales(self, 
                   state: State, 
                   velocity_function: Callable, 
                   vorticity_function: Callable):
        # set the colorbar and vector field scales. 
        # should be done using the last frame of the animation/ plot
        velocity_field = velocity_function(x=self.scales_grid, state=state)

        magnitude = np.linalg.norm(velocity_field, axis=1)
        vorticity_field = vorticity_function(x=self.grid, state=state)

        if self.use_quiver:
            self.quiver.set_UVC(velocity_field[:,0], velocity_field[:, 1])

        if self.use_velocity_cmap:
            self.colorbar = self.fig.colorbar(self.velocity_cmap, label='Velocity Magnitude') # Update color scale
            self.velocity_cmap.set_clim(magnitude.min(), magnitude.max())

        if self.use_vorticity_cmap:
            self.colorbar = self.fig.colorbar(self.vorticity_cmap, label='Vorticity Magnitude') # Update color scale
            self.vorticity_cmap.set_clim(vorticity_field.min(), vorticity_field.max())

    def update(self, 
               state: State, 
               velocity_function: Callable, 
               vorticity_function: Callable, 
               time: float, 
               update_scales: bool = True):
        field = velocity_function(x=self.grid, state=state)
        vortex_xs = state.vortices.xs

        cmap_field = velocity_function(x=self.grid, state=state)
        magnitude = np.linalg.norm(cmap_field, axis=1)
        magnitude = magnitude.reshape((self.grid_size, self.grid_size))
        vorticity_field = vorticity_function(x=self.grid, state=state)
        vorticity_field = vorticity_field.reshape((self.grid_size, self.grid_size))

        if update_scales and self.use_velocity_cmap: 
            self.colorbar = self.fig.colorbar(self.velocity_cmap, label='Velocity Magnitude') # Update color scale
            self.velocity_cmap.set_clim(magnitude.min(), magnitude.max())

        if update_scales and self.use_vorticity_cmap: 
            self.colorbar.remove()
            self.colorbar = self.fig.colorbar(self.vorticity_cmap, label='Vorticity Magnitude') # Update color scale
            self.vorticity_cmap.set_clim(vorticity_field.min(), vorticity_field.max())

        if self.use_vorticity_cmap:
            self.vorticity_cmap.set_array(vorticity_field.ravel())
        if self.use_velocity_cmap:
            self.velocity_cmap.set_array(magnitude.ravel())
        if self.use_quiver:
            self.quiver.set_UVC(field[:,0], field[:, 1])
        if self.use_scatter:
            in_domain = state.in_domain(vortex_xs)
            self.scatter.set_offsets(vortex_xs[in_domain])
        if self.use_streamplot:
            # NOTE: streamplots won't work with gif's as they can't be modified once drawn
            streamplot_u = velocity_function(self.stream_plot_grid, state=state)
            streamplot_u[~self.in_domain(self.stream_plot_grid)] = np.array([np.nan, np.nan])
            streamplot_u = streamplot_u.reshape(self.grid_size, self.grid_size, 2)
            self.ax.streamplot(self.streamplot_X, self.streamplot_Y, streamplot_u[..., 0], streamplot_u[..., 1], color='black', linewidth=0.8)


        self.ax.set_title(f"t = {time:.2f}s")        


class RandomVortexSolver(ABC):
    # an ABC class for all implementations of the random vortex method
    # each method deviaties in how it computes the velocity field
    # however the simulation and plotting logic remains the same
    def __init__(self,
                 kernel: Kernel,
                 timestep: float,
                 random_vortices: VectorisedRandomVortices,
                 estimate_expectation: bool = False,
                 num_simulations: int = 1,
                 **kwargs  # kwargs get passed to the state
        ):
        self.kernel = kernel  # callable kernel for recovering u
        self.time = 0

        self.timestep = timestep  # timestep of simulation
        self.random_vortices = random_vortices
        self.state = State(
            kernel=kernel, 
            vortices=random_vortices, 
            estimate_expectation=estimate_expectation,
            num_simulations=num_simulations,
            **kwargs
        )

        self.state.viscosity = self.random_vortices.viscosity
        self.state.timestep = self.timestep
        self.state.parametric_mapping = self.random_vortices.mapping
        self.state.parameter_domain = self.random_vortices.parameter_domain

        if not hasattr(self.state, 'G'):
            self.state.G = lambda x: np.zeros(len(x))
        if not hasattr(self.state, 'respawn_vortices'):
            self.state.respawn_vortices = False
        if not hasattr(self.state, 'dd_phi'):
            self.state.dd_phi = lambda x: np.zeros(len(x))
        if not hasattr(self.state, 'eps'):
            self.state.eps = 0.1
        if not hasattr(self.state, 'use_quiver'):
            self.state.use_quiver = True
        if not hasattr(self.state, 'project_to_boundary'):
            self.state.project_to_boundary = lambda x: np.stack(
                [
                    x[..., 0], np.full_like(x[..., 0], 0)
                ], axis=-1
            )
        if not hasattr(self.state, 'dist_from_boundary'):
            self.state.dist_from_boundary = lambda x: np.abs(x[..., 1])
        if not hasattr(self.state, 'in_domain'):
            self.state.in_domain = lambda x: x[..., 1] < 0
        if not hasattr(self.state, 'use_scatter'):
            self.state.use_scatter = True
        if not hasattr(self.state, 'use_velocity_cmap'):
            self.state.use_velocity_cmap = True
        if not hasattr(self.state, 'use_vorticity_cmap'):
            self.state.use_vorticity_cmap = False
        if not hasattr(self.state, 'use_streamplot'):
            self.state.use_streamplot = False
        self.__post__init__()

    def build_gif(self):
        # set up the figures
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 7)
        plot = Plot(
            grid_size=30,
            state=self.state,
            fig=fig,
            ax=ax,
            gif=False,
        )

        plot.set_scales(state=self.states[-1], velocity_function=self.plotting_u, vorticity_function=self.omega)

        frames_per_update_scale = 5
        def update_animation(frame):
            frame_index = frame
            plot.update(
                state=self.states[frame_index],
                velocity_function=self.plotting_u,
                vorticity_function=self.omega,
                time=self.time * (frame_index * self.iterations_per_frame) / (len(self.states) * self.iterations_per_frame),
                update_scales=(frame % frames_per_update_scale) == 0
            )

        ani = animation.FuncAnimation(fig, update_animation, frames=len(self.states) - 1, interval=50, blit=False)
        plt.show()
        ani.save("fluid_flow.gif", writer="ffmpeg")

    def build_plot(self, rows: int = 2, columns: int = 2):
        # TODO: non nxn plots aren't working
        rows, columns = np.max([2, rows]), np.max([2, columns])
        num_frames = len(self.states)
        num_plots = rows * columns
        num_plots = np.min([num_plots, num_frames])

        fig, axs = plt.subplots(rows, columns)
        fig.set_size_inches(7 * columns, 6*rows)
        for i in range(num_plots):
            N = 20
            plot = Plot(
                grid_size=N, 
                state=self.state,
                fig=fig, 
                ax=axs[i % rows][i // columns], 
                gif=False,
            )
            frame_index = i * num_frames // num_plots
            plot.update(
                state=self.states[frame_index], 
                velocity_function=self.plotting_u,
                vorticity_function=self.omega,
                time=self.time * frame_index / num_frames
            )

        plt.savefig("fluid_flow.jpeg", dpi=300, bbox_inches='tight')
        plt.show()

        return plot # TODO: remove this

    def run_simulation(self, iterations: int, iterations_per_frame: int = 1):
        states: List[State] = []

        # first run the simulation, and save snapshots
        for i in tqdm(range(iterations)):
            if i % iterations_per_frame == 0:
                states.append(copy.deepcopy(self.state))

            self.step()  

        self.states = states
        self.iterations_per_frame = iterations_per_frame

    def step(self):
        # run a time step
        self.time += self.timestep  

        # TODO: vectorise this 
        # advance the brownian motions
        for bm in self.random_vortices.brownian_motions:
            if bm.time != self.time:
                bm.step(self.timestep)
                assert bm.time == self.time, f"Brownian Motion time out of sync with solver time."

        # prepare the step
        u = self.u(self.random_vortices.xs)
        self.random_vortices.prepare_step(
            velocity_field = u,
            timestep = self.timestep
        )

        # do the step   
        self.random_vortices.step()

        # run any post-step computationas 
        self.post_step()

    def omega(self, x: np.ndarray, eps=0.5, delta=0.2, state: Optional[State] = None) -> np.ndarray:
        # calculates the vorticity (-\frac{\partial u^1}{\partial y}) using a finite difference method
        # also, it takes a weighted average of neighbouring vorticity for smoothing
        if state is None:
            state = self.state

        x = np.atleast_2d(x)

        delv_delx = (self.omega_u(x, state=state) - self.omega_u(x - np.array([eps, 0]), state=state))[..., 1] / (eps)
        delu_dely = (self.omega_u(x, state=state) - self.omega_u(x - np.array([0, eps]), state=state))[..., 0] / (eps)

        result = (delv_delx - delu_dely) 

        return np.squeeze(result)

    def smooth_omega(self, x: np.ndarray, state: Optional[State] = None, N: int = 40) -> np.ndarray:
        # TODO: not sure this is needed and also doesn't generalise nicely to different domains
        # smooth omega estimator which takes the average vorticity across four neighbouring points
        if state is None:
            state = self.state
        
        domain = self.random_vortices.domain
        steps = np.array([[domain[0][1] - domain[0][0], 0], [0, domain[1][1]-domain[1][0]]]) / N
        omega = np.zeros(len(x))
        for step in [
            + steps[0] + steps[1], 
            - steps[0] + steps[1],
            + steps[0] - steps[1]
            - steps[0] - steps[1]]:
            points = x + step
            omega += self.omega(x=points, state=state) / 4
        return omega

    def theta(self, x: np.ndarray, eps=0.5, delta=0.2, state: Optional[State] = None) -> np.ndarray:
        if state is None:
            state = self.state

        boundary_xs = self.state.project_to_boundary(x)
        return self.omega(x=boundary_xs, eps=eps, delta=delta, state=state)

    def post_step(self):
        # optional method to be overridden for computations after a simulation step
        ...

    def __post__init__(self):
        # optional method to run post init computaitons such as update self.state
        ...

    @property
    def plotting_u(self):
        # overridden by subclass to plot different components of the velocity field 
        return self.u

    @property
    def omega_u(self):
        # overridden by subclass to use a different component of the velocity field for calcuating theta as some introduce instability
        return self.u

    @abstractmethod
    def u(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        ...


class Theorem9SolverR2(RandomVortexSolver):
    def u(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the velocity field for the current vortex configuration
        if state is None:
            state = self.state
        
        # calculate the R2 velocity, vectorising on the vortices (useful when x.shape << vortices.shape)
        # at points x, given a vortex configuration and kernel
        kernel =  state.kernel
        vortex_xs = state.vortices.xs      
        vortex_factors = state.vortices.factors
        num_simulations = state.vortices.num_simulations

        x = np.array(x)
        x = np.atleast_2d(x)
        u = np.array(
            [
                np.sum(
                    kernel.k_R2(pos, vortex_xs) * vortex_factors[:, np.newaxis], axis=0
                ) for pos in x
            ]
        )

        u /= num_simulations

        # may need to flatten to one-dimension!
        return np.squeeze(u)


class Theorem9SolverHalfPlane(RandomVortexSolver):
    def __post__init__(self):
        # set the default `state` variables
        if not hasattr(self.state, 'new_vortex_rate'):
            self.state.new_vortex_rate = 10

        if not hasattr(self.state, 'use_boundary_term'):
            self.state.use_boundary_term = True

        if not hasattr(self.state, 'print_boundary_term'):
            self.state.print_boundary_term = False

        # init the phantom vortices 
        self.state.phantom_vortices = None
        self.phantom_vortices = VectorisedPhatomVortices(
            region=self.random_vortices.domain[0],
            new_vortex_rate=self.state.new_vortex_rate,
            estimate_expectation=self.state.estimate_expectation,
            num_simulations=self.state.num_simulations,
            theta=self.theta
        )
        self.state.phantom_vortices = self.phantom_vortices    

    def u_boundary_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the boundary velocity term for the current vortex configuration
        if state is None:
            state = self.state

        phantom_vortices = state.phantom_vortices
        boundary_term = np.zeros(x.shape)

        # first check we have any phantom vortices
        if phantom_vortices is not None: 
            stacked_xs = phantom_vortices.xs
            pairs_1 = stacked_xs[::2]
            pairs_2 = stacked_xs[1::2]

            factors = (phantom_vortices.factors / phantom_vortices.initial_xs[..., 1])[1::2]
            kernel_1 = state.kernel.k_indicator_half_plane(x, pairs_1)
            kernel_2 = state.kernel.k_indicator_half_plane(x, pairs_2)

            factors = factors[..., np.newaxis, np.newaxis] if kernel_1.ndim == 3 else factors[..., np.newaxis]
            sum_terms = (kernel_1 - kernel_2) * factors
            boundary_term = -2*state.viscosity*state.timestep * np.sum(sum_terms, axis=0)
            
        return boundary_term / state.vortices.n
        
    def u_main_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the first velocity field term for the current vortex configuration
        if state is None:
            state = self.state

        # calculate the half-plane velocity,
        # at points x, given a vortex configuration and kernel (in bounded half-plane {y>0} setup)
        kernel = state.kernel
        vortex_xs = state.vortices.xs   
        vortex_factors = state.vortices.factors 
        num_simulations = state.vortices.num_simulations

        x = np.array(x)
        x = np.atleast_2d(x)
        kernel_difference = (kernel.k_indicator_half_plane(x, vortex_xs) - kernel.k_indicator_half_plane(x, reflect_in_half_plane(vortex_xs)))
        factors = vortex_factors[..., np.newaxis] if kernel_difference.ndim == 2 else vortex_factors[..., np.newaxis, np.newaxis]
        u = np.sum(
            kernel_difference * factors,
            axis=0
        ) 
    
        return u / num_simulations

    def u(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        x = np.array(x)
        u = self.u_main_term(x=x, state=state)

        if state.use_boundary_term:
            bdary_term = self.u_boundary_term(x=x, state=state)
            u += bdary_term

            if state.print_boundary_term:
                print(f"boundary term: {bdary_term}")
                print(f"u term: {u}")

        # may need to squeeze out dimensions!
        return np.squeeze(u)

    def post_step(self):
        if not self.state.use_boundary_term:
            return
        
        # spawn in phantom vortices!
        self.phantom_vortices.generate(self.theta)
        
        # step the random vortices!  # TODO: can this somehow be incorporated into self.step()?
        self.phantom_vortices.prepare_step(
            velocity_field=self.u(self.phantom_vortices.xs),
            timestep=self.timestep
        )
        self.phantom_vortices.step()
    
    @property
    def plotting_u(self):
        if hasattr(self.state, 'plot_all_terms') and self.state.plot_all_terms:
            return self.u
        elif hasattr(self.state, 'plot_first_term') and self.state.plot_first_term:
            return self.u_main_term
        elif hasattr(self.state, 'plot_second_term') and self.state.plot_second_term:
            return self.u_boundary_term
        
        return self.u


class Theorem2pt4(RandomVortexSolver):
    def __post__init__(self):
        if not hasattr(self.state, 'periodic_boundary_conditions'):
            self.state.periodic_boundary_conditions = True
    
    def u_main_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the first velocity field term for the current vortex configuration
        if state is None:
            state = self.state

        # calculate the half-plane velocity,
        # at points x, given a vortex configuration and kernel (in bounded half-plane {y>0} setup)
        kernel = state.kernel
        vortex_xs = state.vortices.xs  
        vortex_factors = state.vortices.factors  # delta_x * omega_0
        num_simulations = state.vortices.num_simulations
        has_vortex_left = ~state.vortices.left_domain

        x = np.atleast_2d(x)
        kernel_term = kernel.k_half_plane(x, vortex_xs)
        factors = vortex_factors * has_vortex_left
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]
        u = np.sum(
            kernel_term * factors,
            axis=0
        ) 
    
        return u / num_simulations

    def u_force_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the force term in the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        x = np.atleast_2d(x)

        kernel = state.kernel
        vortex_xs = state.vortices.xs   
        delta_t = self.timestep
        delta_xs = state.vortices.area_elements 
        force_sums = state.vortices.force_sums
        factors = delta_t * delta_xs * force_sums
        kernel_term = kernel.k_half_plane(x, vortex_xs)
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]

        u = np.sum(
            kernel_term * factors,
            axis=0
        ) * self.timestep

        return u

    def u_boundary_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the boundary term in the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        kernel = state.kernel
        vortex_xs = state.vortices.xs   
        delta_t = self.timestep
        delta_xs = state.vortices.area_elements 
        boundary_sums = state.vortices.boundary_sums
        factors = delta_t * delta_xs * boundary_sums
        kernel_term = kernel.k_half_plane(x, vortex_xs)
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]

        u = np.sum(
            kernel_term * factors,
            axis=0
        ) * self.state.viscosity / (self.state.eps ** 2)

        return u

        return 0

    def u(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        x = np.array(x)

        u = self.u_main_term(x=x, state=state) + self.u_force_term(x=x, state=state) + self.u_boundary_term(x=x, state=state)

        return np.squeeze(u)

    def post_step(self):
        self.state.vortices.left_domain = (self.state.vortices.xs > 0)[..., 1] | self.state.vortices.left_domain

        # reset the sums if needed
        self.state.vortices.force_sums = self.state.vortices.force_sums * (self.state.vortices.xs < 0)[..., 1]
        self.state.vortices.boundary_sums = self.state.vortices.boundary_sums * (self.state.vortices.xs < 0)[..., 1]

        # incriment the sums
        self.state.vortices.force_sums += self.state.G(self.state.vortices.xs)
        dd_phi = self.state.dd_phi(self.state.vortices.xs / self.state.eps)
        boundary_vortices_mask = dd_phi != 0
        theta = np.zeros_like(dd_phi)
        theta[boundary_vortices_mask] = self.theta(self.state.vortices.xs[boundary_vortices_mask])
        self.state.vortices.boundary_sums += dd_phi * theta
        self.state.vortices.boundary_sums *= (self.state.vortices.xs < 0)[..., 1]

        # if using periodic boundary conditions, whack them in!
        if self.state.periodic_boundary_conditions:
            domain_x_width = self.random_vortices.domain[0][1] - self.random_vortices.domain[0][0]
            self.random_vortices.xs[..., 0] = (self.random_vortices.xs[..., 0] - self.random_vortices.domain[0][0]) % (domain_x_width) + self.random_vortices.domain[0][0]

    @property
    def plotting_u(self):   
        if hasattr(self.state, 'plot_all_terms') and self.state.plot_all_terms:
            return self.u
        elif hasattr(self.state, 'plot_boundary_term') and self.state.plot_boundary_term:
            return self.u_boundary_term
        elif hasattr(self.state, 'plot_force_term') and self.state.plot_force_term:
            return self.u_force_term
        
        return self.u
    
    @property
    def omega_u(self):
        if hasattr(self.state, 'use_only_boundary_for_omega') and self.state.use_only_boundary_for_omega:
            return self.u_boundary_term
        if hasattr(self.state, 'use_boundary_for_omega') and self.state.use_boundary_for_omega:
            return self.u
        
                
        return lambda x, state = None: self.u_main_term(x=x, state=state) + self.u_force_term(x=x, state=state)


class Theorem2pt4GeneralDomain(RandomVortexSolver):
    def __post__init__(self):
        if not hasattr(self.state, 'x_periodic_domain'):
            self.state.x_periodic_domain = False
        if not hasattr(self.state, 'y_periodic_domain'):
            self.state.y_periodic_domain = False
    
    def u_main_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the first velocity field term for the current vortex configuration
        if state is None:
            state = self.state

        # calculate the half-plane velocity,
        # at points x, given a vortex configuration and kernel (in bounded half-plane {y>0} setup)
        kernel = state.kernel
        vortex_xs = state.vortices.xs  
        vortex_factors = state.vortices.factors  # delta_x * omega_0
        num_simulations = state.vortices.num_simulations
        has_vortex_left = ~state.vortices.left_domain

        x = np.atleast_2d(x)
        kernel_term = kernel.conformal_k(x, vortex_xs)
        factors = vortex_factors * has_vortex_left
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]
        u = np.sum(
            kernel_term * factors,
            axis=0
        ) 
    
        return u / num_simulations

    def u_force_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the force term in the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        x = np.atleast_2d(x)

        kernel = state.kernel
        vortex_xs = state.vortices.xs   
        delta_t = self.timestep
        delta_xs = state.vortices.area_elements 
        force_sums = state.vortices.force_sums
        factors = delta_t * delta_xs * force_sums
        kernel_term = kernel.conformal_k(x, vortex_xs)
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]

        u = np.sum(
            kernel_term * factors,
            axis=0
        ) * self.timestep

        return u

    def u_boundary_term(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the boundary term in the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        kernel = state.kernel
        vortex_xs = state.vortices.xs   
        delta_t = self.timestep
        delta_xs = state.vortices.area_elements 
        boundary_sums = state.vortices.boundary_sums
        factors = delta_t * delta_xs * boundary_sums
        kernel_term = kernel.conformal_k(x, vortex_xs)
        factors = factors[..., np.newaxis] if kernel_term.ndim == 2 else factors[..., np.newaxis, np.newaxis]

        u = np.sum(
            kernel_term * factors,
            axis=0
        ) * self.state.viscosity / (self.state.eps ** 2)

        return u

    def u(self, x: np.ndarray, state: Optional[State] = None) -> np.ndarray:
        # evaluates the velocity field for the current vortex configuration
        if state is None:
            state = self.state

        x = np.array(x)

        u = self.u_main_term(x=x, state=state) + self.u_force_term(x=x, state=state) + self.u_boundary_term(x=x, state=state)

        return np.squeeze(u)

    def post_step(self):
        in_domain = self.state.in_domain(self.state.vortices.xs)
        self.state.vortices.left_domain = (~in_domain) | self.state.vortices.left_domain

        # incriment the sums
        self.state.vortices.force_sums += self.state.G(self.state.vortices.xs)
        dist_from_boundary = self.state.dist_from_boundary(self.state.vortices.xs)
        dd_phi = self.state.dd_phi(dist_from_boundary / self.state.eps)
        boundary_vortices_mask = dd_phi != 0
        theta = np.zeros_like(dd_phi)
        theta[boundary_vortices_mask] = self.theta(self.state.vortices.xs[boundary_vortices_mask])
        self.state.vortices.boundary_sums += dd_phi * theta

        # reset the sums if needed
        self.state.vortices.force_sums = self.state.vortices.force_sums * in_domain
        self.state.vortices.boundary_sums = self.state.vortices.boundary_sums * in_domain
        self.state.vortices.boundary_sums *= in_domain

        # implement periodic boundary conditions 
        if self.state.x_periodic_domain:
            domain_x_width = self.random_vortices.domain[0][1] - self.random_vortices.domain[0][0]
            self.random_vortices.xs[..., 0] = (self.random_vortices.xs[..., 0] - self.random_vortices.domain[0][0]) % (domain_x_width) + self.random_vortices.domain[0][0]
        if self.state.y_periodic_domain:
            domain_x_width = self.random_vortices.domain[1][1] - self.random_vortices.domain[1][0]
            self.random_vortices.xs[..., 1] = (self.random_vortices.xs[..., 1] - self.random_vortices.domain[1][0]) % (domain_x_width) + self.random_vortices.domain[1][0]

        # optionally respawn vortices_
        if self.state.respawn_vortices:
            xs = self.random_vortices.xs
            domain = self.state.vortices.domain
            x_width = domain[0][1] - domain[0][0]
            y_width = domain[1][1] - domain[1][0]
            respawn_mask = (xs[..., 0] < domain[0][0] - x_width / 2) | (xs[..., 0] > domain[0][1] + x_width / 2) | \
                (xs[..., 1] < domain[1][0] - y_width / 2) | (xs[..., 1] > domain[1][1] + y_width / 2)
            print(np.sum(respawn_mask))
            self.random_vortices.xs[respawn_mask] = self.random_vortices.initial_xs[respawn_mask]

    @property
    def plotting_u(self):   
        if hasattr(self.state, 'plot_all_terms') and self.state.plot_all_terms:
            return self.u
        elif hasattr(self.state, 'plot_boundary_term') and self.state.plot_boundary_term:
            return self.u_boundary_term
        elif hasattr(self.state, 'plot_force_term') and self.state.plot_force_term:
            return self.u_force_term
        
        return self.u
    
    @property
    def omega_u(self):
        if hasattr(self.state, 'use_only_boundary_for_omega') and self.state.use_only_boundary_for_omega:
            return self.u_boundary_term
        if hasattr(self.state, 'use_boundary_for_omega') and self.state.use_boundary_for_omega:
            return self.u
                
        return lambda x, state = None: self.u_main_term(x=x, state=state) + self.u_force_term(x=x, state=state)
