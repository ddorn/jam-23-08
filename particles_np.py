"""
A fast numpy-based particle system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pprint
from random import gauss, uniform, expovariate
from typing import TypeVar, TYPE_CHECKING, Callable

import numpy as np
import pygame.gfxdraw
from pygame import Color, Vector2

from engine import Vec2Like, GFX, State, from_polar, random_rainbow_color, App, FixedScreen, Object, CameraGFX

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from pygame._common import ColorValue


class Particles:
    """A fast numpy-based particle system."""

    def __init__(self, **init_values):
        """Create a new particle system."""
        capacity = 64

        self.pos = np.zeros((capacity, 2))
        self.vel = np.zeros((capacity, 2))
        self.acc = np.zeros((capacity, 2))
        self.color = np.zeros((capacity, 4), dtype=np.uint8)
        self.age = np.zeros(capacity, dtype=np.uint8)
        self.lifespan = np.ones(capacity)
        self.n_alive = 0
        self.size = np.zeros(capacity)

        self.init_values = init_values

        self.to_kill = np.zeros(capacity, dtype=bool)
        self.animations = []

    def __str__(self):
        return f"Particles({self.n_alive} alive, capacity={len(self)})"

    def __len__(self):
        return len(self.pos)

    def new(self,
            pos: Vec2Like,
            vel: Vec2Like = (0, 0),
            acc: Vec2Like = (0, 0),
            color: ColorValue = (255, 255, 255, 255),
            lifespan: float = 60,
            size: float = 1.0,
            **extra_data
            ):
        """Create a new particle."""
        if self.n_alive >= len(self):
            self.resize(len(self) * 2)
        index = self.n_alive

        self.n_alive += 1
        self.pos[index] = pos
        self.vel[index] = vel
        self.acc[index] = acc
        self.color[index] = Color(color)
        self.age[index] = 0
        self.lifespan[index] = lifespan
        self.size[index] = size
        for name, value in extra_data.items():
            getattr(self, name)[index] = value

        for name, value in self.init_values.items():
            if callable(value):
                value = value()
            getattr(self, name)[index] = value

        return index

    @property
    def alpha(self):
        """Transparency of the particles."""
        return self.color[:, 3]

    def resize(self, capacity: int):
        """Expand the capacity of the particle system."""
        assert capacity > len(self)

        for name, array in self.__dict__.items():
            if isinstance(array, np.ndarray):
                self.__dict__[name] = np.resize(array, (capacity, *array.shape[1:]))
        print("Expanded to", capacity)

    def logic(self):
        """Update all the particles."""
        n: int = self.n_alive
        self.vel[:n] += self.acc[:n]
        self.pos[:n] += self.vel[:n]
        self.age[:n] += 1

        life_prop = self.age[:n] / self.lifespan[:n]
        life_prop.clip(0, 1, out=life_prop)
        for animation in self.animations:
            animation(self, life_prop)

        self.to_kill[:n] |= self.age[:n] >= self.lifespan[:n]
        # Reorder the arrays so that dead particles are at the end
        # and alive particles are at the beginning
        self.clean_the_dead()

    def clean_the_dead(self):
        """
        Reorder the arrays so that dead particles are at the end.
        """
        n = self.n_alive
        dead_indices = np.nonzero(self.to_kill[:n])[0]
        if len(dead_indices) == 0:
            return
        alive_indices = np.nonzero(~self.to_kill[:n])[0]
        n_dead = len(dead_indices)
        n_alive = len(alive_indices)
        assert n_dead + n_alive == n
        # We copy the last alive particles into the dead slots
        last_alive = alive_indices[-n_dead:]
        dead_to_replace = dead_indices[:n_alive]
        for array in self.__dict__.values():
            if isinstance(array, np.ndarray):
                array[dead_to_replace] = array[last_alive]
        self.n_alive = n_alive
        # Note: no need to reset to_kill, since it was also reordered

    def draw(self, gfx: GFX):
        """Draw all the particles."""
        raise NotImplementedError

    # Animation functions
    def add_animation(self, animation: Callable[[Particles, np.ndarray], None]):
        """
        Add an animation to the particle system.

        Args:
            animation: A function (particles, life_prop) => None that takes the particle system and
                the proportion of life remaining (0 = just born, 1 = about to die) as arguments.
                Note: life_prop is a numpy array of shape (n_alive,).
        """
        self.animations.append(animation)
        return self

    def animate(self, attr: str, easing: Callable[[np.ndarray], None]):
        """
        Animate an attribute of the particle system.

        Args:
            attr: The name of the attribute to animate.
            easing: A function that takes the proportion of life remaining (0 = just born, 1 = about to die)
                and returns a new value for the attribute.
                Note: life_prop is a numpy array of shape (n_alive,).
        """

        def _anim(particles: Particles, life_prop: np.ndarray):
            getattr(particles, attr)[:particles.n_alive] = easing(life_prop)

        return self.add_animation(_anim)

    def add_fade(self):
        """Particles will fade out as they age."""

        def _anim(particles: Particles, life_prop: np.ndarray):
            particles.color[:particles.n_alive, 3] = 255 * (1 - life_prop)

        return self.add_animation(_anim)

    def add_constant_speed(self, speed: Vec2Like):
        """Particles will move at a constant speed."""

        def _anim(particles: Particles, life_prop: np.ndarray):
            particles.pos[:particles.n_alive] += speed

        return self.add_animation(_anim)

    # Utility functions

    def add_to(self, target: State | Object):
        """Add the particle system to a state or the state of an object."""
        # Note: this function exists because we don't want Particles to be a subclass of Object

        if isinstance(target, State):
            target.add(ParticleObject(self))
        elif isinstance(target, Object):
            if target.state is None:
                # Here the object has not been added to a state yet, so we add it
                # on the next frame
                @target.add_script_decorator
                def _add_later():
                    yield
                    target.state.add(ParticleObject(self))
            else:
                target.state.add(ParticleObject(self))
        else:
            raise TypeError("state must be a State or an Object")
        return self

    def print_one(self, index: int = 0):
        """Print the data of one particle."""
        print("Particle", index)
        d = {
            name: getattr(self, name)[index]
            for name in self.__dict__
            if isinstance(getattr(self, name), np.ndarray)
        }
        pprint(d)


class ParticleObject(Object):
    """A simple wrapper around a particle system to make it an object."""

    def __init__(self, particles: Particles):
        super().__init__((0, 0))
        self.particles = particles

    def draw(self, gfx: GFX):
        super().draw(gfx)
        # We draw and do the logic at once, because particles can be added at any time
        # by any object, and we want to guarantee that logic() was called at least once
        # before draw() so that the animations work properly.
        self.particles.logic()
        self.particles.draw(gfx)


class CircleParticles(Particles):
    def draw(self, gfx: GFX):
        radius = self.size[:self.n_alive].astype(int).tolist()

        if isinstance(gfx, CameraGFX):
            # Shortcircuit for speed
            pos = self.pos[:self.n_alive] + gfx.translation
            pos = pos.astype(int).tolist()
            for color, pos, radius in zip(self.color[:self.n_alive], pos, radius):
                pygame.gfxdraw.filled_circle(gfx.surf, *pos, radius, color)
            return

        pos = self.pos[:self.n_alive].astype(int).tolist()
        for i in range(self.n_alive):
            gfx.circle(self.color[i], Vector2(pos[i]), radius[i])


class ShardParticles(Particles):

    def __init__(self, **init_values):
        super().__init__(**init_values)
        self.tail_length = np.zeros(len(self))
        self.head_length = np.zeros(len(self))

    def new(self,
            pos: Vec2Like,
            vel: Vec2Like = (0, 0),
            acc: Vec2Like = (0, 0),
            color: ColorValue = (255, 255, 255, 255),
            lifespan: float = 100,
            size: float = 1.0,
            tail_length: float = 1.0,
            head_length: float = 3.0,
            **extra_data
            ):
        return super().new(pos, vel, acc, color, lifespan, size,
                           tail_length=tail_length, head_length=head_length,
                           **extra_data)

    def draw(self, gfx: GFX):
        n = self.n_alive
        direction = self.vel[:n] / np.linalg.norm(self.vel[:n], axis=1)[:, None] * self.size[:n, None]
        cross_dir = np.array([direction[:, 1], -direction[:, 0]]).T

        pos = self.pos[:n]
        points = [
            pos + direction * self.head_length[:n, None],
            pos + cross_dir,
            pos - direction * self.tail_length[:n, None],
            pos - cross_dir,
        ]
        points = np.stack(points, axis=1).astype(int).tolist()
        for ps, color in zip(points, self.color[:n].tolist()):
            gfx.polygon(color, ps)


def call_if_needed(x):
    """Call x if it is a function, otherwise return x."""
    return x() if callable(x) else x


@dataclass
class Distribution:
    """A distribution of values."""

    def __call__(self):
        raise NotImplementedError

    # Overload &
    def __and__(self, other: "Distribution" | object):
        """Combine two distributions into a joint distribution."""
        return JointDistribution(self, other)


@dataclass
class JointDistribution(Distribution):
    """A joint distribution of values."""
    distributions: list[Distribution]

    def __init__(self, *distributions: Distribution | object):
        # Flatten nested joint distributions
        self.distributions = []
        for d in distributions:
            if isinstance(d, JointDistribution):
                self.distributions.extend(d.distributions)
            else:
                self.distributions.append(d)

    def __call__(self):
        return tuple(call_if_needed(d) for d in self.distributions)


@dataclass
class Gauss(Distribution):
    """A Gaussian distribution."""
    mean: float
    std: float

    def __call__(self):
        return gauss(self.mean, self.std)


@dataclass
class Uniform(Distribution):
    """A uniform distribution."""
    a: float
    b: float

    def __call__(self):
        return uniform(self.a, self.b)

    @classmethod
    def angle(cls):
        """A uniform distribution of angles in degrees."""
        return cls(0, 360)


@dataclass
class Exp(Distribution):
    """An exponential distribution."""
    rate: float

    def __call__(self):
        return expovariate(self.rate)


@dataclass
class Polar(Distribution):
    """A polar distribution."""
    r: float | Distribution
    theta: float | Distribution
    """The angle in degrees."""

    def __call__(self):
        return from_polar(call_if_needed(self.r), call_if_needed(self.theta))


__all__ = [
    "Particles",
    "CircleParticles",
    "ShardParticles",
    "Gauss",
    "Uniform",
    "Exp",
    "Polar",
    "Distribution",
    "JointDistribution",
]

if __name__ == "__main__":
    SIZE = (1300, 800)
    T = TypeVar("T", bound=Particles)


    class Demo(State):
        BG_COLOR = (0, 0, 49)

        def __init__(self):
            super().__init__()
            self.shard = CircleParticles().add_fade().add_to(self)

        def logic(self):
            super().logic()
            for _ in range(100):
                self.shard.new(
                    pos=(SIZE[0] / 2, SIZE[1] / 2),
                    vel=from_polar(gauss(4, 0.5), uniform(0, 360)),
                    color=random_rainbow_color(70, 90),
                    size=10,
                )
            self.debug.text(self.shard)


    app = App(Demo, FixedScreen(SIZE))
    app.USE_FPS_TITLE = True
    app.run()
