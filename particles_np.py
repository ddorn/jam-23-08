"""
A fast numpy-based particle system.
"""

from __future__ import annotations

from random import gauss, uniform
from typing import TypeVar, TYPE_CHECKING, Callable

import numpy as np
from pygame import Color

from engine import Vec2Like, GFX, State, from_polar, random_rainbow_color, App, FixedScreen, Object

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from pygame._common import ColorValue


class Particles:
    """A fast numpy-based particle system."""

    def __init__(self, capacity: int = 100, **extra_buffers: type):
        """
        Create a new particle system.

        Args:
            capacity: The initial capacity of the particle system.
            **extra_buffers: Extra buffers to add to the particle system.
                The key is the name of the buffer, and the value is the numpy dtype.
                This can be used to create custom animations with custom data without subclassing.
        """

        self.pos = np.zeros((capacity, 2))
        self.vel = np.zeros((capacity, 2))
        self.acc = np.zeros((capacity, 2))
        self.color = np.zeros((capacity, 4), dtype=np.uint8)
        self.age = np.zeros(capacity, dtype=np.uint8)
        self.lifespan = np.ones(capacity)
        self.n_alive = 0
        self.size = np.zeros(capacity)

        self.extra_buffers = extra_buffers
        for name, dtype in extra_buffers.items():
            setattr(self, name, np.zeros(capacity, dtype=dtype))

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
            lifespan: float = 100,
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

        return index

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

    def add_fade(self):
        """
        Particles will fade out as they age.
        """

        def _anim(particles: Particles, life_prop: np.ndarray):
            particles.color[:particles.n_alive, 3] = 255 * (1 - life_prop)

        return self.add_animation(_anim)

    # Utility functions

    def add_to_state(self, state: State):
        """Add the particle system to a state."""
        # We create an object because we don't want Particles to be a subclass of Object
        state.add(ParticleObject(self))
        return self


class ParticleObject(Object):
    """A simple wrapper around a particle system to make it an object."""

    def __init__(self, particles: Particles):
        super().__init__((0, 0))
        self.particles = particles

    def logic(self):
        super().logic()
        self.particles.logic()

    def draw(self, gfx: GFX):
        super().draw(gfx)
        self.particles.draw(gfx)


class CircleParticles(Particles):
    def draw(self, gfx: GFX):
        for i in range(self.n_alive):
            gfx.circle(self.color[i], self.pos[i], self.size[i])


class ShardParticles(Particles):
    def __init__(self, capacity: int = 100, **extra_buffers: type):
        super().__init__(capacity, **extra_buffers)
        self.tail_length = np.zeros(capacity)
        self.head_length = np.zeros(capacity)

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
        points = np.stack(points, axis=1)
        for ps, color in zip(points, self.color[:n]):
            gfx.polygon(color, ps)


if __name__ == "__main__":
    SIZE = (1300, 800)
    T = TypeVar("T", bound=Particles)


    class Demo(State):
        BG_COLOR = (0, 0, 49)

        def __init__(self):
            super().__init__()
            self.shard = CircleParticles().add_fade().add_to_state(self)

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
