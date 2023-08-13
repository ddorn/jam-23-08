"""
A fast numpy-based particle system.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pprint import pprint
from random import gauss, uniform, expovariate
from typing import TypeVar, TYPE_CHECKING, Callable, Union, Iterable, cast

import numpy as np
import pygame.gfxdraw
from pygame import Vector2

from engine import (
    Vec2Like,
    GFX,
    State,
    from_polar,
    App,
    FixedScreen,
    Object,
    CameraGFX,
    SMALL_FONT,
    text as get_text, wrapped_text,
)

if TYPE_CHECKING:
    # noinspection PyProtectedMember
    from pygame._common import ColorValue

T = TypeVar("T")
CallableOr = Union[Callable[[], T], T]

_DUMMY_SURFACE = pygame.Surface((100, 100))


class Particles:
    """
    A fast numpy-based particle system.

    Implementation notes:
    - All arrays and lists attributes are resized when the capacity is exceeded and reordered when particles die.
    - If you need to store a list/ndarray of values that does not correspond to a particle,
        add it to the IGNORED_BUFFERS
    """

    IGNORED_BUFFERS = ("animations",)

    def __init__(self,
                 *,
                 pos: CallableOr[Vec2Like] = (0, 0),
                 vel: CallableOr[Vec2Like] = (0, 0),
                 acc: CallableOr[Vec2Like] = (0, 0),
                 color: CallableOr[ColorValue] = (255, 255, 255, 255),
                 lifespan: CallableOr[float] = 60,
                 size: CallableOr[float] = 10,
                 draw_sorted_by: 'str' = None,
                 **init_values):
        """Create a new particle system."""
        capacity = 64
        self.draw_sorted_by = draw_sorted_by

        self.n_alive = 0
        self.pos = np.zeros((capacity, 2))
        self.vel = np.zeros((capacity, 2))
        self.acc = np.zeros((capacity, 2))
        self.color = np.zeros((capacity, 4), dtype=np.uint8)
        self.age = np.zeros(capacity, dtype=int)
        self.lifespan = np.ones(capacity, dtype=int)
        self.size = np.zeros(capacity)

        self.init_values = {
            "pos": pos,
            "vel": vel,
            "acc": acc,
            "color": color,
            "lifespan": lifespan,
            "size": size,
            **init_values,
        }

        self.animations = []
        self.to_kill = np.zeros(capacity, dtype=bool)
        self.last_dead_alive_swap = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        """
        A tuple of two arrays, the first one containing the indices of the dead particles to replace,
        the second one containing the indices of the last alive particles to swap with the dead ones.
        All particles attributes have moved with:
            array[dead_to_replace] = array[last_alive]
        This is useful to track is some attributes have changed.
        """

    def __str__(self):
        return f"Particles({self.n_alive} alive, capacity={len(self)})"

    def __len__(self):
        return len(self.pos)

    def new(self, **overrides):
        """Create a new particle.

         Overrides the default values given in the constructor with those passed as arguments.
         """

        if self.n_alive >= len(self):
            self.resize(len(self) * 2)
        index = self.n_alive

        self.age[index] = 0
        to_set = {**self.init_values, **overrides}
        for name, value in to_set.items():
            if callable(value):
                value = value()
            if name == "color":
                value = pygame.Color(value)
            getattr(self, name)[index] = value
        self.n_alive += 1

        return index

    @property
    def alpha(self):
        """Transparency of the particles."""
        return self.color[:, 3]

    @property
    def vel_x(self):
        """X component of the velocity."""
        return self.vel[:, 0]

    @property
    def vel_y(self):
        """Y component of the velocity."""
        return self.vel[:, 1]

    def resize(self, capacity: int):
        """Expand the capacity of the particle system."""
        assert capacity > len(self)
        new_space = capacity - len(self)

        for name, array in self.__dict__.items():
            if name in self.IGNORED_BUFFERS:
                continue
            elif isinstance(array, np.ndarray):
                self.__dict__[name] = np.resize(array, (capacity, *array.shape[1:]))
            elif isinstance(array, list):
                array.extend([None] * new_space)
            else:
                continue

        print(f"Resized {self} to {capacity} particles")

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
            self.last_dead_alive_swap = (np.empty(0, dtype=int), np.empty(0, dtype=int))
            return
        alive_indices = np.nonzero(~self.to_kill[:n])[0]
        n_dead = len(dead_indices)
        n_alive = len(alive_indices)
        assert n_dead + n_alive == n
        # We copy the last alive particles into the dead slots
        dead_to_replace = dead_indices[:n_alive]
        last_alive = alive_indices[-n_dead:]
        for name, array in self.__dict__.items():
            if name in self.IGNORED_BUFFERS:
                continue
            elif isinstance(array, np.ndarray):
                array[:n][dead_to_replace] = array[:n][last_alive]
            elif isinstance(array, list) and len(array) == len(self):
                # The same for the list storages, but we need to do it one by one
                for old, new in zip(dead_to_replace, last_alive):
                    array[old] = array[new]

        self.last_dead_alive_swap = (dead_to_replace, last_alive)
        self.n_alive = n_alive
        # Note: no need to reset to_kill, since it was also reordered

    def draw_order(self) -> Iterable[int]:
        """Iterate over the indices of the particles to draw."""
        if self.draw_sorted_by is None:
            return range(self.n_alive)
        else:
            if self.draw_sorted_by[0] == "-":
                return np.argsort(-getattr(self, self.draw_sorted_by[1:])[:self.n_alive])
            return np.argsort(getattr(self, self.draw_sorted_by)[:self.n_alive])

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

    def interpolate(self, attr: str, times: list[float], values: list[float], clip: bool = True):
        """
        Interpolate an attribute of the particle system.

        Args:
            attr: The name of the attribute to interpolate.
            times: The times at which the values are given. Must be sorted and in [0, 1].
            values: The values to interpolate to. Must be the same length as times.
            clip: Whether to clip the values to the range of `times`. (e.g. if times = [0.5, 1] and values = [0, 1],
                any particle with a life_prop below 0.5 will get the value 0.

        Notes:
            This works only on attributes that are 1D numpy arrays.
        """

        times = np.array(times)
        values = np.array(values)

        def _anim(particles: Particles, life_prop: np.ndarray):
            if clip and (times[0] > 0 or times[1] < 1):
                life_prop = life_prop.clip(times[0], times[-1])
            getattr(particles, attr)[:particles.n_alive] = np.interp(life_prop, times, values)

        return self.add_animation(_anim)

    def lerp(self, attr: str, start_value, end_value, start_time: float = 0.0, end_time: float = 1.0, power: float = 1.0):
        """
        Interpolate an attribute of the particle system between two values.

        Args:
            attr: The name of the attribute to interpolate.
            start_value: The value of the attribute at the start time.
            end_value: The value of the attribute at the end time.
            start_time: The time at which the value is `start_value`.
            end_time: The time at which the value is `end_value`.
            power: The power of the interpolation. 1 = linear, 2 = quadratic, etc.

        Notes:
            This works only on attributes that are numpy arrays, but they can have any number of dimensions.
            Any particle with a life_prop below `start_time` will get the value `start_value` and similarly for
            the end of the range.
        """

        # We use floats, because for types like int8 (color), we might go the wrong way around the modulo.
        start_value = np.array(start_value, dtype=float)
        end_value = np.array(end_value, dtype=float)

        def _anim(particles: Particles, life_prop: np.ndarray):
            if start_time > 0 or end_time < 1:
                life_prop = np.clip(life_prop, start_time, end_time)

            if power != 1:
                life_prop = life_prop ** power

            # Un-squeeze life_prop so that it has one more dimension than start_value
            for _ in range(start_value.ndim):
                life_prop = life_prop[:, None]

            getattr(particles, attr)[:particles.n_alive] = start_value + (end_value - start_value) * life_prop

        return self.add_animation(_anim)

    def add_fade(self):
        """Particles will fade out as they age."""

        def _anim(particles: Particles, life_prop: np.ndarray):
            particles.color[:particles.n_alive, 3] = 255 * (1 - life_prop)

        return self.add_animation(_anim)

    def add_constant_speed(self, speed: Vec2Like):
        """Particles will move at a constant speed."""

        def _anim(particles: Particles, _life_prop: np.ndarray):
            particles.pos[:particles.n_alive] += speed

        return self.add_animation(_anim)

    # Utility functions

    def add_to(self, target: State | Object, z_index: int = None):
        """
        Add the particle system to a state or the state of an object.

        Args:
            target: The state or object to add the particle system to.
            z_index: The z-index of the particle system in the state.
        """
        # Note: this function exists because we don't want Particles to be a subclass of Object

        obj = ParticleObject(self)
        if z_index is not None:
            obj.Z = z_index

        if isinstance(target, State):
            target.add(obj)
        elif isinstance(target, Object):
            if target.state is None:
                # Here the object has not been added to a state yet, so we add it
                # on the next frame
                @target.add_script_decorator
                def _add_later():
                    yield
                    target.state.add(obj)

            else:
                target.state.add(obj)
        else:
            raise TypeError("state must be a State or an Object")
        return self

    def print_one(self, index: int = 0):
        """Print the data of one particle."""
        print("Particle", index)
        d = {
            name: array[index]
            for name, array in self.__dict__.items()
            if name not in self.IGNORED_BUFFERS and isinstance(array, (np.ndarray, list))
        }
        pprint(d)

    def empty(self):
        """Remove all particles."""
        self.n_alive = 0


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
        # Using tolist() at once is faster than unpacking ndarray for each particle
        radius = self.size[:self.n_alive].astype(int).tolist()

        if type(gfx) is CameraGFX and isinstance(gfx, CameraGFX):  # second check is for mypy
            # Shortcircuit for speed
            pos = self.pos[:self.n_alive] + gfx.translation
            pos = pos.astype(int).tolist()
            for idx in self.draw_order():
                pygame.gfxdraw.filled_circle(gfx.surf, *pos[idx], radius[idx], self.color[idx])
            return

        pos = self.pos[:self.n_alive].astype(int).tolist()
        if type(gfx) is GFX:
            # Shortcircuit GFX for speed
            for idx in self.draw_order():
                pygame.gfxdraw.filled_circle(gfx.surf, *pos[idx], radius[idx], self.color[idx])
        else:
            for idx in self.draw_order():
                gfx.circle(self.color[idx], pos[idx], radius)


class SquareParticles(Particles):
    def draw(self, gfx: GFX):
        size = self.size[:self.n_alive].astype(int).tolist()
        pos = (self.pos[:self.n_alive] - self.size[:self.n_alive, None] / 2).astype(int).tolist()

        for idx in self.draw_order():
            s = size[idx]
            gfx.rect(self.color[idx], *pos[idx], s, s)


class ShardParticles(Particles):

    def __init__(self,
                 *,
                 pos: CallableOr[Vec2Like] = (0, 0),
                 vel: CallableOr[Vec2Like] = (0, 0),
                 acc: CallableOr[Vec2Like] = (0, 0),
                 color: CallableOr[ColorValue] = (255, 255, 255, 255),
                 lifespan: CallableOr[float] = 60,
                 size: CallableOr[float] = 10,
                 draw_sorted_by: 'str' = None,
                 tail_length: CallableOr[float] = 1.0,
                 head_length: CallableOr[float] = 3.0,
                 **init_values):
        super().__init__(
            pos=pos, vel=vel, acc=acc, color=color, lifespan=lifespan, size=size,
            draw_sorted_by=draw_sorted_by,
            tail_length=tail_length, head_length=head_length,
            **init_values
        )

        super().__init__(**init_values)
        self.tail_length = np.zeros(len(self))
        self.head_length = np.zeros(len(self))

    def draw(self, gfx: GFX):
        n = self.n_alive
        direction = (self.vel[:n] / np.linalg.norm(self.vel[:n], axis=1)[:, None] *
                     self.size[:n, None])
        cross_dir = np.array([direction[:, 1], -direction[:, 0]]).T

        pos = self.pos[:n]
        points = [
            pos + direction * self.head_length[:n, None],
            pos + cross_dir,
            pos - direction * self.tail_length[:n, None],
            pos - cross_dir,
        ]
        point_list = cast(list[list[tuple[int, int]]], np.stack(points, axis=1).astype(int).tolist())
        color = self.color[:n].tolist()

        for idx in self.draw_order():
            gfx.polygon(color[idx], point_list[idx])


class ImageParticles(Particles):

    def __init__(self,
                 *,
                 pos: CallableOr[Vec2Like] = (0, 0),
                 vel: CallableOr[Vec2Like] = (0, 0),
                 acc: CallableOr[Vec2Like] = (0, 0),
                 color: CallableOr[ColorValue] = (255, 255, 255, 255),
                 lifespan: CallableOr[float] = 60,
                 size: CallableOr[float] = 10,
                 draw_sorted_by: 'str' = None,
                 surf: CallableOr[pygame.Surface] = None,
                 **init_values):
        super().__init__(
            pos=pos, vel=vel, acc=acc, color=color, lifespan=lifespan, size=size,
            draw_sorted_by=draw_sorted_by,
            _original_surf=surf,
            **init_values
        )

        self._original_surf: list[pygame.Surface] = [_DUMMY_SURFACE] * len(self)
        self.surf: list[pygame.Surface] = [_DUMMY_SURFACE] * len(self)

    def new(self, **overrides):
        """Create a new particle."""
        if "surf" in overrides:
            overrides["_original_surf"] = overrides.pop("surf")
        index = super().new()
        if self._original_surf[index] is None:
            raise ValueError("'surf' must be provided")
        self.surf[index] = self._original_surf[index].copy()
        return index

    def redraw(self, index: int):
        """Redraw the particle at the given index."""
        w, h = self._original_surf[index].get_size()
        ratio = self.size[index] / min(w, h)
        surf = pygame.transform.smoothscale(self._original_surf[index],
                                            (int(w * ratio), int(h * ratio)))
        surf.set_alpha(self.color[index, 3])
        self.surf[index] = surf

    def logic(self):
        last_size = self.size[:self.n_alive].copy()
        super().logic()
        # We redraw only the particles whose size changed
        for i in np.nonzero(last_size != self.size[:self.n_alive])[0]:
            self.redraw(i)

    def draw(self, gfx: GFX):
        pos = self.pos[:self.n_alive].astype(int).tolist()
        for i in self.draw_order():
            self.surf[i].set_alpha(self.color[i, 3])
            gfx.blit(self.surf[i], center=pos[i])


class TextParticles(Particles):

    def __init__(self,
                 *,
                 pos: CallableOr[Vec2Like] = (0, 0),
                 vel: CallableOr[Vec2Like] = (0, 0),
                 acc: CallableOr[Vec2Like] = (0, 0),
                 color: CallableOr[ColorValue] = (255, 255, 255, 255),
                 lifespan: CallableOr[float] = 60,
                 size: CallableOr[float] = 10,
                 draw_sorted_by: 'str' = None,
                 text: CallableOr[str] = "",
                 font_name: str = SMALL_FONT,
                 anchor: str = "center",
                 **init_values):
        """
        Create a text particle system.

        Args:
            font_name: The font to use for the particles, shared by all of them.
            anchor: The anchor point of the text, shared by all of them.
        """
        super().__init__(pos=pos, vel=vel, acc=acc, color=color, lifespan=lifespan, size=size,
                         draw_sorted_by=draw_sorted_by,
                         text=text,
                         **init_values)
        self.font_name = font_name
        self.anchor = anchor
        self.text: list[str] = ["h7Y4dR"] * len(self)  # Placeholder to notice if something goes wrong
        self.surf: list[pygame.Surface] = [_DUMMY_SURFACE] * len(self)

    # noinspection PyShadowingNames
    def new(self, **overrides):
        index = super().new(**overrides)
        self.redraw(index)

    def redraw(self, index: int):
        """Redraw the particle at the given index."""
        self.surf[index] = get_text(
            self.text[index],
            int(self.size[index]),
            tuple(self.color[index]),
            self.font_name,
        )

    def logic(self):
        last_size = self.size[:self.n_alive].copy()
        super().logic()
        # We redraw only the particles whose size changed
        dead_to_replace, last_alive = self.last_dead_alive_swap
        last_size[dead_to_replace] = last_size[last_alive]
        changed_size = last_size[:self.n_alive] != self.size[:self.n_alive]
        for i in np.nonzero(changed_size)[0]:
            self.redraw(i)

    def draw(self, gfx: GFX):
        pos = self.pos[:self.n_alive].astype(int).tolist()
        for i in self.draw_order():
            surf = self.surf[i]
            surf.set_alpha(self.color[i, 3])
            gfx.blit(self.surf[i], **{self.anchor: pos[i]})


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

    def __add__(self, other):
        """Return the sum of two distributions."""
        return SumDistribution(self, other)


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
class SumDistribution(Distribution):
    """A sum of two distributions."""

    a: Distribution
    b: Distribution

    def __call__(self):
        a = call_if_needed(self.a)
        b = call_if_needed(self.b)
        return a + b


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


class ImagePosDistribution(Distribution):
    """A position sampled from the non-transparent pixels of an image."""

    def __init__(self, image: pygame.Surface, **anchor: Vec2Like):
        self.image = image
        self.anchor = anchor

        self.non_transparent = np.argwhere(pygame.surfarray.pixels_alpha(self.image) != 0)
        self.n_pixels = len(self.non_transparent)
        self.offset = self.image.get_rect(**self.anchor).topleft

        assert self.n_pixels > 0, "Text must have at least one non-transparent pixel"

    def __call__(self):
        i = random.randrange(0, self.n_pixels)
        return self.non_transparent[i] + self.offset


__all__ = [
    "Particles",
    "CircleParticles",
    "ShardParticles",
    "ImageParticles",
    "TextParticles",
    "Gauss",
    "Uniform",
    "Exp",
    "Polar",
    "Distribution",
    "JointDistribution",
]

if __name__ == "__main__":
    SIZE = (1300, 800)
    from pygame import Color

    class Demo(State):
        # BG_COLOR = "#22c1c3"
        BG_COLOR = "#FAFAFA"

        def __init__(self):
            super().__init__()
            image = wrapped_text("Joyeux anniversaire Esther!", 110, "#ffffff", SIZE[0])

            self.p_generator = (
                CircleParticles(
                    pos=ImagePosDistribution(image, center=Vector2(SIZE) / 2),
                    size=3,
                    lifespan=60,
                    acc=Polar(0.02, Uniform.angle()),
                    draw_sorted_by='-age',
                )
                # Interpolate between #fdbb2d and #22c1c3
                # .lerp('color', Color("#ff9966"), Color("#ff5e62"), power=1)
                .interpolate('alpha', [0, 0.05, 0.4, 1], [0, 255, 5, 0])
                .interpolate('size', [0.3, 0.6, 1], [8, 15, 100])
                .add_to(self)
            )

        def logic(self):
            super().logic()
            for _ in range(100):
                idx = self.p_generator.new(
                    # vel=from_polar(gauss(1, 0.3), gauss(-20, 4))
                )
                x = self.p_generator.pos[idx, 0]
                self.p_generator.color[idx] = Color("#ff9966").lerp(Color("#ff5e62"), x / SIZE[0])
            self.debug.text(self.p_generator)


    app = App(Demo, FixedScreen(SIZE))
    app.USE_FPS_TITLE = True
    app.run()
