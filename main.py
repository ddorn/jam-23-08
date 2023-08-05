from __future__ import annotations

import math
import sys
from random import gauss
from time import time

import numpy as np
import pygame.gfxdraw
from pygame import Rect

sys.path.append("engine")
from engine import *
from utility import smooth_breathing

SIZE = W, H = 1600, 1200
SCREEN = Rect(0, 0, *SIZE)
PINK = pygame.Color("#E91E63")
ORANGE = pygame.Color("#F39C12")
INDIGO = pygame.Color("#3F51B5")
DARK = pygame.Color("#2C3E50")
WHITE = pygame.Color("#ECF0F1")
BLACK = pygame.Color("#212121")


def fainter(color, multiplier):
    color = pygame.Color(*color)
    h, s, l, a = color.hsla
    color.hsla = (
        h,
        s,
        l * multiplier + 100 * (1 - multiplier),
        a * multiplier ** 2 / 2,
    )
    return color


class Blob(Object):
    Z = 10

    def __init__(
            self,
            x,
            y,
            size=20,
            part_radius=5,
            points=5,
            controllable=False,
            hue_range: tuple[int, int] | None = None,
            hist_size: int = 50,
            accelration: float = 5,
            friction: float = 0.8,
    ):
        super().__init__((x, y))

        self.radius = size
        self.part_radius = part_radius
        self.acceleration = accelration
        self.friction = friction
        self.n_points = points
        self.hue_range = hue_range
        self.time = 0
        self.std = 0.03
        self.point_dist = np.random.normal(1, self.std, self.radius)
        self.point_angle_shift = np.random.normal(0, self.std, self.radius)
        self.point_radius = np.random.normal(1, self.std, self.radius)
        self.controllable = controllable

        self.hist_size = hist_size
        self.points_to_draw = []

    def color(self, point_index: int):
        color = pygame.color.Color(0)
        angle = point_index / self.n_points * 36 + self.time / 10
        angle = angle % 36 / 36  # between 0 and 1, looping
        if self.hue_range is not None:
            angle = math.sin(angle * math.pi)
            angle = self.hue_range[0] + angle * (self.hue_range[1] - self.hue_range[0])
        else:
            angle *= 360

        color.hsva = (angle, 70, 90, 100)
        return color

    def mk_points(self):
        points = []
        angle = math.pi * 2 / self.n_points
        for i in range(self.n_points):
            angle_i = angle * i + self.time / 10 + self.point_angle_shift[i]
            point = (self.pos + pygame.Vector2(
                math.cos(angle_i),
                math.sin(angle_i),
            ) * self.point_dist[i] * self.radius)
            radius = max(2, int(self.point_radius[i] * self.part_radius))
            points.append([point, self.color(i), radius])
        return points

    def logic(self):
        last_points = self.mk_points()
        super().logic()
        self.time += 1
        alpha = 0.02

        self.point_dist = alpha + (1 - alpha) * (self.point_dist +
                                                 np.random.normal(0, self.std, self.radius))
        self.point_angle_shift = (1 - alpha) * (self.point_angle_shift +
                                                np.random.normal(0, self.std * 2, self.radius))
        self.point_radius = alpha + (1 - alpha) * (self.point_radius +
                                                   np.random.normal(0, self.std * 2, self.radius))

        # Move
        if self.controllable:
            keys = pygame.key.get_pressed()
            self.vel.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.acceleration
            self.vel.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * self.acceleration
            self.vel *= self.friction

            self.pos.y += 5

        # Add enough circle between the last and the current position that it appears a smooth line
        for (p1, color, radius), (p2, _, _) in zip(last_points, self.mk_points()):
            dist = p1.distance_to(p2)
            direction = (p2 - p1).normalize()
            for j in range(0, int(dist), max(1, radius // 3)):
                point = p1 + direction * j
                self.points_to_draw.append((point, color, radius, (self.time, j)))

        # Remove old points
        self.points_to_draw.sort(key=lambda x: x[3])
        self.points_to_draw = [p for p in self.points_to_draw if self.time - p[3][0] < self.hist_size]

        # Wrap around the screen. We do this last so that we don't have lines going across the screen
        if not (0 < self.pos.x < W):
            self.pos.x %= W
        if not (0 < self.pos.y < H):
            self.pos.y %= H

    def draw(self, gfx: GFX, force_alpha: float | None = None):
        super().draw(gfx)
        for point, color, radius, (t, _) in reversed(self.points_to_draw):
            if force_alpha is not None:
                multiplier = force_alpha
            else:
                multiplier = 1 - (self.time - t) / self.hist_size
            pygame.gfxdraw.filled_circle(
                gfx.surf,
                int(point.x),
                int(point.y),
                radius,
                fainter(color, multiplier),
            )


class Fairy(Object):
    def __init__(self, pos):
        super().__init__(pos)

    def logic(self):
        super().logic()


class GameState(State):
    BG_COLOR = None
    FPS = 60

    def __init__(self):
        super().__init__()
        self.fog = pygame.Surface(SIZE)
        self.fog.fill(WHITE)

        self.blob = self.add(Blob(
            W / 2,
            H / 2,
            10,
            6,
            6,
            True,
            accelration=3,
            friction=0.8,
            hist_size=100,
        ))
        self.use_fog = True

        for i in range(10):
            self.add(Fairy(random_in_rect(Rect(0, 0, W, H))))

    def handle_events(self, events):
        super().handle_events(events)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.use_fog = not self.use_fog

    def logic(self):
        super().logic()

        if self.timer == 20:
            self.particles.fountains.append(ParticleFountain.screen_confetti(SCREEN))

    def draw(self, gfx: "GFX"):
        bg_color = gradient(self.timer / self.FPS,
                            (0, "#E1F5FE"),
                            (5, "#9FA8DA"),
                            (10, DARK),
                            (15, BLACK),
                            (25, (0, 0, 0)),
                            )
        gfx.surf.fill(bg_color)

        super().draw(gfx)

        s = smooth_breathing(time())
        s256 = int(s * 255)

        if self.use_fog:
            self.blob.draw(GFX(self.fog), force_alpha=0.3)
            self.fog.set_alpha(s256)
            gfx.surf.blit(self.fog, (0, 0))


def main():
    App(GameState, FixedScreen(SIZE)).run()


if __name__ == "__main__":
    main()
