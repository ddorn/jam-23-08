from __future__ import annotations

import math
from random import gauss, random, randrange, uniform
from time import time

import numpy as np
import pygame.gfxdraw
from pygame import Rect

from engine import *
from utility import smooth_breathing

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
            max_hist_size: int = 100,
            acceleration: float = 5,
            friction: float = 0.8,
    ):

        hit_size = size * 2 + part_radius * 2
        super().__init__((x, y), (hit_size, hit_size))

        self.radius = size
        self.part_radius = part_radius
        self.acceleration = acceleration
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
        self.max_hist_size = max_hist_size
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
            point = (self.center + pygame.Vector2(
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
            for j in range(0, int(dist), max(3, radius // 3)):
                point = p1 + direction * j
                self.points_to_draw.append((point, color, radius, (self.time, j)))

        # Remove old points
        self.points_to_draw.sort(key=lambda x: x[3])
        self.points_to_draw = [p for p in self.points_to_draw if self.time - p[3][0] < self.hist_size]

        # Check collision with fairies
        fairy: Fairy
        for fairy in self.state.get_all(Fairy):
            if fairy.pos.distance_to(self.pos) < fairy.radius + self.radius + self.part_radius:
                fairy.alive = False
                self.hist_size += 3
                for i in range(self.n_points):
                    angle = i * 360 / self.n_points
                    color = from_hsv(angle, 70, 90)
                    self.state.particles.add(
                        ShardParticle(color)
                        .builder()
                        .at(self.center + from_polar(self.radius, angle), angle)
                        .velocity(8)
                        .anim_fade(0.2)
                        .build()
                    )

                self.add_speed_buff(fairy.hue)

                for _ in rrange(1.5):
                    self.state.add(Fairy(random_in_rect(SCREEN.move(self.pos - (W / 2, H / 2)))))

        self.hist_size = min(self.hist_size, self.max_hist_size)
        self.state.debug.text(self.pos)

    def add_speed_buff(self, angle: float):
        @self.add_script_decorator
        def speed_buff():
            for _ in range(10):
                self.vel = from_polar(30, angle)
                yield

    def draw(self, gfx: GFX, force_alpha: float | None = None):
        super().draw(gfx)
        for point, color, radius, (t, _) in reversed(self.points_to_draw):
            if force_alpha is not None:
                multiplier = force_alpha
            else:
                multiplier = 1 - (self.time - t) / self.hist_size

            gfx.circle(fainter(color, multiplier), point, radius)


class Fairy(SpriteObject):
    radius = 40  # For collision detection

    def __init__(self, pos):
        self.hue = randrange(0, 360, 60)
        img = pygame.Surface((80, 80), pygame.SRCALPHA)
        super().__init__(pos, img, )

        # Add a new circle on the image
        for _ in range(256):
            x = gauss(40, 6)
            y = gauss(40, 6)
            radius = max(2.0, gauss(5, 2))
            color = from_hsv(gauss(self.hue, 10), 70, 90)
            pygame.draw.circle(self.image, color, (x, y), radius)

            # Fade out the image
            self.image.fill((0, 0, 0, 1), None, pygame.BLEND_RGBA_SUB)

    def logic(self):
        super().logic()

        self.pos.y += 0

    def draw(self, gfx: "GFX"):
        super().draw(gfx)


class GameState(State):
    BG_COLOR = None
    FPS = 60

    def __init__(self):
        super().__init__()
        self.fog = pygame.Surface(SIZE)
        self.fog.fill(WHITE)

        self.blob = self.add(Blob(
            0,
            0,
            10,
            6,
            6,
            True,
            acceleration=3,
            friction=0.8,
            hist_size=10,
            max_hist_size=50,
        ))
        self.use_fog = True
        self.debug.toggle()

        for i in range(10):
            self.add(Fairy(random_in_rect(Rect(0, 0, W, H))))

    def handle_events(self, events):
        super().handle_events(events)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.use_fog = not self.use_fog
                elif event.key == pygame.K_f:
                    self.FPS = 1000 if self.FPS == 60 else 60

    def logic(self):
        super().logic()

        # Create new fairies below
        if random() < 0.01:
            self.add(Fairy(random_in_rect(SCREEN.move(self.blob.pos - (W / 2, -H / 2)))))

        # De-pop objects that are too far away if we have too many
        fairies = list(self.get_all(Fairy))
        if len(fairies) > 100:
            sorted_objects = sorted(fairies, key=lambda o: o.pos.distance_to(self.blob.pos))
            for o in sorted_objects[100:]:
                o.alive = False

        # if self.timer == 20 * self.FPS:
        #     self.particles.fountains.append(ParticleFountain.screen_confetti(SCREEN))

    def draw(self, gfx: CameraGFX):
        camera_x, camera_y = self.blob.pos
        # Clamp the camera y between 0m and 1000m, but smoothly
        camera_y = soft_clamp(camera_y, 0, 1000 * 100, 1000)
        gfx.world_center = camera_x, camera_y

        bg_color = gradient(self.timer / self.FPS,
                            (0, "#3F51B5"),
                            # (5, "#311B92"),
                            (10, DARK),
                            (15, BLACK),
                            (25, (0, 0, 0)),
                            (30, "#430727"),
                            (35, "#601B06"),
                            (40, "#2E2B08"),
                            (45, "#002620"),
                            )
        gfx.surf.fill(bg_color)

        if self.timer % 5 == 0:
            self.blob.draw(WrapGFX(self.fog), force_alpha=0.3)

        # If at the surcface, draw the sky
        if camera_y < H / 2:
            gfx.rect(WHITE, camera_x - W / 2, -H / 2, W, H / 2)
            # Generate waves particles
            self.generate_waves_particles(0, bg_color.hsva[0])
        if camera_y > 1000 * 100 - H / 2:
            gfx.rect(BLACK, camera_x - W / 2, 1000 * 100, W, H / 2)
            self.generate_waves_particles(1000 * 100, bg_color.hsva[0])

        super().draw(gfx)

        # Show depth
        depth = int(-self.blob.pos.y / 100)
        gfx.text("#FFF176", f"{depth}m", size=120, midbottom=(W / 2, H - 20))

        if self.use_fog:
            self.fog.set_alpha(50)
            gfx.surf.blit(self.fog, (0, 0))

    def generate_waves_particles(self, y: float, hue: float):
        n_particles = 20
        for i in range(n_particles):
            x = uniform(-W / 2, W / 2) + self.blob.pos.x
            self.particles.add(
                CircleParticle("#ffffff")
                .builder()
                .hsv(gauss(hue, 10), gauss(0.2, 0.05), 0.95)
                .sized(gauss(10, 2))
                .anim_fade()
                .at((x, y), gauss(-90, 15))
                .velocity(gauss(1, 0.5))
                .constant_force((0, 1))
                .build()
            )


def main():
    App(GameState, FixedScreen(SIZE), CameraGFX).run()


if __name__ == "__main__":
    print(gfx)
    main()
