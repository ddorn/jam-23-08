from __future__ import annotations

import math
from random import gauss, random, randrange, uniform, choice, expovariate

import numpy as np
import pygame.gfxdraw
from pygame import Rect, Vector2

from engine import *

PINK = pygame.Color("#E91E63")
ORANGE = pygame.Color("#F39C12")
INDIGO = pygame.Color("#3F51B5")
DARK = pygame.Color("#2C3E50")
WHITE = pygame.Color("#ECF0F1")
BLACK = pygame.Color("#212121")

Y_TO_METERS = 1 / 100
METERS_TO_Y = 100
# Y of the surface
SURFACE = 0
# Y of the other side
OTHER_SIDE = 100 * METERS_TO_Y


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
        last_pos = self.pos.copy()
        super().logic()
        self.time += 1
        alpha = 0.02

        # Add some randomness to the points
        self.point_dist = alpha + (1 - alpha) * (self.point_dist +
                                                 np.random.normal(0, self.std, self.radius))
        self.point_angle_shift = (1 - alpha) * (self.point_angle_shift +
                                                np.random.normal(0, self.std * 2, self.radius))
        self.point_radius = alpha + (1 - alpha) * (self.point_radius +
                                                   np.random.normal(0, self.std * 2, self.radius))

        # Move
        if self.pos.x < (SURFACE + OTHER_SIDE) / 2:
            m = chrange(self.pos.y, (SURFACE, SURFACE + 800), (1, 4))
        else:
            m = chrange(self.pos.y, (OTHER_SIDE - 800, OTHER_SIDE), (4, -1))
        gravity = 0.3 * soft_clamp(m, 1, 4, 1)
        self.vel.y += gravity
        if self.pos.y < SURFACE:
            self.vel *= 0.99
        elif self.pos.y > OTHER_SIDE:
            self.vel *= 0.99
        else:
            if self.controllable:
                keys = pygame.key.get_pressed()
                self.vel.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.acceleration
                self.vel.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * self.acceleration
            self.vel *= self.friction

        # If entering/exiting water, make a splash
        self.make_splash(last_pos)

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

    def add_speed_buff(self, angle: float):
        @self.add_script_decorator
        def speed_buff():
            for _ in range(10):
                self.vel = from_polar(30, angle)
                yield

    def make_splash(self, last_pos: pygame.Vector2):
        mini = min(self.pos.y, last_pos.y)
        maxi = max(self.pos.y, last_pos.y)
        if mini < SURFACE < maxi:
            y = SURFACE
        elif mini < OTHER_SIDE < maxi:
            y = OTHER_SIDE
        else:
            return

        if self.vel.y < 0:
            m = 1
            pull = self.vel / 5
        else:
            m = -1
            pull = (0, 0.1)

        angle = self.vel.as_polar()[1] + 180 * (self.vel.y > 0)

        strength = self.vel.length() * 5
        self.state.debug.text(f"strength: {strength}")
        # color = self.state.bg_color()
        color= self.color(0)

        impact_pos = Vector2(self.center.x, y)

        @self.add_script_decorator
        def _():
            for frame in range(1, 3):
                yield
                for _ in rrange(strength):
                    self.state.particles.add(
                        CircleParticle(color)
                        .builder()
                        .sized(size := gauss(10, 2))
                        .at(impact_pos + (gauss(0, 10), 0), gauss(angle, 30))
                        .constant_force(pull)
                        .living(30)
                        # .velocity(gauss(2, 0.4))
                        .velocity(expovariate(0.7))
                        # .acceleration(-0.1)
                        .anim_gravity(0.05)
                        .anim_fade(0.2)
                        .build()
                    )



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
            pygame.draw.circle(self.image, self.color(), (x, y), radius)

            # Fade out the image
            self.image.fill((0, 0, 0, 1), None, pygame.BLEND_RGBA_SUB)

    def logic(self):
        super().logic()

        self.pos.y += 0

        # self.state.particles.add(
        #     CircleParticle(self.color())
        #     .builder()
        #     .at(self.center + from_polar(gauss(self.radius / 3), uniform(0, 360)), gauss(self.hue, 10))
        #     .sized(5)
        #     .anim_fade()
        #     .build()
        # )

    def color(self, alpha: int = 100):
        return from_hsv(gauss(self.hue, 10), 70, 90, alpha)

    def draw(self, gfx: "GFX"):
        super().draw(gfx)

        # gfx.circle(self.color(10), self.center, self.radius)
        # gfx.circle(self.color(20), self.center, self.radius / 2)
        # gfx.circle(self.color(40), self.center, self.radius / 4)


class GameState(State):
    BG_COLOR = None
    FPS = 60

    def __init__(self):
        super().__init__()
        self.fog = pygame.Surface(SIZE)
        self.fog.fill(WHITE)

        self.blob = self.add(Blob(
            W / 2,
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

    def bg_color(self):
        colors = [
            (0, "#3F51B5"),
            # (5, "#311B92"),
            (10, DARK),
            (15, BLACK),
            (25, (0, 0, 0)),
            (30, "#430727"),
            (35, "#601B06"),
            (40, "#2E2B08"),
            (45, "#002620"),

        ]
        return gradient(self.timer / self.FPS, *colors)

    def draw(self, gfx: CameraGFX):
        camera_x, camera_y = self.blob.pos
        # Clamp the camera y between 0m and 1000m, but smoothly
        camera_y = soft_clamp(camera_y, SURFACE, OTHER_SIDE, 1000)
        gfx.world_center = camera_x, camera_y

        gfx.surf.fill(self.bg_color())

        if self.timer % 5 == 0:
            self.blob.draw(WrapGFX(self.fog), force_alpha=0.3)

        # If at the surface, draw the sky
        if camera_y < SURFACE + H:
            gfx.rect(WHITE, camera_x, SURFACE, W, H, anchor='midbottom')
            self.generate_waves_particles(SURFACE)
        if camera_y > OTHER_SIDE - H:
            gfx.rect(BLACK, camera_x, OTHER_SIDE, W, H, anchor='midtop')
            self.generate_waves_particles(OTHER_SIDE)

        super().draw(gfx)

        # Show depth
        depth = int(-self.blob.pos.y / 100)
        gfx.text("#FFF176", f"{depth}m", size=120, midbottom=(W / 2, H - 20))

        if self.use_fog:
            self.fog.set_alpha(50)
            gfx.surf.blit(self.fog, (0, 0))

    def generate_waves_particles(self, y: float):
        hue, sat, val, _alpha = self.bg_color().hsva
        n_particles = 20
        for i in range(n_particles):
            x = uniform(-W / 2, W / 2) + self.blob.pos.x
            self.particles.add(
                CircleParticle("#ffffff")
                .builder()
                .hsv(
                    gauss(hue, 10),
                    gauss(0.2, 0.05),
                    0.95)
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
