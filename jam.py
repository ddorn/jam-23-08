from __future__ import annotations

import math
import random

import numpy as np
import pygame
import pygame.gfxdraw

BG_COLOR = pygame.Color("#252145")
BG_COLOR = pygame.Color("#F0F4FF")
W, H = 800, 600


def fainter(color, multiplier):
    color = pygame.Color(*color)
    h, s, l, a = color.hsla
    color.hsla = (
        h,
        s,
        l * multiplier + 100 * (1 - multiplier),
        a * multiplier**2,
    )
    return color


class Blob:

    def __init__(
        self,
        x,
        y,
        size=20,
        part_radius=5,
        points=5,
        controllable=False,
        hue_range: tuple[int, int] | None = None,
        hist_size: int = 200,
    ):
        self.pos = pygame.Vector2(x, y)
        self.size = size
        self.part_radius = part_radius
        self.time = 0
        self.n_points = points
        self.hue_range = hue_range
        self.std = 0.03
        self.point_dist = np.random.normal(1, self.std, self.size)
        self.point_angle_shift = np.random.normal(0, self.std, self.size)
        self.point_radius = np.random.normal(1, self.std, self.size)
        self.controllable = controllable

        self.hist_size = hist_size
        self.old_points = [[] for _ in range(self.n_points)]

    def color(self, point_index: int):
        color = pygame.color.Color(0)
        angle = point_index / self.n_points * 36 + self.time
        angle = angle % 36 / 36  # between 0 and 1, looping
        if self.hue_range is not None:
            angle = math.sin(angle * math.pi)
            angle = self.hue_range[0] + angle * (self.hue_range[1] - self.hue_range[0])
        else:
            angle *= 360

        color.hsva = (angle, 70, 90, 100)
        return color

    def event(self, event):
        pass

    def update(self):
        self.time += 0.1
        alpha = 0.02
        self.point_dist = alpha + (1 - alpha) * (self.point_dist +
                                                 np.random.normal(0, self.std, self.size))
        self.point_angle_shift = (1 - alpha) * (self.point_angle_shift +
                                                np.random.normal(0, self.std * 2, self.size))
        self.point_radius = alpha + (1 - alpha) * (self.point_radius +
                                                   np.random.normal(0, self.std * 2, self.size))

        # Move
        if self.controllable:
            speed = 10
            keys = pygame.key.get_pressed()
            self.pos.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * speed
            self.pos.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * speed

        # Add the points to the old points
        angle = math.pi * 2 / self.n_points
        for i in range(self.n_points):
            angle_i = angle * i + self.time + self.point_angle_shift[i]
            point = (pygame.Vector2(
                math.cos(angle_i),
                math.sin(angle_i),
            ) * self.point_dist[i] * self.size + self.pos)
            self.old_points[i].append([point, self.color(i), self.point_radius[i]])
            del self.old_points[i][:-self.hist_size]

    def draw(self, screen):
        if self.time % 2 < 0.1:
            # Most of the time, draw only the last point
            start = max(0, len(self.old_points[0]) - 3)
        else:
            start = 0

        points_to_draw = [
            # (pos, color, radius, order)
        ]

        for points in self.old_points:
            for i, ((p1, color, radius), (p2, _, _)) in enumerate(
                    zip(points[start:], points[start + 1:]),
                    start=self.hist_size - len(points),
            ):
                # use a fainter color for older points
                multiplier = i / self.hist_size
                color = fainter(color, multiplier)

                # Compute all points between p1 and p2
                p1 = pygame.Vector2(p1)
                dist = p1.distance_to(p2)
                dir = (p2 - p1).normalize()
                radius = max(2, int(radius * self.part_radius))
                for j in range(0, int(dist), max(1, radius // 3)):
                    point = p1 + dir * j
                    points_to_draw.append((point, color, radius, i + j / dist))

        # Draw the points in order
        for point, color, radius, _ in sorted(points_to_draw, key=lambda x: x[3]):
            pygame.gfxdraw.filled_circle(
                screen,
                int(point.x),
                int(point.y),
                radius,
                color,
            )
        # pygame.draw.line(screen, color, p1, p2, self.size // 4)
        # pygame.gfxdraw.filled_circle(screen, int(p1.x), int(p2.y), self.size // 4, color)


class AIBlob(Blob):

    def __init__(self, x, y, size=20, points=5, **kwargs):
        super().__init__(x, y, size, points=points, **kwargs)
        self.target = pygame.Vector2(0, 0)
        self.target_vel = pygame.Vector2(0, 0)
        self.vel = pygame.Vector2(0, 0)

        for _ in range(self.hist_size):
            self.update()

    def update(self):
        dist = self.pos.distance_to(self.target)
        # Random acceleration
        speed_gain = 100 / (2 + dist) + 2
        self.target_vel.x += random.gauss(0, speed_gain)
        self.target_vel.y += random.gauss(0, speed_gain)

        # Clamp the speed
        max_speed = 15
        self.target_vel.x = max(-max_speed, min(self.target_vel.x, max_speed))
        self.target_vel.y = max(-max_speed, min(self.target_vel.y, max_speed))

        self.target += self.target_vel
        # Keep the target in the screen
        if self.target.x < 0 or self.target.x > W:
            self.target_vel.x = 0
        if self.target.y < 0 or self.target.y > H:
            self.target_vel.y = 0
        self.target.x = max(0, min(self.target.x, W))
        self.target.y = max(0, min(self.target.y, H))
        # self.target.x %= W
        # self.target.y %= H
        # self.pos.x %= W
        # self.pos.y %= H

        # Move towards the target
        self.vel += (self.target - self.pos).normalize() * 2
        self.vel *= 0.9
        self.pos += self.vel

        super().update()

    def draw(self, screen):
        super().draw(screen)
        # pygame.draw.circle(screen, (0, 0, 0), self.target, 10)


def main():
    global W, H
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.flip()
    pygame.key.set_repeat(50, 10)
    pygame.display.set_caption("pygame window")
    objects = [
        Blob(100, 100, controllable=True),
        AIBlob(200, 200, 30, part_radius=15, hue_range=(190, 250)),
        AIBlob(200, 200, 30, part_radius=10, hue_range=(190, 240)),
    ]

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.VIDEORESIZE:
                W, H = event.size
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
                screen.fill(BG_COLOR)

            for object in objects:
                object.event(event)

        for obj in objects:
            obj.update()

        # screen.fill(BG_COLOR)
        for obj in objects:
            obj.draw(screen)
        pygame.display.update()
        clock.tick()
        fps = clock.get_fps()
        pygame.display.set_caption(f"pygame window - FPS: {fps:.2f}")

    pygame.quit()


if __name__ == "__main__":
    main()
