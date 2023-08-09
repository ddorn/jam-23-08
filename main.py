from __future__ import annotations

import math
from random import gauss, random, randrange, uniform, choice, expovariate, shuffle
from time import time

import numpy as np
import pygame.gfxdraw
from pygame import Rect, Vector2, Color

from engine import *
from particles_np import CircleParticles, Gauss, Polar, TextParticles

PINK = pygame.Color("#E91E63")
ORANGE = pygame.Color("#F39C12")
INDIGO = pygame.Color("#3F51B5")
DARK = pygame.Color("#2C3E50")
WHITE = pygame.Color("#ECF0F1")
BLACK = pygame.Color("#212121")

Y_TO_METERS = 1 / 100
METERS_TO_Y = 100
# Y of the surface
SURFACE = 0 * METERS_TO_Y
# Y of the other side
OTHER_SIDE = 10 * METERS_TO_Y


def fainter(color, multiplier):
    color = pygame.Color(*color)
    h, s, l, a = color.hsla
    color.hsla = (
        h,
        s,
        l * multiplier + 100 * (1 - multiplier),
        a * multiplier**2 / 2,
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

        self.thoughts = (
            TextParticles(anchor='midbottom', lifespan=5 * 60, font_name=BIG_FONT)
            .add_fade()
            .animate('alpha', lambda s: np.interp(s, [0, 0.02, 1], [0, 255, 0]))
            .animate('vel_y', lambda s: np.cos(s * 5 * 2 * np.pi) * 1 - 1 * s)
            .add_to(self, self.Z + 1)
        )
        self.last_thought = 0
        self.thoughts_done = set()
        self.bong_combo = 0
        self.last_bong = 0
        self.ouch_count = 0
        self.last_ouch = 0
        self.achievements = set()

    def think(self, txt: str, color: Color = None, force: bool = True, once: bool = True, big: bool = False) -> bool:
        if not force and self.last_thought < 60:
            print(f"Skipped thought: {txt}, {self.last_thought} | not forced")
            return False
        if once and txt in self.thoughts_done:
            print(f"Skipped thought: {txt}, {self.last_thought} | already done")
            return False

        x, y = self.rect.midtop
        y = clamp(y, SURFACE - H / 2 + 100, OTHER_SIDE + H / 2 - 100)
        if color is None:
            color = WHITE if y > SURFACE + 50 else BLACK
        size = 60 if big else 40
        self.thoughts.new(pos=(x, y), text=txt, color=color, size=size)
        self.last_thought = 0
        self.thoughts_done.add(txt)
        return True

    def think_multi(self, *txts: str, color: Color = None, force: bool = True, once: bool = True, big: bool = False) -> bool:
        if once and txts[0] in self.thoughts_done:
            print(f"Skipped thought: {txts[0]}, {self.last_thought} | already done - multi")
            return False

        @self.add_script_decorator
        def _():
            for txt in txts:
                while not self.think(txt, color, force, False, big):
                    yield from range(60)
                yield from range(60)

        return True

    def script(self):
        yield from range(30)
        for i in range(1, 4):
            txt = (
                    "A" * randrange(4, 6 + i)
                    + "H" * randrange(1, 2 + i)
                    + "!" * randrange(1, 2 + i)
            )
            self.think(txt, RED, once=False, big=True)
            yield from range(randrange(8, 20 - i * 2))

        yield from self.wait_until(lambda: self.pos.y > SURFACE)
        self.achieve("Got wet", "You got wet!")
        self.think("Oh", BLACK)
        yield from range(15)
        self.think("I guess I'm fine?")
        yield from range(15)

        def chance(n):
            """Returns True in expectation every n seconds"""
            return random() < 1 / (n * 60)

        while True:
            yield
            if chance(5):
                thought = choice([
                    "It's so quiet here...",
                    "How did I get here?",
                    "Why am I doing this?",
                    "Should I stay or should I go?",
                    "Something is fishy...",
                ])
                self.think(thought, force=False)
            elif chance(60):
                self.think_multi(
                    "I think...",
                    "Therefore I am?!",
                    force=False,
                )
                self.achieve("Self-awareness")

    def achieve(self, name: str, description: str = ""):
        if name in self.achievements:
            return
        self.state.add(Achievement(name, description))
        self.achievements.add(name)

    def logic(self):
        last_points = self.mk_points()
        last_pos = self.pos.copy()
        self.last_thought += 1
        self.last_ouch += 1
        self.time += 1

        super().logic()

        # Draw: Add some randomness to the points
        alpha = 0.02
        self.point_dist = alpha + (1 - alpha) * (self.point_dist +
                                                 np.random.normal(0, self.std, self.radius))
        self.point_angle_shift = (1 - alpha) * (self.point_angle_shift +
                                                np.random.normal(0, self.std * 2, self.radius))
        self.point_radius = alpha + (1 - alpha) * (self.point_radius +
                                                   np.random.normal(0, self.std * 2, self.radius))

        # Apply gravity
        if self.controllable:
            if self.pos.y < SURFACE:
                gravity = 0.3
            elif self.pos.y > OTHER_SIDE:
                gravity = -0.3
            else:
                gravity = 0
            self.vel.y += gravity

            # Move
            if self.pos.y < SURFACE:
                self.vel *= 0.98
            elif self.pos.y > OTHER_SIDE:
                self.vel *= 0.98
            else:
                keys = pygame.key.get_pressed()
                self.vel.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.acceleration
                self.vel.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * self.acceleration
                self.vel *= self.friction

        self.do_collisions()

        # If entering/exiting water, make a splash
        self.make_splash(last_pos)

        # Stop the bong streak if needed
        self.last_bong += 1
        if self.last_bong > 15 and not self.airborn:
            self.bong_combo = 0
        self.state.debug.text(self.bong_combo)

        # Draw: Add enough circle between the last and the current position that it appears a smooth line
        for (p1, color, radius), (p2, _, _) in zip(last_points, self.mk_points()):
            dist = p1.distance_to(p2)
            direction = (p2 - p1).normalize()
            for j in range(0, int(dist), max(3, radius // 3)):
                point = p1 + direction * j
                self.points_to_draw.append((point, color, radius, (self.time, j)))
        # Draw: Remove old points
        self.hist_size = min(self.hist_size, self.max_hist_size)
        self.points_to_draw = [
            p for p in self.points_to_draw if self.time - p[3][0] < self.hist_size
        ]

    def do_collisions(self):
        # Check collision with fairies
        fairy: Fairy
        for obj in self.state.objects:
            if not self.collides(obj):
                continue

            if isinstance(obj, Fairy):
                obj.alive = False
                self.hist_size += 3
                self.bong_combo += 1
                self.last_bong = 0
                self.add_speed_buff(obj.hue)
                # Generate shards particles
                for i in range(self.n_points):
                    angle = i * 360 / self.n_points
                    color = from_hsv(angle, 70, 90)
                    # fmt: off
                    self.state.particles.add(
                        ShardParticle(color)
                        .builder()
                        .at(self.center + from_polar(self.radius, angle), angle)
                        .velocity(8)
                        .anim_fade(0.2)
                        .build()
                    )
                    # fmt: on

                for _ in rrange(1.5):
                    self.state.add(Fairy(random_in_rect(SCREEN.move(self.pos - (W / 2, H / 2)))))

            elif isinstance(obj, Enemy) and self.last_ouch > 30:
                self.ouch_count += 1
                self.last_ouch = 0
                self.state.do_shake(4, 10)
                if self.think_multi("Hey!! Don't do that again!", "It was scary...", big=True):
                    pass
                elif self.think_multi("What did I say?", "DON'T DO THAT!", big=True):
                    pass
                else:
                    self.think("I'm outta here!", big=True)
                    @self.add_script_decorator
                    def _run_away():
                        pos = self.pos.copy()
                        target = (pos.x, SURFACE - 300)
                        for i in range(20):
                            self.pos = pos.lerp(target, i / 20)
                            yield
                        self.state.replace_state(EndState(self.state))

                angle = (self.pos - obj.pos).angle_to((0, 1))
                self.add_speed_buff(angle, False)

    def collides(self, other: Object):
        try:
            # noinspection PyUnresolvedReferences
            return self.pos.distance_to(other.pos) < self.radius + self.part_radius + other.radius
        except AttributeError:
            return self.rect.colliderect(other.rect)

    def add_speed_buff(self, angle: float, yiha_thought: bool = True):

        @self.add_script_decorator
        def speed_buff():
            for _ in range(10):
                self.vel = from_polar(20, angle)
                yield

        # Think
        if not yiha_thought:
            return
        base_yihas = ["Wohoo!", "Yay!",
                      "Boing", "Plop!", "Bong!",
                      "!!!", "Oh yeah!", "Yesss", "Yess",
                      "YOOOOO", "Hell yeah!",
                      "What?", "I love this!", "!!!!!",
                      "YES YES YES",
                      ]
        if self.bong_combo > 7:
            shuffle(base_yihas)
            base_yihas = base_yihas[:3]
            base_yihas += ["WOW WOW WOW", "I can't stooop!", "I'm on fire!"]
        if self.bong_combo > 4:
            base_yihas += ["Wow!", "Amazing!", "Unbelievable!"]
        if self.bong_combo > 1:
            base_yihas += ["Combo!", f"Combo x{self.bong_combo}"]
        if self.bong_combo > 0:
            base_yihas += ["Bong bong!"]
        yiha = choice(base_yihas)
        self.think(yiha, from_hsv(angle, 70, 90), once=False)

    def make_splash(self, last_pos: pygame.Vector2):
        mini = min(self.pos.y, last_pos.y)
        maxi = max(self.pos.y, last_pos.y)
        if mini < SURFACE < maxi:
            y = SURFACE
            flip_y = False
        elif mini < OTHER_SIDE < maxi:
            y = OTHER_SIDE
            flip_y = True
        else:
            return

        if self.vel.y < 0:
            pull = self.vel / 5
        else:
            pull = (0, 0.1)

        angle = self.vel.as_polar()[1] + 180 * (self.vel.y > 0)
        if flip_y:
            angle += 180

        strength = self.vel.length() * 5
        # color = self.state.bg_color()
        color = self.color(0)

        impact_pos = Vector2(self.center.x, y)

        @self.add_script_decorator
        def _():
            for frame in range(1, 3):
                yield
                for _ in rrange(strength):
                    # fmt: off
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
                    # fmt: on

    @property
    def airborn(self):
        return self.pos.y < SURFACE or self.pos.y > OTHER_SIDE

    # Drawing related

    def draw(self, gfx: GFX, force_alpha: float | None = None):
        super().draw(gfx)
        for point, color, radius, (t, _) in reversed(self.points_to_draw):
            if force_alpha is not None:
                multiplier = force_alpha
            else:
                multiplier = 1 - (self.time - t) / self.hist_size

            try:
                gfx.circle(fainter(color, multiplier), point, radius)
            except ValueError:
                print(multiplier, self.time, t, self.hist_size, color)
                raise

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


class Achievement(Object):
    SIZE = 300, 100

    def __init__(self, name: str, description: str):
        r = Rect((0, 0), self.SIZE)
        r.bottomright = (W - 10, H - 10)
        super().__init__(r.topleft, r.size)
        self.name = name
        self.description = description

    def script(self):
        yield from range(60 * 6)
        self.alive = False

    def draw(self, gfx: GFX):
        super().draw(gfx)

        gfx.box((33, 33, 33, 128), self.rect, ui=True)
        r = gfx.text("#FFC107", self.name, topleft=self.pos + Vector2(10, 10), size=30)
        gfx.text("#FFEB3B", self.description, topleft=r.bottomleft + Vector2(0, 10), size=20)


class Ocean(Object):
    Z = -100
    state: GameState

    def __init__(self):
        super().__init__((0, 0))
        self.start_time = time()

        # fmt: off
        self.current_particles = (
            CircleParticles(lifespan=Gauss(200, 30),
                            size=4,
                            vel=Gauss(4, 0.5) & 0)
            .animate('alpha',
                     lambda lp: np.interp(lp,
                                          [0, 0.3, 0.7, 1],
                                          [0, 120, 120, 0]))
            .add_to(self, self.Z + 1)
        )
        self.wave_particles = (
            CircleParticles(lifespan=60,
                            size=Gauss(10, 2),
                            vel=Polar(Gauss(1, 0.5), Gauss(-90, 15)))
            .add_fade()
            .add_constant_speed((0, 1))
            .add_to(self, self.Z + 1)
        )
        # fmt: on

    def color(self):
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
        return gradient(time() - self.start_time, *colors)

    @property
    def rect(self):
        r = Rect(0, 0, W, OTHER_SIDE - SURFACE)
        r.center = self.state.camera_pos
        r.top = clamp(r.top, SURFACE, OTHER_SIDE)
        r.bottom = clamp(r.bottom, SURFACE, OTHER_SIDE)
        return r

    def logic(self):
        super().logic()

        # Move everything in the ocean right
        for obj in self.state.get_all(Object):
            if obj is self:
                continue
            if self.rect.colliderect(obj.rect):
                obj.pos.x += 2

        # Generate current particles
        h, s, v, _ = self.color().hsva
        for _ in range(2):
            self.current_particles.new(
                pos=random_in_rect(self.rect, (-1, 2)),
                color=from_hsv(gauss(h, 10), s, gauss(v + 20, 10)),
            )

    def draw(self, gfx: GFX):
        super().draw(gfx)
        gfx.rect(self.color(), *self.rect)

        # If at the surface, draw the sky
        x, y = self.state.camera_pos
        if y < SURFACE + H:
            gfx.rect(WHITE, x, SURFACE, W, H, anchor="midbottom")
            self.generate_waves_particles(SURFACE)
        if y > OTHER_SIDE - H:
            gfx.rect(BLACK, x, OTHER_SIDE, W, H, anchor="midtop")
            self.generate_waves_particles(OTHER_SIDE)

    def generate_waves_particles(self, y: float):
        hue, sat, val, _ = self.color().hsva
        n_particles = 20
        for i in range(n_particles):
            # 1.5 is to have some even when blob moves fast
            x = uniform(-W / 1.5, W / 1.5) + self.state.blob.pos.x
            self.wave_particles.new(
                pos=(x, y),
                color=from_hsv(
                    gauss(hue, 10),
                    gauss(sat - 20, 5),
                    gauss(val + 20, 5),
                ),
            )


class Fairy(SpriteObject):
    radius = 40  # For collision detection

    def __init__(self, pos):
        self.hue = randrange(0, 360, 60)
        img = pygame.Surface((80, 80), pygame.SRCALPHA)
        super().__init__(
            pos,
            img,
        )

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


class Enemy(Object):
    SIZE = 20
    SPEED = 20
    state: GameState

    def __init__(self, pos):
        super().__init__(pos, (self.SIZE, self.SIZE), (-self.SPEED, 0))
        self.timer = 0
        self.launched = False
        self.warning_duration = 60

    def logic(self):
        self.timer += 1

        super().logic()

        if self.timer < self.warning_duration / 2:
            self.pos = self.state.blob.pos + (W / 2, 0)
        elif self.timer < self.warning_duration:
            self.pos.x = self.state.blob.pos.x + W / 2
        else:
            self.launched = True

            h, s, v, _ = Color(choice([RED, WHITE, BLACK])).hsva

            # fmt: off
            self.state.particles.add(
                CircleParticle()
                .builder()
                .hsv(h, s, v)
                .at(self.center, gauss(180, 10))
                .velocity(gauss(2, 1))
                .living(30)
                .anim_fade()
                .anim_gravity((-0.1, 0))
                .sized(1 + expovariate(0.9))
                .build()
            )
            # fmt: on

            # for color in [RED, WHITE, BLACK]:
            for _ in range(3):
                # fmt: off
                self.state.particles.add(
                    CircleParticle()
                    .builder()
                    .hsv(0, 0, 100)
                    .at(self.center
                        # + from_polar(gauss(3, 2), uniform(0, 360)),
                        , gauss(180, 30))
                    .velocity(3)
                    # .constant_force(self.vel / 5)
                    .living(8)
                    .transparency(120)
                    .sized(gauss(20, 1))
                    .anim_shrink()
                    .build()
                )
                # fmt: on

        if self.pos.x < self.state.blob.pos.x - W:
            self.alive = False

    def draw(self, gfx: GFX):
        super().draw(gfx)

        if self.launched:
            # gfx.box(RED, self.rect)
            pass
        else:
            prop = self.timer / self.warning_duration * 2
            alpha = lerp(0, 255, prop, clamp=True)
            radius = lerp(1, self.SIZE * 2, prop, clamp=True)
            gfx.circle(RED + (alpha, ), self.center, radius)


class GameState(State):
    BG_COLOR = None
    FPS = 60

    def __init__(self):
        super().__init__()
        self.fog = pygame.Surface(SIZE)
        self.fog.fill(WHITE)
        self.use_fog = True
        self.add(Ocean())

        self.blob = self.add(
            Blob(
                W / 2,
                -H / 2,
                10,
                6,
                6,
                True,
                acceleration=3,
                friction=0.8,
                hist_size=10,
                max_hist_size=50,
            ))
        # self.debug.toggle()

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
                    self.debug.text("FPS cap: " + str(self.FPS))
                elif event.key == pygame.K_d:
                    self.replace_state(EndState(self))

    def logic(self):
        super().logic()

        # Create new fairies below
        if random() < 0.04:
            self.add(Fairy(random_in_rect(SCREEN.move(self.blob.pos - (W / 2, H / 2)))))

        # De-pop objects that are too far away if we have too many
        stuff = list(self.get_all(Fairy, Enemy))
        if len(stuff) > 100:
            sorted_objects = sorted(stuff, key=lambda o: o.pos.distance_to(self.blob.pos))
            for o in sorted_objects[100:]:
                o.alive = False

        if random() < 0.01:
            self.add(Enemy(self.blob.pos + (W / 2, gauss(0, H / 10))))

        # if self.timer == 20 * self.FPS:
        #     self.particles.fountains.append(ParticleFountain.screen_confetti(SCREEN))

    def draw(self, gfx: CameraGFX):
        gfx.world_center = self.camera_pos

        if self.timer % 5 == 0:
            self.blob.draw(WrapGFX(self.fog), force_alpha=0.3)

        super().draw(gfx)

        # Show depth
        depth = int(-self.blob.pos.y / 100)
        depth = int(self.blob.pos.x / 100)
        gfx.text("#FFF176", f"{depth}m", size=120, midbottom=(W / 2, H - 20))

        if self.use_fog:
            self.fog.set_alpha(30)
            gfx.surf.blit(self.fog, (0, 0))

    @property
    def camera_pos(self):
        x, y = self.blob.pos
        # Clamp the camera y so that the surfaces never cross half the screen
        y = soft_clamp(y, SURFACE, OTHER_SIDE, 10 * METERS_TO_Y)
        return Vector2(x, y)


class EndState(State):
    BG_COLOR = None

    def __init__(self, game: GameState):
        super().__init__()
        self.blob = self.add(game.blob)
        self.blob.vel *= 0
        self.blob.controllable = False
        self.blob.thoughts.add_to(self)
        self.blob.thoughts.empty()
        self.fog = game.fog
        self.show_menu = 1

    def script(self):
        yield from range(60 * 2)
        self.blob.think("We would not be here if you had listened...", Color("#B7410E"), big=True)
        yield from range(90)
        self.blob.think("But now we are here. So what do you want to do?")
        self.show_menu = self.timer

    def draw(self, gfx: CameraGFX):
        a = self.timer * 2
        if a < 256:
            self.fog.set_alpha(self.timer)
        else:
            self.fog.set_alpha(None)
        gfx.blit(self.fog, ui=True, topleft=(0, 0))
        super().draw(gfx)

        if not self.show_menu:
            return

        b = self.timer - self.show_menu
        with gfx.ui():
            r = gfx.rect(DARK, W/2, H/2, 300, 100, 1, "center")
            r_top = Rect(r)
            r_top.bottom = r.top - 50
            gfx.box(tuple(ORANGE)[:3] + (128,), r_top)
            gfx.rect(DARK, *r_top, 1)
            gfx.text(DARK, "Fall from high (again?)", size=40, center=r_top.center)
            r_bottom = r.copy()
            r_bottom.top = r.bottom + 50
            gfx.rect(DARK, *r_bottom, 1)
            gfx.box(tuple(ORANGE)[:3] + (128,), r_bottom)
            gfx.text(DARK, "Spread love in the world", size=40, center=r_bottom.center)



class Game(App):
    NAME = "Little Blob got wet"
    USE_FPS_TITLE = True

    def __init__(self):
        super().__init__(GameState, FixedScreen(SIZE), CameraGFX)


if __name__ == "__main__":
    Game().run()
