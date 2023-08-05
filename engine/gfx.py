from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pygame
import pygame.gfxdraw
from pygame import Vector2, Rect

if TYPE_CHECKING:
    from pygame._common import ColorValue, RectValue

__all__ = ["GFX", "WrapGFX", "CameraGFX"]


# /!\ Experimental class.
#
# The main idea of this was to be a wrapper around pygame.draw and pygame.gfxdraw
# that would take world transformations into account when drawing.
# One might think that the problem is that it would be slow, but I don't think the
# overhead will be that significant, except, maybe, when drawing lots of small things,
# like particles.
# The main problem right now is that it is impractical. I haven't yet found a standard way to
# do this and change it from game to game...
# There are thus multiple functions that do not work here, and some that have no sense,
# but eventually I'll figure all this out.
#
# For now, just think of GFX as a wrapper around pygame.Surface with a better .blit() method.

Vec2Like = Union[tuple[int, int], Vector2]

class GFX:
    def __init__(self, surf: pygame.Surface):
        self.surf = surf

    def blit(self, surf, ui: bool = False, **anchor):
        """Blit a surface directly on the underlying surface, coordinates are in pixels."""
        r = surf.get_rect(**self.edit_anchor(anchor, ui))
        self.surf.blit(surf, r)
        return r

    # Draw functions

    def line(self, color: ColorValue, start_pos: Vec2Like, end_pos: Vec2Like, ui: bool = False):
        """Draw a line in world coordinates. Does not support alpha."""
        start_pos = self.edit_pos(start_pos, ui)
        end_pos = self.edit_pos(end_pos, ui)
        pygame.draw.line(self.surf, color, start_pos, end_pos)

    def rect(self, color: ColorValue, x, y, w, h, width=0, anchor: str = None):
        """Draw a rectangle in world coordinates. Does not support alpha for width != 0."""
        x, y = self.edit_pos((x, y))
        r = pygame.Rect(x, y, w, h)

        if anchor:
            setattr(r, anchor, (x, y))

        if width == 0:
            pygame.gfxdraw.box(self.surf, r, color)
        else:
            pygame.draw.rect(self.surf, color, r, width)

    def box(self, color: ColorValue, rect: RectValue):
        """Draw a filled rectangle in world coordinates. Supports alpha."""
        rect = self.edit_rect(Rect(rect))
        pygame.gfxdraw.box(self.surf, rect, color)

    def polygon(self, color: ColorValue, points: list[Vec2Like], width: int = 0, ui: bool = False):
        """Draw a polygon in world coordinates. Does not support alpha for width != 0."""
        points = [self.edit_pos(p, ui) for p in points]
        if width == 0:
            pygame.gfxdraw.filled_polygon(self.surf, points, color)
        else:
            pygame.gfxdraw.aapolygon(self.surf, points, color)


    def circle(self, color: ColorValue, pos: Vec2Like, radius: float, width: int = 0, ui: bool = False):
        """Draw a circle in world coordinates. Supports alpha for width = 0 or 1."""
        pos = self.edit_pos(pos, ui)

        if width == 0:
            pygame.gfxdraw.filled_circle(self.surf, round(pos.x), round(pos.y), round(radius), color)
        elif width == 1:
            pygame.gfxdraw.aacircle(self.surf, round(pos.x), round(pos.y), round(radius), color)
        else:
            pygame.draw.circle(self.surf, color, pos, radius, width)

    # For subclasses to manipulate coordinates

    def edit_pos(self, pos: Vec2Like, ui: bool = False):
        return pos

    def edit_rect(self, rect: RectValue, ui: bool = False):
        rect.center = self.edit_pos(rect.center, ui)
        return rect

    def edit_anchor(self, anchor, ui: bool = False):
        anchor, value = anchor.popitem()
        value = self.edit_pos(value, ui)
        return {anchor: value}

    # Below are functions that need a closer look

    '''
    # Position / size conversion functions

    def to_ui(self, pos):
        """Convert a position in the screen to ui coordinates."""
        return pygame.Vector2(pos) / self.ui_scale

    def to_world(self, pos):
        """Convert a position in the screen to world coordinates."""
        # noinspection PyTypeChecker
        return (
            pygame.Vector2(pos)
            - (self.surf.get_width() / 2, self.surf.get_height() / 2)
        ) / self.world_scale + self.world_center

    def scale_ui_pos(self, x, y):
        return (int(x * self.surf.get_width()), int(y * self.surf.get_height()))

    def scale_world_size(self, w, h):
        return (int(w * self.world_scale), int(h * self.world_scale))

    def scale_world_pos(self, x, y):
        return (
            int((x - self.world_center.x) * self.world_scale)
            + self.surf.get_width() // 2,
            int((y - self.world_center.y) * self.world_scale)
            + self.surf.get_height() // 2,
        )

    # Surface related functions

    @lru_cache(maxsize=2000)
    def scale_surf(self, surf: pygame.Surface, factor):
        if factor == 1:
            return surf
        if isinstance(factor, (tuple, pygame.Vector2)):
            size = (int(factor[0]), int(factor[1]))
        else:
            size = (int(surf.get_width() * factor), int(surf.get_height() * factor))
        return pygame.transform.scale(surf, size)

    def ui_blit(self, surf: pygame.Surface, **anchor):
        assert len(anchor) == 1
        anchor, value = anchor.popitem()

        s = self.scale_surf(surf, self.ui_scale)
        r = s.get_rect(**{anchor: self.scale_ui_pos(*value)})
        self.surf.blit(s, r)

    def world_blit(self, surf, pos, size, anchor="topleft"):
        s = self.scale_surf(surf, vec2int(size * self.world_scale))
        r = s.get_rect(**{anchor: pos * self.world_scale})
        r.topleft -= self.world_center
        self.surf.blit(s, r)

    def grid(self, surf, pos, blocks, steps, color=(255, 255, 255, 100)):
        """
        Draw a grid in world space.

        Args:
            surf: The surface on which to draw
            pos: World position of the topleft corner
            blocks (Tuple[int, int]): Number of columns and rows (width, height)
            steps: size of each square block, in world coordinates
            color: Color of the grid. Supports alpha.
        """

        top, left = self.scale_world_pos(*pos)
        bottom = top + steps
        right = left + steps
        for x in range(blocks[0] + 1):
            pygame.gfxdraw.line(surf, x, top, x, bottom, color)
        for y in range(blocks[0] + 1):
            pygame.gfxdraw.line(surf, left, y, right, y, color)

    def fill(self, color):
        self.surf.fill(color)

    def scroll(self, dx, dy):
        self.surf.scroll(dx, dy)

    @contextmanager
    def focus(self, rect):
        """Set the draw rectangle with clip, and translate all draw calls
        so that (0, 0) is the topleft of the given rectangle.
        """

        rect = pygame.Rect(rect)

        previous_clip = self.surf.get_clip()
        self.surf.set_clip(rect)
        self.translation = pygame.Vector2(rect.topleft)
        yield
        self.surf.set_clip(previous_clip)
        if previous_clip:
            self.translation = pygame.Vector2(previous_clip.topleft)

    # UI functions

    def text(self, txt, size, color, font_name=None, **anchor):
        """Draw a text on the screen."""
        t = text(txt, size, color, font_name)
        return self.blit(t, **anchor)

    def wrap_text(self, txt, size, color, max_width, font_name=None, align=False, **anchor):
        t = wrapped_text(txt, size, color, max_width, font_name, align)
        return self.blit(t, **anchor)

    '''


class CameraGFX(GFX):

    def __init__(self, surf: pygame.Surface):
        super().__init__(surf)
        self.translation = pygame.Vector2()
        """World coordinates that are in the center of the screen."""

    @property
    def world_center(self) -> Vector2:
        """Set the world coordinates that are in the center of the screen."""
        return -self.translation + pygame.Vector2(self.surf.get_size()) / 2

    @world_center.setter
    def world_center(self, pos: Vec2Like):
        self.translation = -pygame.Vector2(pos) + pygame.Vector2(self.surf.get_size()) / 2

    def edit_pos(self, pos: Vec2Like, ui: bool = False):
        if ui:
            return pos
        return pos + self.translation


class WrapGFX(GFX):

    def edit_pos(self, pos: Vec2Like, ui: bool = False):
        """Return position so that it is on screen as if the screen was a torus."""
        # We need to copy here, modifying in place might affect the user
        if ui:
            return pos
        pos = Vector2(pos)
        pos.x %= self.surf.get_width()
        pos.y %= self.surf.get_height()
        return pos



