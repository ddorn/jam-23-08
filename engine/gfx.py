from contextlib import contextmanager

import pygame
import pygame.gfxdraw

from .assets import text, wrapped_text

__all__ = ["GFX"]


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


class GFX:
    def __init__(self, surf: pygame.Surface):
        self.surf = surf
        self.world_center = pygame.Vector2(0, 0)
        """World coordinates that are in the center of the screen."""
        self.world_scale = 1
        """How many pixel is one world unit."""
        self.ui_scale = 1

        self.translation = pygame.Vector2()
        """Translate all draw/blit by this amount."""

    # Positions / size conversion functions

    '''
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
    '''

    def blit(self, surf, **anchor):
        """Blit a surface directly on the underlying surface, coordinates are in pixels."""

        r = surf.get_rect(**anchor)
        r.topleft += self.translation
        self.surf.blit(surf, r)

        return r

    def text(self, txt, size, color, font_name=None, **anchor):
        """Draw a text on the screen."""
        t = text(txt, size, color, font_name)
        return self.blit(t, **anchor)

    def wrap_text(self, txt, size, color, max_width, font_name=None, align=False, **anchor):
        t = wrapped_text(txt, size, color, max_width, font_name, align)
        return self.blit(t, **anchor)

    # Draw functions

    def rect(self, x, y, w, h, color, width=0, anchor: str = None):
        """Draw a rectangle in rect coordinates."""
        r = pygame.Rect(x, y, w * self.world_scale, h * self.world_scale)

        if anchor:
            setattr(r, anchor, (x, y))

        r.topleft += self.translation

        pygame.draw.rect(self.surf, color, r, width)

    def box(self, rect, color):
        rect = pygame.Rect(rect)
        rect.topleft += self.translation
        pygame.gfxdraw.box(self.surf, rect, color)

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
        bottom = top + steps * self.world_scale
        right = left + steps * self.world_scale
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
