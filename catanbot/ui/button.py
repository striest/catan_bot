import pygame

class Button:
	"""
	A class fro making buttons (it's a rectangle with text and collision detection).
	"""
	def __init__(self, screen, color, text, loc, size, fontsize = 24, border = 5):
		"""
		Create a rectangle with top-left at loc and of size size.
		"""
		assert 2*border < size[0] and 2*border < size[1], 'Border too big'
		self.screen = screen
		self.color = color
		self.text = text
		self.font = pygame.font.SysFont(None, fontsize)
		self.loc = loc
		self.size = size
		self.rect = pygame.Rect(self.loc[0] + border, self.loc[1] + border, self.size[0] - 2*border, self.size[1] - 2*border)
		self.border = border
		self.border_rect = pygame.Rect(self.loc[0], self.loc[1], self.size[0], self.size[1])

	def render(self):
		img = pygame.draw.rect(self.screen, (0, 0, 0), self.border_rect, 0)
		img = pygame.draw.rect(self.screen, self.color, self.rect, 0)
		img = self.font.render(self.text, True, (0, 0, 0))
		iw, ih = img.get_rect().size
		self.screen.blit(img, (self.loc[0] + self.size[0]/2 - iw/2, self.loc[1] + self.size[1]/2 - ih/2))

	def collidepoint(self, coord):
		"""
		Check overlap with coordinate
		"""
		return self.rect.collidepoint(coord)
