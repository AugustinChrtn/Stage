import pygame
import math




class Graphique:

	def __init__(self, screen_size,cell_width, 
		cell_height, cell_margin,init, grid):

		# define colors
		self.BLACK = (0, 0, 0)
		self.WHITE = (255, 255, 255)
		self.GREEN = (0, 255, 0)
		self.RED = (255, 0, 0)
		self.BLUE = (0, 0, 255)
		self.YELLOW = (255, 255, 0)

		# cell dimensions
		self.WIDTH = cell_width
		self.HEIGHT = cell_height
		self.MARGIN = cell_margin
		self.color = self.WHITE


		pygame.init()
		pygame.font.init()

		# set the width and height of the screen (width , height)
		self.size = (screen_size, screen_size)
		self.screen = pygame.display.set_mode(self.size)

		self.font = pygame.font.SysFont('arial', 20)

		pygame.display.set_caption("Classe d'états")

		self.clock = pygame.time.Clock()

		self.init = init
		self.grid = grid


		self.screen.fill(self.BLACK)

		for row in range(len(grid)):
			for col in range(len(grid[0])):
				if [row, col] == self.init:
					self.color = self.BLUE
				elif grid[row][col] == 1:
					self.color =self.BLACK
				else:
					self.color = self.WHITE
				pygame.draw.rect(self.screen,
					self.color,
					[(self.MARGIN + self.WIDTH)*col+self.MARGIN,
					(self.MARGIN + self.HEIGHT)*row+self.MARGIN,
					self.WIDTH,
					self.HEIGHT])
		self.DrawArrow(345,310,self.BLACK,angle=0)
		self.DrawArrow(645,710,self.RED,angle=0)
		self.DrawArrow(290,650,self.GREEN,angle=270)
		label_1=self.font.render('+10',1,self.BLACK)
		label_2=self.font.render('non final',1,self.RED)
		label_3=self.font.render('+100',1,self.GREEN)
		self.screen.blit(label_1, (327, 250))
		self.screen.blit(label_2, (613, 650))
		self.screen.blit(label_3, (310, 610))
	def show(self):
		pygame.display.flip()

	def DrawArrow(self,x,y,color,angle=0):
		def rotate(pos, angle):	
			cen = (5+x,0+y)
			angle *= -(math.pi/180)
			cos_theta = math.cos(angle)
			sin_theta = math.sin(angle)
			ret = ((cos_theta * (pos[0] - cen[0]) - sin_theta * (pos[1] - cen[1])) + cen[0],
			(sin_theta * (pos[0] - cen[0]) + cos_theta * (pos[1] - cen[1])) + cen[1])
			return ret
		
		p0 = rotate((0+x  ,-20+y),angle+90)
		p1 = rotate((0+x  ,20+y ),angle+90)
		p2 = rotate((40+x ,0+y ),angle+90)
		
		pygame.draw.polygon(self.screen, color, [p0,p1,p2])
          
        
        
        
        
grid = [[0 for col in range(8)] for row in range(8)]

init = [0, 0]

screen_size = 800
cell_width = 96
cell_height = 96
cell_margin = 4
grid[5][6]=1
grid[5][5]=1
grid[5][7]=1
grid[6][5]=1

gridworld = Graphique(screen_size,cell_width, cell_height, cell_margin,init, grid)
gridworld.show()
pygame.time.delay(10000)
pygame.quit()