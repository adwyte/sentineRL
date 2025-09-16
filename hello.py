import pygame
import os

# Constants
CELL_SIZE = 64
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Cell types
GROUND = "ground"
OBSTACLE = "obstacle"
DRONE = "drone"
BANDIT = "bandit"

# Load and resize sprites
def load_and_resize(filename):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, (CELL_SIZE, CELL_SIZE))

ground_img = load_and_resize("ground.jpg")
obstacle_img = load_and_resize("Obstacle.jpg")
drone_img = load_and_resize("drone.jpg")
bandit_img = load_and_resize("bandit.jpg")

# Sample grid setup (populate obstacles, drones, bandits as required)
grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
grid[3][4] = OBSTACLE
grid[2][5] = DRONE
grid[1][1] = BANDIT

def draw_grid(screen):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell = grid[y][x]
            pos = (x * CELL_SIZE, y * CELL_SIZE)
            if cell == OBSTACLE:
                screen.blit(obstacle_img, pos)
            else:
                screen.blit(ground_img, pos)
                if cell == DRONE:
                    screen.blit(drone_img, pos)
                elif cell == BANDIT:
                    screen.blit(bandit_img, pos)

pygame.init()
screen = pygame.display.set_mode((CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT))
draw_grid(screen)
pygame.display.update()
