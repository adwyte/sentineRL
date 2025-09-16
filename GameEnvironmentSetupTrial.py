import pygame
import sys

# Cell types
GROUND = "ground"
OBSTACLE = "obstacle"
DRONE = "drone"
BANDIT = "bandit"

# Grid parameters
CELL_SIZE = 64
GRID_WIDTH, GRID_HEIGHT = 10, 10

pygame.init()
screen = pygame.display.set_mode((CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT))
pygame.display.set_caption("Swarm RL Environment")

# Load and resize sprites
def load_and_resize(filename):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, (CELL_SIZE, CELL_SIZE))

ground_img = load_and_resize("Sprites/ground.png")
# Reduce the opacity of ground 
ground_img.set_alpha(50)

obstacle_img = load_and_resize("Sprites/Obstacle.png")
drone_img = load_and_resize("Sprites/drone.png")
bandit_img = load_and_resize("Sprites/bandit.png")

# Grid parameters
CELL_SIZE = 64
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Lists of coordinates
drones = [(2, 1), (7, 2), (0, 8), (6, 6), (9, 3)]
bandits = [(1, 5), (6, 0), (8, 7)]
obstacles = [
    (3, 3), (3, 4), (3, 5), (4, 3), (5, 3),     # vertical wall
    (7, 5), (8, 5), (9, 5),                     # horizontal wall
    (6, 7), (6, 8), (7, 8),                     # bottom cluster
    (1, 7), (2, 7), (1, 8), (2, 8)              # left bottom block
]

# Initialize grid to ground
grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

# Place obstacles
for x, y in obstacles:
    grid[y][x] = OBSTACLE

# Place drones (only if not obstacle)
for x, y in drones:
    if grid[y][x] == GROUND:
        grid[y][x] = DRONE

# Place bandits (only if not obstacle or drone)
for x, y in bandits:
    if grid[y][x] == GROUND:
        grid[y][x] = BANDIT

def draw_grid(screen):
    # Draw cells
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
    # Draw grid lines (pale black)
    grid_color = (30, 30, 30)  # Pale black
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, grid_color, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, grid_color, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_grid(screen)
    pygame.display.update()
    pygame.time.wait(30)  # Optional: limit frame rate

pygame.quit()
sys.exit()
