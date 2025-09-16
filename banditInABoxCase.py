import pygame
import sys
import random

# Grid and UI parameters
GROUND, OBSTACLE, DRONE, BANDIT = "ground", "obstacle", "drone", "bandit"
CELL_SIZE, GRID_WIDTH, GRID_HEIGHT = 64, 10, 10
CONSOLE_HEIGHT = 120  # for status/messages
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT + CONSOLE_HEIGHT

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Swarm RL Simulation")

def load_and_resize(filename, size=(CELL_SIZE, CELL_SIZE)):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, size)

ground_img = load_and_resize("Sprites/ground.png"); ground_img.set_alpha(120)
obstacle_img = load_and_resize("Sprites/Obstacle.png")
drone_img = load_and_resize("Sprites/drone.png")
bandit_img = load_and_resize("Sprites/bandit.png")
blue_flag_img = load_and_resize("Sprites/BlueFlag.png", size=(24, 24))
red_flag_img = load_and_resize("Sprites/RedFlag.png", size=(24, 24))

drones = [(2, 1), (7, 2), (0, 8), (6, 6), (9, 3)]
bandits = [(1, 5), (6, 0), (8, 7)]
obstacles = [
    (3, 3), (3, 4), (3, 5), (4, 3), (5, 3),
    (7, 5), (8, 5), (9, 5), (6, 7), (6, 8), (7, 8), (1, 7), (2, 7), (1, 8), (2, 8),
    # Obstacle ring for undetectable bandit @ (6,0)
    (5,0), (5,1), (6,1), (7,1), (7,0)
]
grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
for x, y in obstacles: grid[y][x] = OBSTACLE
for x, y in bandits: grid[y][x] = BANDIT

coverage = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
detected_bandits = set()
bandit_cells_detected = set()

clock = pygame.time.Clock()
message_lines = []
MAX_LINES = CONSOLE_HEIGHT // 20

def log_message(msg):
    message_lines.append(msg)
    if len(message_lines) > MAX_LINES:
        message_lines.pop(0)

def draw_console_panel():
    panel_rect = pygame.Rect(0, CELL_SIZE * GRID_HEIGHT, WINDOW_WIDTH, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, (20, 20, 20), panel_rect)
    font = pygame.font.SysFont('consolas', 20)
    for i, line in enumerate(message_lines):
        txt = font.render(line, True, (200, 200, 220))
        screen.blit(txt, (10, CELL_SIZE * GRID_HEIGHT + 4 + i * 20))

def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell = grid[y][x]
            pos = (x * CELL_SIZE, y * CELL_SIZE)
            if cell == OBSTACLE:
                screen.blit(obstacle_img, pos)
            else:
                screen.blit(ground_img, pos)
                if cell == BANDIT:
                    screen.blit(bandit_img, pos)
                if coverage[y][x]:
                    screen.blit(blue_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
                if (x, y) in bandit_cells_detected:
                    screen.blit(red_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
    for drone_x, drone_y in drones:
        sensed_cells = sensory_field(drone_x, drone_y)
        for sx, sy in sensed_cells:
            if grid[sy][sx] != OBSTACLE:
                overlay = pygame.Surface((CELL_SIZE, CELL_SIZE))
                overlay.set_alpha(60)
                overlay.fill((30, 120, 220))
                screen.blit(overlay, (sx * CELL_SIZE, sy * CELL_SIZE))
        screen.blit(drone_img, (drone_x * CELL_SIZE, drone_y * CELL_SIZE))
    grid_color = (30, 30, 30)
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, grid_color, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, grid_color, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)

def available_moves(pos):
    x, y = pos
    moves = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
            if grid[ny][nx] != OBSTACLE and (nx, ny) not in drones:
                moves.append((nx, ny))
    return moves

def is_obstacle_blocked(x0, y0, x1, y1):
    if abs(x1-x0) > 1 or abs(y1-y0) > 1:
        return True
    if grid[y1][x1] == OBSTACLE:
        return True
    return False

def sensory_field(drone_x, drone_y):
    visible_cells = set()
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = drone_x + dx, drone_y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if not is_obstacle_blocked(drone_x, drone_y, nx, ny):
                    visible_cells.add((nx, ny))
    return visible_cells

def percent_visited():
    total = sum(1 for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH)
                if grid[y][x] == GROUND or grid[y][x] == BANDIT)
    visited = sum(1 for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH)
                  if coverage[y][x] and (grid[y][x] == GROUND or grid[y][x] == BANDIT))
    return visited, total, visited / total if total > 0 else 0

all_visited = False
max_steps = 500
step_count = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()

    if not all_visited and step_count < max_steps:
        for i in range(len(drones)):
            x, y = drones[i]
            moves = available_moves((x, y))
            if moves:
                drones[i] = random.choice(moves)
        for drone_x, drone_y in drones:
            coverage[drone_y][drone_x] = True
            seen_cells = sensory_field(drone_x, drone_y)
            for b in seen_cells:
                if grid[b[1]][b[0]] == BANDIT:
                    detected_bandits.add(b)
                    bandit_cells_detected.add(b)
        visited, total, frac = percent_visited()
        all_visited = (visited == total) or frac > 0.90
        step_count += 1
        if step_count % 10 == 0:
            log_message(f'Bandits detected: {len(detected_bandits)}')
            log_message(f'Coverage: {int(frac * 100)}%')
        if step_count >= max_steps or all_visited:
            log_message(f"Simulation stopped at step {step_count}")

    draw_grid()
    draw_console_panel()
    pygame.display.update()
    clock.tick(5)
