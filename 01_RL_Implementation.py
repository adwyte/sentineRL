## Key Setup Notes

## State: Agent position + local sensory map (flatten grid, or (x, y, local view)).

## Action space: [Up, Down, Left, Right, Stay]

## Reward: +1 for visiting a new cell, +10 for seeing a new bandit, -10 for hitting an obstacle or moving out of bounds, 0 otherwise.

## Q-learning: Tabular for small state spaces, else Deep Q-Network if state is big.

## Training output: Print progress after every episode in CMD: episode number, total reward, % coverage, bandits detected, steps to finish.

import pygame
import sys
import random
import numpy as np

# Game/Env constants
GROUND, OBSTACLE, BANDIT = "ground", "obstacle", "bandit"
CELL_SIZE, GRID_WIDTH, GRID_HEIGHT = 64, 10, 10
CONSOLE_HEIGHT = 100
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + CONSOLE_HEIGHT

# RL constants
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]  # up,down,left,right,stay
ACTION_NAMES = ['UP','DOWN','LEFT','RIGHT','STAY']
nA = len(ACTIONS)
alpha, gamma, epsilon = 0.1, 0.9, 0.1  # learning rate, discount, epsilon
episodes, max_steps = 100, 200

# Asset loading function
def load_and_resize(filename, size=(CELL_SIZE, CELL_SIZE)):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, size)

def make_grid():
    # Define your grid here
    obstacles = [
        (3, 3), (3, 4), (3, 5), (4, 3), (5, 3),
        (7, 5), (8, 5), (9, 5), (6, 7), (6, 8), (7, 8), (1, 7), (2, 7), (1, 8), (2, 8),
        (5,0), (5,1), (6,1), (7,1), (7,0)
    ]
    bandits = [(1, 5), (6, 0), (8, 7)]
    grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for x, y in obstacles: grid[y][x] = OBSTACLE
    for x, y in bandits: grid[y][x] = BANDIT
    return grid, obstacles, bandits

def state_to_idx(x, y):
    return y * GRID_WIDTH + x

### Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("RL Drone Surveillance Viz")
ground_img = load_and_resize("Sprites/ground.png"); ground_img.set_alpha(120)
obstacle_img = load_and_resize("Sprites/Obstacle.png")
drone_img = load_and_resize("Sprites/drone.png")
bandit_img = load_and_resize("Sprites/bandit.png")
blue_flag_img = load_and_resize("Sprites/BlueFlag.png", (24, 24))
red_flag_img = load_and_resize("Sprites/RedFlag.png", (24, 24))
clock = pygame.time.Clock()

# Draw environment (for 1 drone visualization)
def draw_env(grid, visit_map, drone, bandits_found, bandit_cells_detected, step_count, ep, ep_reward, coverage, detected, done_flag):
    screen.fill((40,40,40))
    # Grid
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
                if visit_map[y][x]:
                    screen.blit(blue_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
                if (x, y) in bandit_cells_detected:
                    screen.blit(red_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
    dx,dy = drone
    screen.blit(drone_img, (dx * CELL_SIZE, dy * CELL_SIZE))
    grid_color = (30, 30, 30)
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, grid_color, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, grid_color, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)
    # Console Panel
    panel_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE, WINDOW_WIDTH, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, (20,20,20), panel_rect)
    font = pygame.font.SysFont('consolas', 20)
    lines = [
        f"Episode {ep}  Step {step_count}  Reward {ep_reward}",
        f"Coverage: {int(coverage*100)}%   Bandits detected: {detected}" + ("   DONE!" if done_flag else ""),
        f"Press [X] in window to quit any time."
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, (200, 200, 220) if not done_flag else (80,250,80))
        screen.blit(txt, (10, GRID_HEIGHT * CELL_SIZE + 4 + i*24))
    pygame.display.update()
    clock.tick(10)


def q_train_and_visualize():
    Q = np.zeros((GRID_WIDTH * GRID_HEIGHT, nA))
    for ep in range(1, episodes+1):
        grid, obstacles, bandit_list = make_grid()
        drone = (2, 1) # Fixed for visualization, or randomize
        visit_map = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        bandits_found = set()
        bandit_cells_detected = set()
        total_reward, step = 0, 0
        done = False
        while step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            x, y = drone
            s_idx = state_to_idx(x, y)
            if np.random.rand() < epsilon:
                a = random.randint(0, nA-1)
            else:
                a = np.argmax(Q[s_idx])
            dx, dy = ACTIONS[a]
            nx, ny = x+dx, y+dy
            reward = 0
            fail = False
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if grid[ny][nx] == OBSTACLE:
                    nx, ny = x, y
                    reward = -10
                    fail = True
                else:
                    is_new = not visit_map[ny][nx]
                    visit_map[ny][nx] = True
                    reward = 1 if is_new else 0
                    if grid[ny][nx] == BANDIT and (nx, ny) not in bandits_found:
                        bandits_found.add((nx, ny))
                        bandit_cells_detected.add((nx, ny))
                        reward += 10
            else:
                nx, ny = x, y
                reward = -10
                fail = True
            new_s_idx = state_to_idx(nx, ny)
            Q[s_idx, a] += alpha * (reward + gamma * np.max(Q[new_s_idx]) - Q[s_idx, a])
            drone = (nx, ny)
            total_reward += reward
            step += 1
            coverage = np.sum(visit_map)/np.sum(np.array(grid)!=OBSTACLE)
            # update Pygame window every STEP or every N steps
            draw_env(grid, visit_map, drone, bandits_found, bandit_cells_detected, step, ep, total_reward, coverage, len(bandits_found), done)
            done = (coverage >= 0.95)
            if done:
                break
        # CMD output each episode
        print(f"Ep {ep:3} | Steps={step:3} Reward={total_reward:4} Coverage={coverage*100:5.1f}% Bandits={len(bandits_found)}")
    print("Training complete. You can close the window.")

if __name__ == "__main__":
    q_train_and_visualize()
