import pygame
import sys
import random
import numpy as np
import copy
from collections import deque

GROUND, OBSTACLE, BANDIT = "ground", "obstacle", "bandit"
CELL_SIZE, GRID_WIDTH, GRID_HEIGHT = 64, 10, 10
CONSOLE_HEIGHT = 120
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + CONSOLE_HEIGHT
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up,down,left,right,stay
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
nA = len(ACTIONS)
alpha, gamma = 0.1, 0.9
episodes, max_steps = 200, 250
initial_epsilon, final_epsilon, epsilon_decay = 1.0, 0.05, 0.995
patience = 40

# Fuel consumption costs
FUEL_COSTS = {0: 1.2, 1: 0.8, 2: 1.0, 3: 1.0, 4: 0.1}  # up, down, left, right, stay
STUCK_DETECTION_WINDOW = 10  # Steps to look back for stuck detection
MIN_NEW_COVERAGE_THRESHOLD = 2  # Min new cells in window to avoid "stuck"

def load_and_resize(filename, size=(CELL_SIZE, CELL_SIZE)):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, size)

def make_grid():
    obstacles = [
        (3, 3), (3, 4), (3, 5), (4, 3), (5, 3),
        (7, 5), (8, 5), (9, 5), (6, 7), (6, 8), (7, 8), (1, 7), (2, 7), (1, 8), (2, 8),
        (5, 0), (5, 1), (6, 1), (7, 1), (7, 0)
    ]
    bandits = [(1, 5), (6, 0), (8, 7)]
    grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for x, y in obstacles: grid[y][x] = OBSTACLE
    for x, y in bandits: grid[y][x] = BANDIT
    return grid, obstacles, bandits

def sensory_field(x, y, grid):
    visible = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if abs(dx) + abs(dy) == 2:
                    if grid[y][nx] == OBSTACLE or grid[ny][x] == OBSTACLE:
                        continue
                if abs(dx) + abs(dy) == 1 and grid[ny][nx] == OBSTACLE:
                    continue
                if grid[ny][nx] != OBSTACLE:
                    visible.add((nx, ny))
    return visible

def state_to_tuple(x, y, grid):
    field_obs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                cell = grid[ny][nx]
            else:
                cell = "OOB"
            field_obs.append(cell)
    return (x, y, tuple(field_obs))

def calculate_fuel_cost(action):
    """Calculate fuel cost for an action"""
    return FUEL_COSTS.get(action, 1.0)

def flag_priority(x, y, seen_map, visit_count_map):
    """Calculate priority for a flagged cell based on local density and visit frequency"""
    if not seen_map[y][x]:  # Only calculate for seen cells
        return 0
    
    # Count visited neighbors in 3x3
    visited_neighbors = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if seen_map[ny][nx]:
                    visited_neighbors += 1
    
    # Priority = fewer visited neighbors + penalty for high visit count
    locality_priority = (9 - visited_neighbors) * 2  # Higher when fewer neighbors visited
    revisit_penalty = -visit_count_map[y][x] * 0.5  # Lower when visited many times
    
    # Bonus for being near unvisited cells (frontier)
    frontier_bonus = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if not seen_map[ny][nx]:
                    frontier_bonus += 3
    
    return locality_priority + revisit_penalty + frontier_bonus

def find_highest_priority_flag(seen_map, visit_count_map):
    """Find the cell with highest priority among all flagged cells"""
    best_priority = -999
    best_cell = None
    
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if seen_map[y][x]:  # Only consider seen cells
                priority = flag_priority(x, y, seen_map, visit_count_map)
                if priority > best_priority:
                    best_priority = priority
                    best_cell = (x, y)
    
    return best_cell, best_priority

def simple_pathfind(start, goal, grid):
    """Simple A* pathfinding to navigate to target cell"""
    from heapq import heappush, heappop
    
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and 
                grid[ny][nx] != OBSTACLE):
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score, neighbor))
    
    return []  # No path found

def detect_stuck(position_history, coverage_history):
    """Detect if drone is stuck based on recent history"""
    if len(position_history) < STUCK_DETECTION_WINDOW:
        return False
    
    recent_positions = list(position_history)[-STUCK_DETECTION_WINDOW:]
    recent_coverage = list(coverage_history)[-STUCK_DETECTION_WINDOW:]
    
    # Check for position cycling
    unique_positions = set(recent_positions)
    if len(unique_positions) <= 3:  # Cycling between few positions
        return True
    
    # Check for lack of coverage progress
    coverage_increase = recent_coverage[-1] - recent_coverage[0]
    if coverage_increase < MIN_NEW_COVERAGE_THRESHOLD:
        return True
    
    return False

def draw_env(grid, seen_map, drone, bandits_found, bandit_cells_detected, step_count, ep, ep_reward, coverage, detected, done_flag, fuel_consumed, fuel_efficiency, title=""):
    screen.fill((40, 40, 40))
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
                if seen_map[y][x]:
                    screen.blit(blue_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
                if (x, y) in bandit_cells_detected:
                    screen.blit(red_flag_img, (x * CELL_SIZE + CELL_SIZE - 28, y * CELL_SIZE + 4))
    
    dx, dy = drone
    screen.blit(drone_img, (dx * CELL_SIZE, dy * CELL_SIZE))
    grid_color = (30, 30, 30)
    for x in range(GRID_WIDTH + 1): 
        pygame.draw.line(screen, grid_color, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1): 
        pygame.draw.line(screen, grid_color, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)
    
    panel_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE, WINDOW_WIDTH, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, (20, 20, 20), panel_rect)
    font = pygame.font.SysFont('consolas', 18)
    lines = [
        f"{title}",
        f"Step {step_count}  Reward {ep_reward}  Fuel {fuel_consumed:.1f}",
        f"Coverage: {int(coverage * 100)}%  Bandits: {detected}  Efficiency: {fuel_efficiency:.3f}",
        ("DONE!" if done_flag else "Press [X] to quit")
    ]
    for i, line in enumerate(lines):
        txt = font.render(line, True, (200, 200, 220) if not done_flag else (80, 250, 80))
        screen.blit(txt, (10, GRID_HEIGHT * CELL_SIZE + 4 + i * 22))
    pygame.display.update()
    clock.tick(10)

def rl_episode(Q, epsilon, grid, play_mode=False, render=False, info=""):
    drone = (2, 1)
    seen_map = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    visit_count_map = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    bandits_found = set()
    bandit_cells_detected = set()
    total_reward, step = 0, 0
    total_fuel_consumed = 0
    unique_blocks_visited = 0
    steps_since_new_seen = 0
    done = False
    
    # For stuck detection
    position_history = deque(maxlen=STUCK_DETECTION_WINDOW)
    coverage_history = deque(maxlen=STUCK_DETECTION_WINDOW)
    intervention_mode = False
    intervention_path = []
    
    while step < max_steps:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit(); 
                    sys.exit()
        
        x, y = drone
        visit_count_map[y][x] += 1
        coverage = np.sum(seen_map) / np.sum(np.array(grid) != OBSTACLE)
        position_history.append((x, y))
        coverage_history.append(coverage)
        
        # Check for stuck condition
        is_stuck = detect_stuck(position_history, coverage_history)
        
        if is_stuck and not intervention_mode:
            # Trigger intervention
            target_cell, priority = find_highest_priority_flag(seen_map, visit_count_map)
            if target_cell and target_cell != (x, y):
                intervention_path = simple_pathfind((x, y), target_cell, grid)
                intervention_mode = True
                print(f"  STUCK DETECTED at ({x},{y})! Navigating to priority cell {target_cell} (priority: {priority:.1f})")
        
        # Action selection
        if intervention_mode and intervention_path:
            # Follow intervention path
            next_pos = intervention_path[0]
            intervention_path = intervention_path[1:]
            for ai, (dx, dy) in enumerate(ACTIONS):
                if (x + dx, y + dy) == next_pos:
                    a = ai
                    break
            else:
                a = 4  # Stay if no matching action
            
            if not intervention_path:  # Reached target
                intervention_mode = False
        else:
            # Normal RL action selection
            intervention_mode = False
            s_state = state_to_tuple(x, y, grid)
            if s_state not in Q:
                Q[s_state] = np.ones(nA) * 50
            
            action_indices = []
            for ai, (dx, dy) in enumerate(ACTIONS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and grid[ny][nx] != OBSTACLE:
                    action_indices.append(ai)
            if not action_indices: 
                action_indices = [4]
            
            if not play_mode:
                if np.random.rand() < epsilon:
                    a = random.choice(action_indices)
                else:
                    best = np.argmax(Q[s_state][action_indices])
                    a = action_indices[best]
            else:
                a = action_indices[np.argmax(Q[s_state][action_indices])]
        
        # Execute action
        dx, dy = ACTIONS[a]
        nx, ny = x + dx, y + dy
        fuel_cost = calculate_fuel_cost(a)
        total_fuel_consumed += fuel_cost
        
        # Calculate rewards and update seen map
        reward = 0
        newly_seen = 0
        for sx, sy in sensory_field(nx, ny, grid):
            if not seen_map[sy][sx]:
                seen_map[sy][sx] = True
                reward += 3
                newly_seen += 1
                unique_blocks_visited += 1
            if grid[sy][sx] == BANDIT and (sx, sy) not in bandits_found:
                bandits_found.add((sx, sy))
                bandit_cells_detected.add((sx, sy))
                reward += 10
        
        if newly_seen == 0:
            reward = -2
        
        # Q-learning update (only if not in play mode)
        if not play_mode and not intervention_mode:
            s_next_state = state_to_tuple(nx, ny, grid)
            if s_next_state not in Q:
                Q[s_next_state] = np.ones(nA) * 50
            Q[s_state][a] += alpha * (reward + gamma * np.max(Q[s_next_state]) - Q[s_state][a])
        
        drone = (nx, ny)
        total_reward += reward
        step += 1
        
        # Calculate fuel efficiency
        fuel_efficiency = unique_blocks_visited / total_fuel_consumed if total_fuel_consumed > 0 else 0
        
        if render:
            draw_env(grid, seen_map, drone, bandits_found, bandit_cells_detected, step, "-", total_reward, coverage, len(bandits_found), done, total_fuel_consumed, fuel_efficiency, title=info)
        
        if newly_seen > 0:
            steps_since_new_seen = 0
        else:
            steps_since_new_seen += 1
        
        done = (coverage >= 0.95) or (steps_since_new_seen >= patience)
        if done: 
            break
    
    return {
        "reward": total_reward,
        "coverage": coverage,
        "bandits": len(bandits_found),
        "fuel_consumed": total_fuel_consumed,
        "fuel_efficiency": fuel_efficiency,
        "unique_blocks": unique_blocks_visited,
        "Q": copy.deepcopy(Q)
    }

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("RL Drone Surveillance with Anti-Stuck & Fuel Tracking")
    ground_img = load_and_resize("Sprites/ground.png"); ground_img.set_alpha(120)
    obstacle_img = load_and_resize("Sprites/Obstacle.png")
    drone_img = load_and_resize("Sprites/drone.png")
    bandit_img = load_and_resize("Sprites/bandit.png")
    blue_flag_img = load_and_resize("Sprites/BlueFlag.png", (24, 24))
    red_flag_img = load_and_resize("Sprites/RedFlag.png", (24, 24))
    clock = pygame.time.Clock()

    Q = {}
    epsilon = initial_epsilon
    ep_stats = []

    grid, obstacles, bandit_list = make_grid()
    print("Training RL Agent with Stuck Detection and Fuel Tracking...")

    for ep in range(1, episodes + 1):
        result = rl_episode(Q, epsilon, grid, play_mode=False, render=False)
        ep_stats.append({
            "reward": result["reward"],
            "coverage": result["coverage"],
            "bandits": result["bandits"],
            "fuel_consumed": result["fuel_consumed"],
            "fuel_efficiency": result["fuel_efficiency"],
            "unique_blocks": result["unique_blocks"],
            "Q": copy.deepcopy(Q)
        })
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        if ep % 10 == 0 or ep <= 5 or ep > episodes - 5:
            print(f"Ep {ep:3} | Reward={result['reward']:4} Coverage={result['coverage']*100:5.1f}% " +
                  f"Bandits={result['bandits']} Fuel={result['fuel_consumed']:.1f} Eff={result['fuel_efficiency']:.3f}")

    # Visualization phase
    first5 = [0, 1, 2, 3, 4]
    last5 = list(range(episodes - 5, episodes))
    best5 = sorted(range(episodes), key=lambda i: ep_stats[i]["reward"], reverse=True)[:5]
    vis_indices = []
    for idx in first5 + best5 + last5:
        if idx not in vis_indices: 
            vis_indices.append(idx)

    print("\nVisualizing selected episodes...")
    for i in vis_indices:
        info = f"Ep {i+1} ({'First' if i<5 else ('Best' if i in best5 else 'Last')})"
        result = ep_stats[i]
        print(f"Showing {info} | R={result['reward']} C={result['coverage']*100:.1f}% " +
              f"B={result['bandits']} F={result['fuel_consumed']:.1f} E={result['fuel_efficiency']:.3f}")
        _ = rl_episode(result["Q"], 0, grid, play_mode=True, render=True, info=info)
        pygame.time.wait(750)
    
    print("Training complete. You can close the window.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); 
                sys.exit()
