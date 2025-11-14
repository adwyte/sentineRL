import pygame
import sys
import random
import numpy as np
import copy
from collections import deque
from heapq import heappush, heappop

GROUND, OBSTACLE, BANDIT = "ground", "obstacle", "bandit"
CELL_SIZE, GRID_WIDTH, GRID_HEIGHT = 64, 10, 10
CONSOLE_HEIGHT = 150
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + CONSOLE_HEIGHT
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up,down,left,right,stay
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
nA = len(ACTIONS)

# RL Parameters
alpha, gamma = 0.15, 0.92
episodes, max_steps = 300, 300
initial_epsilon, final_epsilon, epsilon_decay = 1.0, 0.05, 0.995
patience = 50

# Fuel costs with differentiated movement
FUEL_COSTS = {0: 1.2, 1: 0.8, 2: 1.0, 3: 1.0, 4: 0.1}
STUCK_DETECTION_WINDOW = 12
MIN_NEW_COVERAGE_THRESHOLD = 3

# Multi-drone configuration
N_DRONES = 3
DRONE_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
COMMUNICATION_RANGE = 3  # Drones can share info within this range

# Drone starting positions (spread across the grid)
DRONE_STARTS = [(2, 1), (7, 1), (2, 8)]


def load_and_resize(filename, size=(CELL_SIZE, CELL_SIZE)):
    img = pygame.image.load(filename).convert_alpha()
    return pygame.transform.smoothscale(img, size)


def make_grid():
    obstacles = [
        (3, 3), (3, 4), (3, 5), (4, 3), (5, 3),
        (7, 5), (8, 5), (9, 5), (6, 7), (6, 8), (7, 8),
        (1, 7), (2, 7), (1, 8), (2, 8),
        (5, 0), (5, 1), (6, 1), (7, 1), (7, 0)
    ]
    bandits = [(1, 5), (6, 0), (8, 7)]
    grid = [[GROUND for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for x, y in obstacles:
        grid[y][x] = OBSTACLE
    for x, y in bandits:
        grid[y][x] = BANDIT
    return grid, obstacles, bandits


def sensory_field(x, y, grid):
    """Enhanced sensory field with better coverage"""
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


def state_to_tuple(x, y, grid, drone_id, other_drones_nearby):
    """Enhanced state representation with drone awareness"""
    field_obs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                cell = grid[ny][nx]
            else:
                cell = "OOB"
            field_obs.append(cell)

    # Add awareness of nearby drones (simplified)
    nearby_count = len(other_drones_nearby)
    return (drone_id, x, y, tuple(field_obs), nearby_count)


def calculate_fuel_cost(action):
    return FUEL_COSTS.get(action, 1.0)


def simple_pathfind(start, goal, grid, other_drone_positions):
    """A* pathfinding with drone collision avoidance"""

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            # Avoid other drones and obstacles
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                    grid[ny][nx] != OBSTACLE and neighbor not in other_drone_positions):

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score, neighbor))

    return []


def calculate_frontier_cells(seen_map, grid):
    """Find cells on the frontier (seen cells adjacent to unseen)"""
    frontier = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if seen_map[y][x]:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                            not seen_map[ny][nx] and grid[ny][nx] != OBSTACLE):
                        frontier.append((x, y))
                        break
    return frontier


def find_best_exploration_target(drone_pos, seen_map, visit_count_map, grid, other_targets):
    """Find optimal exploration target using utility function"""
    best_score = -999999
    best_cell = None

    frontier = calculate_frontier_cells(seen_map, grid)

    for fx, fy in frontier:
        if (fx, fy) in other_targets:
            continue

        # Distance cost
        dist = abs(drone_pos[0] - fx) + abs(drone_pos[1] - fy)

        # Unvisited neighbors bonus
        unvisited_neighbors = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = fx + dx, fy + dy
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                    not seen_map[ny][nx] and grid[ny][nx] != OBSTACLE):
                unvisited_neighbors += 1

        # Visit count penalty
        visit_penalty = visit_count_map[fy][fx] * 2

        # Utility score
        score = unvisited_neighbors * 10 - dist * 0.5 - visit_penalty

        if score > best_score:
            best_score = score
            best_cell = (fx, fy)

    return best_cell


def detect_stuck(position_history, coverage_history):
    """Detect if drone is stuck"""
    if len(position_history) < STUCK_DETECTION_WINDOW:
        return False

    recent_positions = list(position_history)[-STUCK_DETECTION_WINDOW:]
    recent_coverage = list(coverage_history)[-STUCK_DETECTION_WINDOW:]

    unique_positions = set(recent_positions)
    if len(unique_positions) <= 3:
        return True

    coverage_increase = recent_coverage[-1] - recent_coverage[0]
    if coverage_increase < MIN_NEW_COVERAGE_THRESHOLD:
        return True

    return False


def draw_env(grid, seen_map, drones, bandits_found, bandit_cells_detected, step_count,
             ep_reward, coverage, fuel_consumed, fuel_efficiency, title=""):
    screen.fill((40, 40, 40))

    # Draw grid
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

    # Draw drones with different colors
    for i, drone_state in enumerate(drones):
        dx, dy = drone_state['pos']
        drone_surf = drone_img.copy()
        color_overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        color_overlay.fill((*DRONE_COLORS[i], 80))
        drone_surf.blit(color_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(drone_surf, (dx * CELL_SIZE, dy * CELL_SIZE))

        # Draw drone ID
        font_small = pygame.font.SysFont('consolas', 14, bold=True)
        id_text = font_small.render(f"D{i}", True, (255, 255, 255))
        screen.blit(id_text, (dx * CELL_SIZE + 4, dy * CELL_SIZE + 4))

    # Grid lines
    grid_color = (30, 30, 30)
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, grid_color, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, grid_color, (0, y * CELL_SIZE), (GRID_WIDTH * CELL_SIZE, y * CELL_SIZE), 1)

    # Info panel
    panel_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE, WINDOW_WIDTH, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, (20, 20, 20), panel_rect)
    font = pygame.font.SysFont('consolas', 16)

    lines = [
        f"{title}",
        f"Step: {step_count} | Reward: {ep_reward} | Fuel: {fuel_consumed:.1f}",
        f"Coverage: {int(coverage * 100)}% | Bandits: {len(bandits_found)} | Efficiency: {fuel_efficiency:.3f}",
        f"Drones: {N_DRONES} | Press [X] to quit"
    ]

    for i, line in enumerate(lines):
        txt = font.render(line, True, (200, 200, 220))
        screen.blit(txt, (10, GRID_HEIGHT * CELL_SIZE + 4 + i * 26))

    pygame.display.update()
    clock.tick(10)


def swarm_rl_episode(Q_tables, epsilon, grid, play_mode=False, render=False, info=""):
    """Multi-drone swarm episode with coordination"""

    # Initialize drones at different starting positions
    drones = []
    for i in range(N_DRONES):
        drones.append({
            'id': i,
            'pos': DRONE_STARTS[i],
            'position_history': deque(maxlen=STUCK_DETECTION_WINDOW),
            'intervention_mode': False,
            'intervention_path': [],
            'fuel_consumed': 0,
            'unique_blocks': 0
        })

    # Shared knowledge base
    seen_map = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    visit_count_map = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    bandits_found = set()
    bandit_cells_detected = set()
    coverage_history = deque(maxlen=STUCK_DETECTION_WINDOW)

    total_reward = 0
    step = 0
    done = False

    while step < max_steps and not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        step_reward = 0
        newly_seen_total = 0

        # Get current drone positions for collision avoidance
        current_positions = [d['pos'] for d in drones]

        # Each drone takes action
        for drone_idx, drone in enumerate(drones):
            x, y = drone['pos']
            visit_count_map[y][x] += 1

            # Get nearby drones for state representation
            nearby_drones = []
            for other_idx, other in enumerate(drones):
                if other_idx != drone_idx:
                    ox, oy = other['pos']
                    if abs(x - ox) + abs(y - oy) <= COMMUNICATION_RANGE:
                        nearby_drones.append(other['pos'])

            # Update history
            drone['position_history'].append((x, y))

            # Check if stuck
            coverage = np.sum(seen_map) / np.sum(np.array(grid) != OBSTACLE)
            is_stuck = detect_stuck(drone['position_history'], coverage_history)

            # Intervention logic
            if is_stuck and not drone['intervention_mode']:
                other_targets = [d['intervention_path'][0] if d['intervention_path'] else None
                                 for d in drones if d['id'] != drone_idx]
                target = find_best_exploration_target(
                    (x, y), seen_map, visit_count_map, grid, other_targets
                )
                if target and target != (x, y):
                    other_pos = [d['pos'] for d in drones if d['id'] != drone_idx]
                    path = simple_pathfind((x, y), target, grid, other_pos)
                    if path:
                        drone['intervention_mode'] = True
                        drone['intervention_path'] = path

            # Action selection
            if drone['intervention_mode'] and drone['intervention_path']:
                next_pos = drone['intervention_path'][0]
                drone['intervention_path'] = drone['intervention_path'][1:]

                for ai, (dx, dy) in enumerate(ACTIONS):
                    if (x + dx, y + dy) == next_pos:
                        a = ai
                        break
                else:
                    a = 4

                if not drone['intervention_path']:
                    drone['intervention_mode'] = False
            else:
                drone['intervention_mode'] = False

                # RL action selection
                s_state = state_to_tuple(x, y, grid, drone_idx, nearby_drones)
                Q = Q_tables[drone_idx]

                if s_state not in Q:
                    Q[s_state] = np.ones(nA) * 50

                # Valid actions (avoid obstacles and other drones)
                action_indices = []
                for ai, (dx, dy) in enumerate(ACTIONS):
                    nx, ny = x + dx, y + dy
                    valid = (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                             grid[ny][nx] != OBSTACLE)
                    # Avoid immediate collisions
                    if valid and (nx, ny) not in [d['pos'] for d in drones if d['id'] != drone_idx]:
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

            # Collision check with other drones
            if (nx, ny) in [d['pos'] for d in drones if d['id'] != drone_idx]:
                nx, ny = x, y  # Stay in place
                a = 4

            fuel_cost = calculate_fuel_cost(a)
            drone['fuel_consumed'] += fuel_cost

            # Calculate rewards
            reward = 0
            newly_seen = 0

            for sx, sy in sensory_field(nx, ny, grid):
                if not seen_map[sy][sx]:
                    seen_map[sy][sx] = True
                    reward += 3
                    newly_seen += 1
                    drone['unique_blocks'] += 1
                    newly_seen_total += 1

                if grid[sy][sx] == BANDIT and (sx, sy) not in bandits_found:
                    bandits_found.add((sx, sy))
                    bandit_cells_detected.add((sx, sy))
                    reward += 15  # Higher reward for bandit detection

            if newly_seen == 0:
                reward = -1.5

            # Cooperation bonus: reward for exploring near but not overlapping
            for other in nearby_drones:
                dist = abs(nx - other[0]) + abs(ny - other[1])
                if 2 <= dist <= COMMUNICATION_RANGE:
                    reward += 0.5  # Small coordination bonus

            # Q-learning update
            if not play_mode and not drone['intervention_mode']:
                s_next_state = state_to_tuple(nx, ny, grid, drone_idx, nearby_drones)
                if s_next_state not in Q:
                    Q[s_next_state] = np.ones(nA) * 50
                Q[s_state][a] += alpha * (reward + gamma * np.max(Q[s_next_state]) - Q[s_state][a])

            drone['pos'] = (nx, ny)
            step_reward += reward

        total_reward += step_reward
        step += 1

        # Update coverage history
        coverage = np.sum(seen_map) / np.sum(np.array(grid) != OBSTACLE)
        coverage_history.append(coverage)

        # Calculate metrics
        total_fuel = sum(d['fuel_consumed'] for d in drones)
        total_unique = sum(d['unique_blocks'] for d in drones)
        fuel_efficiency = total_unique / total_fuel if total_fuel > 0 else 0

        if render:
            draw_env(grid, seen_map, drones, bandits_found, bandit_cells_detected,
                     step, total_reward, coverage, total_fuel, fuel_efficiency, title=info)

        # Termination conditions
        if coverage >= 0.95 or (step >= 50 and newly_seen_total == 0 and
                                all(len(d['position_history']) >= STUCK_DETECTION_WINDOW
                                    for d in drones)):
            done = True

    return {
        "reward": total_reward,
        "coverage": coverage,
        "bandits": len(bandits_found),
        "fuel_consumed": total_fuel,
        "fuel_efficiency": fuel_efficiency,
        "unique_blocks": total_unique,
        "Q_tables": [copy.deepcopy(Q_tables[i]) for i in range(N_DRONES)]
    }


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Multi-Drone Swarm RL ({N_DRONES} Drones)")

    ground_img = load_and_resize("Sprites/base.png")
    ground_img.set_alpha(120)
    obstacle_img = load_and_resize("Sprites/mountain.png")
    drone_img = load_and_resize("Sprites/drone.png")
    bandit_img = load_and_resize("Sprites/camp.png")
    blue_flag_img = load_and_resize("Sprites/BlueFlag.png", (24, 24))
    red_flag_img = load_and_resize("Sprites/RedFlag.png", (24, 24))
    clock = pygame.time.Clock()

    # Separate Q-table for each drone
    Q_tables = [{} for _ in range(N_DRONES)]
    epsilon = initial_epsilon
    ep_stats = []

    grid, obstacles, bandit_list = make_grid()
    print(f"Training {N_DRONES}-Drone Swarm with RL...")
    print(f"Starting positions: {DRONE_STARTS[:N_DRONES]}")

    for ep in range(1, episodes + 1):
        result = swarm_rl_episode(Q_tables, epsilon, grid, play_mode=False, render=False)
        ep_stats.append(result)
        epsilon = max(final_epsilon, epsilon * epsilon_decay)

        if ep % 10 == 0 or ep <= 5 or ep > episodes - 5:
            print(f"Ep {ep:3} | R={result['reward']:5.0f} C={result['coverage'] * 100:5.1f}% " +
                  f"B={result['bandits']} F={result['fuel_consumed']:6.1f} Eff={result['fuel_efficiency']:.3f}")

    # Visualization
    print("\nVisualizing episodes...")
    first5 = list(range(min(5, episodes)))
    last5 = list(range(max(0, episodes - 5), episodes))
    best5 = sorted(range(episodes), key=lambda i: ep_stats[i]["reward"], reverse=True)[:5]

    vis_indices = []
    for idx in first5 + best5 + last5:
        if idx not in vis_indices:
            vis_indices.append(idx)

    for i in vis_indices:
        info = f"Ep {i + 1} ({'First' if i < 5 else ('Best' if i in best5 else 'Last')})"
        result = ep_stats[i]
        print(f"{info} | R={result['reward']:.0f} C={result['coverage'] * 100:.1f}% " +
              f"B={result['bandits']} F={result['fuel_consumed']:.1f} E={result['fuel_efficiency']:.3f}")
        Q_tables = result["Q_tables"]
        _ = swarm_rl_episode(Q_tables, 0, grid, play_mode=True, render=True, info=info)
        pygame.time.wait(1000)

    print("\nTraining complete!")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
