import pygame
import sys
import random
import numpy as np
import copy
from collections import deque

# --- constants (mostly same as your original) ---
GROUND, OBSTACLE, BANDIT = "ground", "obstacle", "bandit"
CELL_SIZE, GRID_WIDTH, GRID_HEIGHT = 64, 10, 10
CONSOLE_HEIGHT = 160
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + CONSOLE_HEIGHT

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up,down,left,right,stay
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
nA = len(ACTIONS)
alpha, gamma = 0.1, 0.9
episodes, max_steps = 200, 250
initial_epsilon, final_epsilon, epsilon_decay = 1.0, 0.05, 0.995
patience = 40

# Fuel consumption costs (indexed by action index)
FUEL_COSTS = {0: 1.2, 1: 0.8, 2: 1.0, 3: 1.0, 4: 0.1}
STUCK_DETECTION_WINDOW = 10
MIN_NEW_COVERAGE_THRESHOLD = 2

# Swarm-specific
NUM_DRONES = 3
START_POSITIONS = [(2,1), (2,2), (1,1)]  # initial positions for drones (length >= NUM_DRONES)

# --- New tuning params to encourage efficient coverage ---
NOVELTY_WEIGHT = 8.0     # how strongly to prefer actions that yield new cells (per fuel)
VISIT_PENALTY = 0.8      # penalty proportional to visit count at the target cell
NEW_SEEN_REWARD = 5      # reward per newly observed cell (was 3 before)

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
                # diagonal occlusion: only allow diagonals if adjacent orthogonals not blocked
                if abs(dx) + abs(dy) == 2:
                    if grid[y][nx] == OBSTACLE or grid[ny][x] == OBSTACLE:
                        continue
                if abs(dx) + abs(dy) == 1 and grid[ny][nx] == OBSTACLE:
                    continue
                if grid[ny][nx] != OBSTACLE:
                    visible.add((nx, ny))
    return visible

def state_to_tuple(drone_id, x, y, grid):
    """Include drone_id to namespace per-agent Q states"""
    field_obs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                cell = grid[ny][nx]
            else:
                cell = "OOB"
            field_obs.append(cell)
    return (drone_id, x, y, tuple(field_obs))

def calculate_fuel_cost(action):
    return FUEL_COSTS.get(action, 1.0)

def ensure_state(Q, state, init_value=None):
    """Ensure Q[state] exists. Use optimistic initialization by default."""
    if init_value is None:
        init_value = np.ones(nA) * 50
    if state not in Q:
        Q[state] = init_value.copy()

def flag_priority(x, y, seen_map, visit_count_map):
    if not seen_map[y][x]:
        return 0
    visited_neighbors = 0
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx, ny = x+dx, y+dy
            if 0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT:
                if seen_map[ny][nx]:
                    visited_neighbors += 1
    locality_priority = (9 - visited_neighbors) * 2
    revisit_penalty = -visit_count_map[y][x] * 0.5
    frontier_bonus = 0
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx, ny = x+dx, y+dy
            if 0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT:
                if not seen_map[ny][nx]:
                    frontier_bonus += 3
    return locality_priority + revisit_penalty + frontier_bonus

def find_highest_priority_flag(seen_map, visit_count_map):
    best_priority = -999
    best_cell = None
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if seen_map[y][x]:
                p = flag_priority(x, y, seen_map, visit_count_map)
                if p > best_priority:
                    best_priority = p
                    best_cell = (x, y)
    return best_cell, best_priority

def simple_pathfind(start, goal, grid):
    from heapq import heappush, heappop
    def heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        if current in visited:
            continue
        visited.add(current)
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            if 0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT and grid[ny][nx] != OBSTACLE:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f, neighbor))
    return []

def detect_stuck(position_history, coverage_history):
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

def draw_env(grid, seen_map, drones, bandits_found, bandit_cells_detected, step_count, ep, ep_reward, coverage, total_bandits, done_flag, fuel_consumed_by_drones, fuel_efficiency_by_drones, title=""):
    screen.fill((40,40,40))
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell = grid[y][x]
            pos = (x*CELL_SIZE, y*CELL_SIZE)
            if cell == OBSTACLE:
                screen.blit(obstacle_img, pos)
            else:
                screen.blit(ground_img, pos)
                if cell == BANDIT:
                    screen.blit(bandit_img, pos)
                if seen_map[y][x]:
                    screen.blit(blue_flag_img, (x*CELL_SIZE+CELL_SIZE-28, y*CELL_SIZE+4))
                if (x,y) in bandit_cells_detected:
                    screen.blit(red_flag_img, (x*CELL_SIZE+CELL_SIZE-28, y*CELL_SIZE+4))
    # draw drones - slightly offset so multiple drones in same cell are visible
    offsets = [(0,0), (12,12), (24,0), (0,24), (24,24)]
    for i, (dx,dy) in enumerate(drones):
        off = offsets[i % len(offsets)]
        screen.blit(drone_img, (dx*CELL_SIZE + off[0], dy*CELL_SIZE + off[1]))
        # label drone id
        font = pygame.font.SysFont('consolas', 14)
        txt = font.render(f"D{i}", True, (255,255,255))
        screen.blit(txt, (dx*CELL_SIZE+2+off[0], dy*CELL_SIZE+2+off[1]))
    grid_color = (30,30,30)
    for x in range(GRID_WIDTH+1):
        pygame.draw.line(screen, grid_color, (x*CELL_SIZE,0), (x*CELL_SIZE, GRID_HEIGHT*CELL_SIZE), 1)
    for y in range(GRID_HEIGHT+1):
        pygame.draw.line(screen, grid_color, (0,y*CELL_SIZE), (GRID_WIDTH*CELL_SIZE, y*CELL_SIZE), 1)
    # console panel
    panel_rect = pygame.Rect(0, GRID_HEIGHT*CELL_SIZE, WINDOW_WIDTH, CONSOLE_HEIGHT)
    pygame.draw.rect(screen, (20,20,20), panel_rect)
    font = pygame.font.SysFont('consolas', 16)
    lines = [
        f"{title}",
        f"Step {step_count}  Reward {ep_reward:.1f}  Coverage {coverage*100:.1f}%",
        f"Bandits found {total_bandits}  Drones {len(drones)}",
    ]
    # per-drone fuel/eff info
    for i in range(len(drones)):
        fe = fuel_efficiency_by_drones[i] if i < len(fuel_efficiency_by_drones) else 0
        fc = fuel_consumed_by_drones[i] if i < len(fuel_consumed_by_drones) else 0
        lines.append(f" D{i}: Fuel={fc:.1f} Eff={fe:.3f}")
    for i, line in enumerate(lines):
        txt = font.render(line, True, (200,200,220) if not done_flag else (80,250,80))
        screen.blit(txt, (10, GRID_HEIGHT*CELL_SIZE + 4 + i*20))
    pygame.display.update()
    clock.tick(10)

def rl_episode_swarm(Q, epsilon, grid, num_drones=NUM_DRONES, start_positions=None, play_mode=False, render=False, info=""):
    if start_positions is None:
        start_positions = START_POSITIONS[:num_drones]
    # Initialize shared structures
    seen_map = [[False for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    visit_count_map = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    bandits_found = set()
    bandit_cells_detected = set()
    total_reward = 0
    step = 0
    done = False

    # Each drone state
    drones = [start_positions[i] if i < len(start_positions) else (0,i) for i in range(num_drones)]
    unique_blocks_visited = [0]*num_drones
    steps_since_new_seen = [0]*num_drones
    fuel_consumed = [0.0]*num_drones

    # stuck detection and intervention per drone
    position_history = [deque(maxlen=STUCK_DETECTION_WINDOW) for _ in range(num_drones)]
    coverage_history = [deque(maxlen=STUCK_DETECTION_WINDOW) for _ in range(num_drones)]
    intervention_mode = [False]*num_drones
    intervention_path = [[] for _ in range(num_drones)]

    # Mark initial sensory fields as seen (so coverage begins)
    for i, (x,y) in enumerate(drones):
        for sx,sy in sensory_field(x,y,grid):
            if not seen_map[sy][sx]:
                seen_map[sy][sx] = True
                unique_blocks_visited[i] += 1
                if grid[sy][sx] == BANDIT:
                    bandits_found.add((sx,sy))
                    bandit_cells_detected.add((sx,sy))

    while step < max_steps:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

        # compute shared coverage
        coverage = np.sum(seen_map) / np.sum(np.array(grid) != OBSTACLE)

        # decide actions for each drone (sequential)
        proposed_positions = [None]*num_drones
        chosen_actions = [4]*num_drones

        # collect actions
        for i in range(num_drones):
            x, y = drones[i]
            visit_count_map[y][x] += 1
            position_history[i].append((x,y))
            coverage_history[i].append(coverage)

            # detect stuck
            is_stuck = detect_stuck(position_history[i], coverage_history[i])
            if is_stuck and not intervention_mode[i]:
                target_cell, priority = find_highest_priority_flag(seen_map, visit_count_map)
                if target_cell and target_cell != (x,y):
                    intervention_path[i] = simple_pathfind((x,y), target_cell, grid)
                    intervention_mode[i] = True
                    print(f"[Drone {i}] STUCK at {(x,y)} -> navigating to {target_cell} (p={priority:.1f})")

            # intervention path
            if intervention_mode[i] and intervention_path[i]:
                next_pos = intervention_path[i][0]; intervention_path[i] = intervention_path[i][1:]
                # map next_pos to action
                for ai,(dx,dy) in enumerate(ACTIONS):
                    if (x+dx, y+dy) == next_pos:
                        chosen_actions[i] = ai
                        break
                else:
                    chosen_actions[i] = 4
                if not intervention_path[i]:
                    intervention_mode[i] = False
            else:
                # normal RL action with novelty-aware scoring
                intervention_mode[i] = False
                s_state = state_to_tuple(i, x, y, grid)
                ensure_state(Q, s_state)  # make sure current state exists

                action_indices = []
                action_scores = []
                for ai,(dx,dy) in enumerate(ACTIONS):
                    nx, ny = x+dx, y+dy
                    if not (0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT) or grid[ny][nx] == OBSTACLE:
                        continue
                    action_indices.append(ai)

                    # compute expected new cells from moving to (nx,ny)
                    visible = sensory_field(nx, ny, grid)
                    expected_new = sum(1 for (sx,sy) in visible if not seen_map[sy][sx])

                    # fuel cost for that action
                    fuel_cost = calculate_fuel_cost(ai)
                    visit_pen = visit_count_map[ny][nx]

                    # score combines learned Q-value and novelty per fuel, minus a penalty for visited cells
                    q_val = Q[s_state][ai] if ai < len(Q[s_state]) else 0
                    score = q_val + NOVELTY_WEIGHT * (expected_new / (fuel_cost + 1e-6)) - VISIT_PENALTY * visit_pen
                    action_scores.append(score)

                if not action_indices:
                    action_indices = [4]
                    chosen_actions[i] = 4
                else:
                    if not play_mode:
                        if np.random.rand() < epsilon:
                            # exploration: prefer less visited moves (weighted)
                            weights = []
                            for ai in action_indices:
                                nx, ny = x + ACTIONS[ai][0], y + ACTIONS[ai][1]
                                w = 1.0 / (1.0 + visit_count_map[ny][nx])  # less visited -> higher weight
                                weights.append(w)
                            # normalize weights
                            s = sum(weights)
                            if s <= 0:
                                chosen_actions[i] = random.choice(action_indices)
                            else:
                                probs = [w/s for w in weights]
                                chosen_actions[i] = random.choices(action_indices, probs)[0]
                        else:
                            # exploitation: pick action with max effective score
                            best_idx = int(np.argmax(action_scores))
                            chosen_actions[i] = action_indices[best_idx]
                    else:
                        # play mode: greedy by effective score
                        best_idx = int(np.argmax(action_scores))
                        chosen_actions[i] = action_indices[best_idx]

            # propose next position (collision avoidance later)
            dx, dy = ACTIONS[chosen_actions[i]]
            nx, ny = x+dx, y+dy
            # bounds/clamp
            if not (0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT) or grid[ny][nx] == OBSTACLE:
                nx, ny = x, y
                chosen_actions[i] = 4
            proposed_positions[i] = (nx, ny)

        # execute actions sequentially but prevent collisions: if proposed pos already taken by earlier executed drone this step, stay instead
        executed_positions = []
        rewards_this_step = [0]*num_drones
        newly_seen_total = [0]*num_drones

        for i in range(num_drones):
            x, y = drones[i]
            nx, ny = proposed_positions[i]
            if (nx,ny) in executed_positions and (nx,ny) != (x,y):
                # collision avoid - stay
                nx, ny = x, y
                chosen_actions[i] = 4

            # fuel
            fuel_cost = calculate_fuel_cost(chosen_actions[i])
            fuel_consumed[i] += fuel_cost

            # compute sensing & reward
            reward = 0
            newly_seen = 0
            visible = sensory_field(nx, ny, grid)
            for sx, sy in visible:
                if not seen_map[sy][sx]:
                    seen_map[sy][sx] = True
                    reward += NEW_SEEN_REWARD
                    newly_seen += 1
                    unique_blocks_visited[i] += 1
                if grid[sy][sx] == BANDIT and (sx, sy) not in bandits_found:
                    bandits_found.add((sx, sy))
                    bandit_cells_detected.add((sx, sy))
                    reward += 10

            # penalty for moving into cells already heavily visited
            if visit_count_map[ny][nx] > 0 and newly_seen == 0:
                reward -= 1.0 * min(visit_count_map[ny][nx], 5)  # small penalty scaled by visit count

            if newly_seen == 0 and reward == 0:
                # small negative reward to discourage pointless moves
                reward -= 2

            # Q-learning update (skip during play_mode)
            s_state = state_to_tuple(i, x, y, grid)
            nx_state = state_to_tuple(i, nx, ny, grid)

            if not play_mode and not intervention_mode[i]:
                # ensure both states exist to avoid KeyError
                ensure_state(Q, s_state)
                ensure_state(Q, nx_state)
                Q[s_state][chosen_actions[i]] += alpha * (reward + gamma * np.max(Q[nx_state]) - Q[s_state][chosen_actions[i]])

            # update drone
            drones[i] = (nx, ny)
            executed_positions.append((nx, ny))
            total_reward += reward
            rewards_this_step[i] = reward
            newly_seen_total[i] = newly_seen
            if newly_seen > 0:
                steps_since_new_seen[i] = 0
            else:
                steps_since_new_seen[i] += 1

        step += 1

        # compute overall metrics
        coverage = np.sum(seen_map) / np.sum(np.array(grid) != OBSTACLE)
        done_flags = [(coverage >= 0.95) or (s >= patience) for s in steps_since_new_seen]
        done = all(done_flags)  # done when all drones are idle/unproductive or coverage reached

        # per-drone fuel efficiency
        fuel_efficiency = [unique_blocks_visited[i] / fuel_consumed[i] if fuel_consumed[i] > 0 else 0 for i in range(num_drones)]

        if render:
            draw_env(grid, seen_map, drones, bandits_found, bandit_cells_detected, step, "-", total_reward, coverage, len(bandits_found), done, fuel_consumed, fuel_efficiency, title=info)

        if done:
            break

    # aggregate stats
    return {
        "reward": total_reward,
        "coverage": coverage,
        "bandits": len(bandits_found),
        "fuel_consumed": sum(fuel_consumed),
        "fuel_efficiency_by_drones": fuel_efficiency,
        "fuel_consumed_by_drones": fuel_consumed,
        "unique_blocks_per_drone": unique_blocks_visited,
        "Q": copy.deepcopy(Q)
    }

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("RL Swarm Drone Surveillance (Novelty-aware)")
    ground_img = load_and_resize("Sprites/base.png"); ground_img.set_alpha(120)
    obstacle_img = load_and_resize("Sprites/mountain.png")
    drone_img = load_and_resize("Sprites/drone.png")
    bandit_img = load_and_resize("Sprites/camp.png")
    blue_flag_img = load_and_resize("Sprites/BlueFlag.png", (24,24))
    red_flag_img = load_and_resize("Sprites/RedFlag.png", (24,24))
    clock = pygame.time.Clock()

    Q = {}
    epsilon = initial_epsilon
    ep_stats = []

    grid, obstacles, bandit_list = make_grid()
    print("Training RL Swarm with Novelty-aware Action Scoring...")

    for ep in range(1, episodes+1):
        result = rl_episode_swarm(Q, epsilon, grid, num_drones=NUM_DRONES, start_positions=START_POSITIONS, play_mode=False, render=False)
        ep_stats.append({
            "reward": result["reward"],
            "coverage": result["coverage"],
            "bandits": result["bandits"],
            "fuel_consumed": result["fuel_consumed"],
            "fuel_efficiency_by_drones": result["fuel_efficiency_by_drones"],
            "unique_blocks_per_drone": result["unique_blocks_per_drone"],
            "Q": copy.deepcopy(Q)
        })
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        if ep % 10 == 0 or ep <= 5 or ep > episodes - 5:
            print(f"Ep {ep:3} | Reward={result['reward']:5.1f} Coverage={result['coverage']*100:5.1f}% " +
                  f"Bandits={result['bandits']} Fuel={result['fuel_consumed']:.1f}")

    # Visualize some episodes
    first5 = [0,1,2,3,4]
    last5 = list(range(episodes-5, episodes))
    best5 = sorted(range(episodes), key=lambda i: ep_stats[i]["reward"], reverse=True)[:5]
    vis_indices = []
    for idx in first5 + best5 + last5:
        if idx not in vis_indices:
            vis_indices.append(idx)

    print("\nVisualizing selected swarm episodes...")
    for i in vis_indices:
        info = f"Ep {i+1}"
        result = ep_stats[i]
        print(f"Showing {info} | R={result['reward']} C={result['coverage']*100:.1f}% B={result['bandits']} F={result['fuel_consumed']:.1f}")
        _ = rl_episode_swarm(result["Q"], 0, grid, num_drones=NUM_DRONES, start_positions=START_POSITIONS, play_mode=True, render=True, info=info)
        pygame.time.wait(750)

    print("Training complete. Close window to exit.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

