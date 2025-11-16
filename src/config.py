from geometry.type import ShapeType

# game
framerate = 30

# screen
screen_width = 1280  # 720p width (reduced from 1920 for macOS compatibility)
screen_height = 720  # 720p height (reduced from 1080 for macOS compatibility)
screen_color = (255, 255, 255)

# station
num_stations = 10
station_size = 30
station_capacity = 12
station_color = (0, 0, 0)
station_shape_type_list = [
    ShapeType.RECT,
    ShapeType.CIRCLE,
    ShapeType.TRIANGLE,
    ShapeType.CROSS,
]
station_passengers_per_row = 4
station_min_distance = 100  # Minimum distance between stations in pixels

# passenger
passenger_size = 5
passenger_color = (128, 128, 128)
passenger_spawning_start_step = 1
passenger_spawning_interval_step = 10 * framerate
passenger_display_buffer = 3 * passenger_size
passenger_max_waiting_time = 60000  # 60 seconds for max wait indicator (in ms)

# Scoring based on waiting time (asymmetric to penalize delays more)
score_max_bonus = 3  # Minimal bonus for fast delivery (just doing your job!)
score_max_penalty = -3  # Maximum penalty for very slow delivery
score_excellent_time = 10000  # 10 seconds or less = max bonus (in ms)
score_poor_time = 60000  # 60 seconds or more = max penalty (in ms)

# metro
num_metros = 3
metro_size = 30
metro_color = (200, 200, 200)
metro_capacity = 6
metro_speed_per_ms = 200 / 1000  # pixels / ms
metro_passengers_per_row = 4

# path
num_paths = 3
path_width = 10
path_order_shift = 10

# button
button_color = (180, 180, 180)
button_size = 30

# path button
path_button_buffer = 20
path_button_dist_to_bottom = 50
path_button_start_left = 500
path_button_cross_size = 25
path_button_cross_width = 5

# text
score_font_size = 50
score_display_coords = (20, 20)

# RL configuration
rl_num_stations = 5
rl_num_paths = 2
rl_num_metros = 2
rl_num_station_types = 3  # Number of station types to use (2-4)
rl_step_interval = 30  # RL step every 30 game frames (1 second at 30fps)
rl_max_episode_steps = 1800  # 1 hour game time
rl_max_station_passengers = 20  # Overcrowding limit
rl_max_total_passengers = 100  # System overload limit
