from __future__ import annotations

import pprint
import random
from typing import Dict, List
import numpy as np

import pygame

from config import (
    num_metros,
    num_paths,
    num_stations,
    passenger_color,
    passenger_size,
    passenger_spawning_interval_step,
    passenger_spawning_start_step,
    score_display_coords,
    score_font_size,
    screen_width,
    screen_height,
    station_shape_type_list,
    rl_max_station_passengers,
    rl_max_total_passengers,
    score_max_bonus,
    score_max_penalty,
    score_excellent_time,
    score_poor_time
)
from entity.get_entity import get_random_stations
from entity.metro import Metro
from entity.passenger import Passenger
from entity.path import Path
from entity.station import Station
from entity.station import Station
from event.event import Event
from event.keyboard import KeyboardEvent
from event.mouse import MouseEvent
from event.type import KeyboardEventType, MouseEventType
from geometry.point import Point
from geometry.type import ShapeType
from graph.graph_algo import bfs, build_station_nodes_dict
from graph.node import Node
from travel_plan import TravelPlan
from type import Color
from ui.button import Button
from ui.path_button import PathButton, get_path_buttons
from utils import get_shape_from_type, hue_to_rgb

TravelPlans = Dict[Passenger, TravelPlan]
pp = pprint.PrettyPrinter(indent=4)


class Mediator:
    def __init__(self) -> None:
        pygame.font.init()

        # configs
        self.passenger_spawning_step = passenger_spawning_start_step
        self.passenger_spawning_interval_step = passenger_spawning_interval_step
        self.num_paths = num_paths
        self.num_metros = num_metros
        self.num_stations = num_stations

        # UI
        self.path_buttons = get_path_buttons(self.num_paths)
        self.path_to_button: Dict[Path, PathButton] = {}
        self.buttons = [*self.path_buttons]
        self.font = pygame.font.SysFont("arial", score_font_size)

        # entities
        self.stations = get_random_stations(self.num_stations)
        self.metros: List[Metro] = []
        self.paths: List[Path] = []
        self.passengers: List[Passenger] = []
        self.path_colors: Dict[Color, bool] = {}
        for i in range(num_paths):
            color = hue_to_rgb(i / (num_paths + 1))
            self.path_colors[color] = False  # not taken
        self.path_to_color: Dict[Path, Color] = {}

        # status
        self.time_ms = 0
        self.steps = 0
        self.steps_since_last_spawn = self.passenger_spawning_interval_step + 1
        self.is_mouse_down = False
        self.is_creating_path = False
        self.path_being_created: Path | None = None
        self.travel_plans: TravelPlans = {}
        self.is_paused = False
        self.score = 0
        self.is_game_over = False
        self.game_over_reason = ""
        
        # Game over thresholds
        self.max_station_passengers = rl_max_station_passengers
        self.max_total_passengers = rl_max_total_passengers

    def assign_paths_to_buttons(self):
        # Only assign to buttons if they exist (UI mode)
        if self.path_buttons:
            for path_button in self.path_buttons:
                path_button.remove_path()

            self.path_to_button = {}
            for i in range(min(len(self.paths), len(self.path_buttons))):
                path = self.paths[i]
                button = self.path_buttons[i]
                button.assign_path(path)
                self.path_to_button[path] = button

    def render(self, screen: pygame.surface.Surface) -> None:
        for idx, path in enumerate(self.paths):
            path_order = idx - round(self.num_paths / 2)
            path.draw(screen, path_order)
        for station in self.stations:
            station.draw(screen)
        for metro in self.metros:
            metro.draw(screen)
        for button in self.buttons:
            button.draw(screen)
        text_surface = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        screen.blit(text_surface, score_display_coords)
        
        # Display game over message
        if self.is_game_over:
            game_over_font = pygame.font.SysFont("arial", 72, bold=True)
            reason_font = pygame.font.SysFont("arial", 36)
            
            # Main game over text
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            game_over_rect = game_over_text.get_rect(center=(screen_width // 2, screen_height // 2 - 50))
            
            # Semi-transparent background
            overlay = pygame.Surface((screen_width, screen_height))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            
            # Draw game over text
            screen.blit(game_over_text, game_over_rect)
            
            # Draw reason
            reason_text = reason_font.render(self.game_over_reason, True, (255, 255, 255))
            reason_rect = reason_text.get_rect(center=(screen_width // 2, screen_height // 2 + 50))
            screen.blit(reason_text, reason_rect)

    def react_mouse_event(self, event: MouseEvent):
        entity = self.get_containing_entity(event.position)

        if event.event_type == MouseEventType.MOUSE_DOWN:
            self.is_mouse_down = True
            if entity:
                if isinstance(entity, Station):
                    self.start_path_on_station(entity)

        elif event.event_type == MouseEventType.MOUSE_UP:
            self.is_mouse_down = False
            if self.is_creating_path:
                assert self.path_being_created is not None
                if entity and isinstance(entity, Station):
                    self.end_path_on_station(entity)
                else:
                    self.abort_path_creation()
            else:
                if entity and isinstance(entity, PathButton):
                    if entity.path:
                        self.remove_path(entity.path)

        elif event.event_type == MouseEventType.MOUSE_MOTION:
            if self.is_mouse_down:
                if self.is_creating_path and self.path_being_created:
                    if entity and isinstance(entity, Station):
                        self.add_station_to_path(entity)
                    else:
                        self.path_being_created.set_temporary_point(event.position)
            else:
                if entity and isinstance(entity, Button):
                    entity.on_hover()
                else:
                    for button in self.buttons:
                        button.on_exit()

    def react_keyboard_event(self, event: KeyboardEvent):
        if event.event_type == KeyboardEventType.KEY_UP:
            if event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused

    def react(self, event: Event | None):
        # Don't process user input when game is over
        if self.is_game_over:
            return
            
        if isinstance(event, MouseEvent):
            self.react_mouse_event(event)
        elif isinstance(event, KeyboardEvent):
            self.react_keyboard_event(event)

    def get_containing_entity(self, position: Point):
        for station in self.stations:
            if station.contains(position):
                return station
        for button in self.buttons:
            if button.contains(position):
                return button

    def remove_path(self, path: Path):
        # Only update button if path is tracked (UI mode)
        if path in self.path_to_button:
            self.path_to_button[path].remove_path()
        
        for metro in path.metros:
            for passenger in metro.passengers:
                self.passengers.remove(passenger)
            self.metros.remove(metro)
        self.release_color_for_path(path)
        self.paths.remove(path)
        self.assign_paths_to_buttons()
        self.find_travel_plan_for_passengers()

    def start_path_on_station(self, station: Station) -> None:
        if len(self.paths) < self.num_paths:
            self.is_creating_path = True
            assigned_color = (0, 0, 0)
            for path_color, taken in self.path_colors.items():
                if not taken:
                    assigned_color = path_color
                    self.path_colors[path_color] = True
                    break
            path = Path(assigned_color)
            self.path_to_color[path] = assigned_color
            path.add_station(station)
            path.is_being_created = True
            self.path_being_created = path
            self.paths.append(path)

    def add_station_to_path(self, station: Station) -> None:
        assert self.path_being_created is not None
        if self.path_being_created.stations[-1] == station:
            return
        # loop
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
        # non-loop
        elif self.path_being_created.stations[0] != station:
            if self.path_being_created.is_looped:
                self.path_being_created.remove_loop()
            self.path_being_created.add_station(station)

    def abort_path_creation(self) -> None:
        assert self.path_being_created is not None
        self.is_creating_path = False
        self.release_color_for_path(self.path_being_created)
        self.paths.remove(self.path_being_created)
        self.path_being_created = None

    def release_color_for_path(self, path: Path) -> None:
        self.path_colors[path.color] = False
        del self.path_to_color[path]

    def finish_path_creation(self) -> None:
        assert self.path_being_created is not None
        self.is_creating_path = False
        self.path_being_created.is_being_created = False
        self.path_being_created.remove_temporary_point()
        if len(self.metros) < self.num_metros:
            metro = Metro()
            self.path_being_created.add_metro(metro)
            self.metros.append(metro)
        self.path_being_created = None
        self.assign_paths_to_buttons()

    def end_path_on_station(self, station: Station) -> None:
        assert self.path_being_created is not None
        # current station de-dupe
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[-1] == station
        ):
            self.finish_path_creation()
        # loop
        elif (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
            self.finish_path_creation()
        # non-loop
        elif self.path_being_created.stations[0] != station:
            self.path_being_created.add_station(station)
            self.finish_path_creation()
        else:
            self.abort_path_creation()

    def get_station_shape_types(self):
        station_shape_types: List[ShapeType] = []
        for station in self.stations:
            if station.shape.type not in station_shape_types:
                station_shape_types.append(station.shape.type)
        return station_shape_types

    def is_passenger_spawn_time(self) -> bool:
        return (
            self.steps == self.passenger_spawning_step
            or self.steps_since_last_spawn == self.passenger_spawning_interval_step
        )

    def spawn_passengers(self):
        for station in self.stations:
            station_types = self.get_station_shape_types()
            other_station_shape_types = [
                x for x in station_types if x != station.shape.type
            ]
            
            # Safety check: if no other types available, skip passenger spawning
            if not other_station_shape_types:
                continue
                
            destination_shape_type = random.choice(other_station_shape_types)
            destination_shape = get_shape_from_type(
                destination_shape_type, passenger_color, passenger_size
            )
            passenger = Passenger(destination_shape)
            if station.has_room():
                station.add_passenger(passenger)
                self.passengers.append(passenger)

    def increment_time(self, dt_ms: int) -> None:
        if self.is_paused or self.is_game_over:
            return

        # record time
        self.time_ms += dt_ms
        self.steps += 1
        self.steps_since_last_spawn += 1

        # Update waiting time for all passengers
        for passenger in self.passengers:
            passenger.increment_waiting_time(dt_ms)

        # move metros
        for path in self.paths:
            for metro in path.metros:
                path.move_metro(metro, dt_ms)

        # spawn passengers
        if self.is_passenger_spawn_time():
            self.spawn_passengers()
            self.steps_since_last_spawn = 0

        self.find_travel_plan_for_passengers()
        self.move_passengers()
        
        # Check for game over conditions
        self.check_game_over()
    
    def calculate_passenger_score(self, waiting_time_ms: int) -> int:
        """
        Calculate score based on passenger waiting time.
        Returns a score from score_max_bonus to score_max_penalty.
        
        - Excellent delivery (≤ 10s): +5 to +3 points
        - Good delivery (10-30s): +3 to 0 points  
        - Acceptable (30-45s): 0 to -2 points
        - Poor (45-60s): -2 to -5 points
        - Very poor (≥ 60s): -5 points
        """
        if waiting_time_ms <= score_excellent_time:
            # Excellent: max bonus for very fast delivery
            return score_max_bonus
        elif waiting_time_ms >= score_poor_time:
            # Poor: max penalty for very slow delivery
            return score_max_penalty
        else:
            # Linear interpolation between excellent and poor
            # As waiting time increases from excellent_time to poor_time,
            # score decreases from max_bonus to max_penalty
            time_ratio = (waiting_time_ms - score_excellent_time) / (score_poor_time - score_excellent_time)
            score = score_max_bonus - (score_max_bonus - score_max_penalty) * time_ratio
            return round(score)
    
    def check_game_over(self) -> None:
        """Check if game should end due to failure conditions"""
        if self.is_game_over:
            return
        
        # Check for station overcrowding
        for station in self.stations:
            if len(station.passengers) > self.max_station_passengers:
                self.is_game_over = True
                self.game_over_reason = f"Game Over! Station overcrowded: {len(station.passengers)} passengers (max: {self.max_station_passengers})"
                return
        
        # Check for system overload
        if len(self.passengers) > self.max_total_passengers:
            self.is_game_over = True
            self.game_over_reason = f"Game Over! System overload: {len(self.passengers)} passengers (max: {self.max_total_passengers})"
            return

    def move_passengers(self) -> None:
        for metro in self.metros:
            if metro.current_station:
                passengers_to_remove = []
                passengers_from_metro_to_station = []
                passengers_from_station_to_metro = []

                # queue
                for passenger in metro.passengers:
                    if (
                        metro.current_station.shape.type
                        == passenger.destination_shape.type
                    ):
                        passengers_to_remove.append(passenger)
                    elif (
                        self.travel_plans[passenger].get_next_station()
                        == metro.current_station
                    ):
                        passengers_from_metro_to_station.append(passenger)
                for passenger in metro.current_station.passengers:
                    if (
                        self.travel_plans[passenger].next_path
                        and self.travel_plans[passenger].next_path.id == metro.path_id  # type: ignore
                    ):
                        passengers_from_station_to_metro.append(passenger)

                # process
                for passenger in passengers_to_remove:
                    passenger.is_at_destination = True
                    
                    # Calculate score based on waiting time
                    points = self.calculate_passenger_score(passenger.waiting_time)
                    self.score += points
                    
                    # Optional: Print feedback for debugging
                    # print(f"Passenger delivered in {passenger.waiting_time/1000:.1f}s: {points:+d} points")
                    
                    metro.remove_passenger(passenger)
                    self.passengers.remove(passenger)
                    del self.travel_plans[passenger]

                for passenger in passengers_from_metro_to_station:
                    if metro.current_station.has_room():
                        metro.move_passenger(passenger, metro.current_station)
                        self.travel_plans[passenger].increment_next_station()
                        self.find_next_path_for_passenger_at_station(
                            passenger, metro.current_station
                        )

                for passenger in passengers_from_station_to_metro:
                    if metro.has_room():
                        metro.current_station.move_passenger(passenger, metro)

    def get_stations_for_shape_type(self, shape_type: ShapeType):
        stations: List[Station] = []
        for station in self.stations:
            if station.shape.type == shape_type:
                stations.append(station)
        random.shuffle(stations)

        return stations

    def find_shared_path(self, station_a: Station, station_b: Station) -> Path | None:
        """Find a path connecting station_a and station_b. Returns the first match found."""
        for path in self.paths:
            stations = path.stations
            if (station_a in stations) and (station_b in stations):
                return path
        return None
    
    def find_best_path_for_journey(self, current_station: Station, next_station: Station, remaining_journey: List[Node]) -> Path | None:
        """
        Find the best path connecting current_station to next_station that also
        continues toward the passenger's final destination.
        
        Scoring strategy:
        1. PRIMARY: Minimize distance (hops) from current to next station
        2. SECONDARY: Prefer lines that continue further toward destination
        
        This ensures passengers take shortcuts when available, but when multiple
        lines have the same distance, they choose the one that continues further.
        """
        candidate_paths = []
        
        for path in self.paths:
            stations = path.stations
            if (current_station in stations) and (next_station in stations):
                # Calculate distance from current to next station
                current_idx = stations.index(current_station)
                next_idx = stations.index(next_station)
                distance_to_next = abs(next_idx - current_idx)
                
                # Calculate how far this line continues toward destination
                # Count how many future stations this line covers
                continuation_score = 0
                for future_node in remaining_journey[1:]:  # Skip next_station (already counted)
                    if future_node.station in stations:
                        continuation_score += 1
                
                # Store path with (distance_to_next, -continuation_score)
                # We negate continuation_score so higher scores become lower (better) when sorting
                candidate_paths.append((path, distance_to_next, -continuation_score))
        
        if candidate_paths:
            # Sort by distance first (lower is better), then by continuation (higher is better, so -score is lower)
            # Returns the path with shortest distance, and among ties, the one that continues furthest
            best_path = min(candidate_paths, key=lambda x: (x[1], x[2]))[0]
            return best_path
        return None

    def passenger_has_travel_plan(self, passenger: Passenger) -> bool:
        return (
            passenger in self.travel_plans
            and self.travel_plans[passenger].next_path is not None
        )

    def find_next_path_for_passenger_at_station(
        self, passenger: Passenger, station: Station
    ):
        next_station = self.travel_plans[passenger].get_next_station()
        assert next_station is not None
        
        # Use the smarter path selection that considers the entire journey
        remaining_journey = self.travel_plans[passenger].node_path[self.travel_plans[passenger].next_station_idx:]
        next_path = self.find_best_path_for_journey(station, next_station, remaining_journey)
        
        # Fallback to simple shared path if no path found (shouldn't happen if graph is correct)
        if next_path is None:
            next_path = self.find_shared_path(station, next_station)
        
        self.travel_plans[passenger].next_path = next_path

    def skip_stations_on_same_path(self, node_path: List[Node]):
        assert len(node_path) >= 2
        if len(node_path) == 2:
            return node_path
        else:
            nodes_to_remove = []
            i = 0
            j = 1
            path_set_list = [x.paths for x in node_path]
            path_set_list.append(set())
            while j <= len(path_set_list) - 1:
                set_a = path_set_list[i]
                set_b = path_set_list[j]
                if set_a & set_b:
                    j += 1
                else:
                    for k in range(i + 1, j - 1):
                        nodes_to_remove.append(node_path[k])
                    i = j - 1
                    j += 1
            for node in nodes_to_remove:
                node_path.remove(node)
        return node_path

    def find_travel_plan_for_passengers(self) -> None:
        station_nodes_dict = build_station_nodes_dict(self.stations, self.paths)
        for station in self.stations:
            for passenger in station.passengers:
                # update travel plan for all passengers regardless of existing plan
                #if not self.passenger_has_travel_plan(passenger):
                    possible_dst_stations = self.get_stations_for_shape_type(
                        passenger.destination_shape.type
                    )
                    
                    # Find the shortest path among all possible destination stations
                    shortest_path = None
                    shortest_length = float('inf')
                    passenger_at_destination = False
                    
                    for possible_dst_station in possible_dst_stations:
                        start = station_nodes_dict[station]
                        end = station_nodes_dict[possible_dst_station]
                        node_path = bfs(start, end)
                        
                        if len(node_path) == 1:
                            # passenger arrived at destination
                            station.remove_passenger(passenger)
                            self.passengers.remove(passenger)
                            passenger.is_at_destination = True
                            del self.travel_plans[passenger]
                            passenger_at_destination = True
                            break
                        elif len(node_path) > 1:
                            # Compare path lengths and keep the shortest one
                            if len(node_path) < shortest_length:
                                shortest_length = len(node_path)
                                shortest_path = node_path
                    
                    # If passenger is at destination, we already handled it
                    if passenger_at_destination:
                        continue
                    
                    # Set the shortest path found, or null path if no path exists
                    if shortest_path is not None:
                        shortest_path = self.skip_stations_on_same_path(shortest_path)
                        self.travel_plans[passenger] = TravelPlan(shortest_path[1:])
                        self.find_next_path_for_passenger_at_station(
                            passenger, station
                        )
                    else:
                        self.travel_plans[passenger] = TravelPlan([])

    def debug_print(self) -> None:
        print("=== Mediator ===")
        print(f"time_ms: {self.time_ms}")
        print(f"steps: {self.steps}")
        print(f"num_paths: {len(self.paths)} / {self.num_paths}")
        print(f"num_metros: {len(self.metros)} / {self.num_metros}")
        print(f"num_stations: {len(self.stations)} / {self.num_stations}")
        print(f"num_passengers: {len(self.passengers)}")
        print("paths:")
        for path in self.paths:
            print(f"  - {path}")
            print(f"    stations: {[station.id for station in path.stations]}")
            print(f"    metros: {[metro.id for metro in path.metros]}")
        print("stations:")
        for station in self.stations:
            print(f"  - {station.id}: {[passenger.id for passenger in station.passengers]}")
        print("metros:")
        for metro in self.metros:
            current_station_id = (
                metro.current_station.id if metro.current_station else None
            )
            print(
                f"  - {metro.id}: current_station={current_station_id}, passengers={[passenger.id for passenger in metro.passengers]}"
            )
        print("travel_plans:")
        for passenger, travel_plan in self.travel_plans.items():
            next_station_id = (
                travel_plan.get_next_station().id
                if travel_plan.get_next_station()
                else None
            )
            next_path_id = travel_plan.next_path.id if travel_plan.next_path else None
            remaining_stations = [node.station.id for node in travel_plan.remaining_stations]
            print(
                f"  - {passenger.id}: next_station={next_station_id}, next_path={next_path_id}, remaining_stations={remaining_stations}"
            )
        print("===")

    # RL Methods
    def rl_build_path(self, station_ids: List[int]) -> bool:
        """
        Directly build a path connecting the given stations by ID.
        Returns True if successful, False if invalid action.
        """
        if len(station_ids) < 2:
            return False
        
        if len(self.paths) >= self.num_paths:
            return False
            
        # Get stations by ID
        stations = []
        for station_id in station_ids:
            if station_id >= len(self.stations) or station_id < 0:
                return False
            stations.append(self.stations[station_id])
        
        # Check if path creation is valid
        if not self._is_valid_path_creation(stations):
            return False
        
        # Create path directly (similar to existing logic)
        assigned_color = self._get_available_color()
        if assigned_color is None:
            return False
            
        path = Path(assigned_color)
        self.path_to_color[path] = assigned_color
        
        # Add all stations to path
        for station in stations:
            path.add_station(station)
        
        # Check if it should be a loop
        if len(stations) > 2 and stations[0] == stations[-1]:
            path.set_loop()
        
        path.is_being_created = False
        
        # Add metro if available
        if len(self.metros) < self.num_metros:
            metro = Metro()
            path.add_metro(metro)
            self.metros.append(metro)
        
        self.paths.append(path)
        self.assign_paths_to_buttons()
        self.find_travel_plan_for_passengers()
        
        return True

    def rl_remove_path(self, path_id: int) -> bool:
        """
        Directly remove a path by index.
        Returns True if successful, False if invalid.
        """
        if path_id >= len(self.paths) or path_id < 0:
            return False
        
        path = self.paths[path_id]
        self.remove_path(path)  # Use existing method
        return True

    def rl_extend_path(self, path_id: int, station_id: int) -> bool:
        """
        Extend an existing path by adding a station to it.
        Returns True if successful, False if invalid action.
        """
        # Validate inputs
        if path_id >= len(self.paths) or path_id < 0:
            return False
        if station_id >= len(self.stations) or station_id < 0:
            return False
        
        path = self.paths[path_id]
        new_station = self.stations[station_id]
        
        # Check if station is already on this path
        if new_station in path.stations:
            return False
        
        # Check if adding this station would create a valid extension
        # We can extend from either end of the path (unless it's a loop)
        if path.is_looped:
            return False  # Can't extend a looped path
        
        # Try to extend from the end
        if len(path.stations) > 0:
            # Create temporary extended path to validate
            extended_stations = path.stations + [new_station]
            if self._is_valid_path_creation(extended_stations):
                # Add the station to the existing path
                path.add_station(new_station)
                
                # Update path segments and other path properties
                path.update_segments()
                
                # Recalculate travel plans for all passengers
                self.find_travel_plan_for_passengers()
                
                return True
        
        return False

    def _get_available_color(self):
        """Helper method to get an available color for new path"""
        for path_color, taken in self.path_colors.items():
            if not taken:
                self.path_colors[path_color] = True
                return path_color
        return None

    def _is_valid_path_creation(self, stations: List[Station]) -> bool:
        """Validate if path creation is allowed"""
        # Basic validation
        if len(stations) < 2:
            return False
        
        # Check for duplicates (except potential loop)
        for i in range(len(stations) - 1):
            if stations[i] in stations[i+1:len(stations)-1]:  # Allow loop at end
                return False
        
        # Check if stations are already connected on same path
        for path in self.paths:
            path_stations = set(path.stations)
            if all(station in path_stations for station in stations):
                return False  # Path already exists
        
        return True

    def get_rl_state_vector(self) -> np.ndarray:
        """
        Get flattened state vector for RL algorithms with dynamic size based on configuration.
        """
        state = []
        
        # Station features (num_stations × 8 features)
        for i in range(self.num_stations):
            if i < len(self.stations):
                station = self.stations[i]
                state.extend([
                    station.position.left / screen_width,  # Normalized position
                    station.position.top / screen_height,
                    len(station.passengers) / station.capacity,  # Occupancy ratio
                    float(station.shape.type.value),  # Station type (1-4 from enum value)
                    float(self._station_has_path(station)),  # Connected (0/1)
                    self._get_avg_passenger_wait_time(station),  # Avg wait time
                    len(station.passengers) / 20.0,  # Normalized passenger count
                    self._get_station_destination_diversity(station)  # Destination entropy
                ])
            else:
                state.extend([0.0] * 8)  # Empty station
        
        # Path features (num_paths × 5 features)
        for i in range(self.num_paths):
            if i < len(self.paths):
                path = self.paths[i]
                state.extend([
                    len(path.stations) / float(self.num_stations),  # Normalized length
                    float(len(path.metros)),  # Metro count
                    float(path.is_looped),  # Is loop
                    self._get_path_passenger_count(path) / 20.0,  # Total passengers (normalized)
                    self._get_path_efficiency(path),  # Usage efficiency
                ])
            else:
                state.extend([0.0] * 5)  # Empty path
        
        # Global state (8 features)
        state.extend([
            len(self.passengers) / 100.0,  # Total passengers (normalized)
            self.score / 100.0,  # Score (normalized) 
            self.time_ms / (60 * 1000),  # Time in minutes
            len([s for s in self.stations if len(s.passengers) > 10]) / len(self.stations),  # Overcrowd ratio
            len(self.paths) / self.num_paths,  # Path utilization
            len(self.metros) / self.num_metros,  # Metro utilization
            self._get_system_efficiency(),  # Overall efficiency
            self._get_connectivity_score()  # Network connectivity
        ])
        
        return np.array(state, dtype=np.float32)

    def _station_has_path(self, station: Station) -> int:
        """Check if station is connected to any path"""
        for path in self.paths:
            if station in path.stations:
                return 1
        return 0

    def _get_avg_passenger_wait_time(self, station: Station) -> float:
        """Calculate average waiting time for passengers at station"""
        if not station.passengers:
            return 0.0
        # Simplified - could track actual spawn times
        return min(len(station.passengers) * 0.05, 1.0)  # Normalized wait time proxy

    def _get_station_destination_diversity(self, station: Station) -> float:
        """Calculate entropy of passenger destinations at station"""
        if not station.passengers:
            return 0.0
        
        destinations = [p.destination_shape.type for p in station.passengers]
        unique_dest = set(destinations)
        if len(unique_dest) <= 1:
            return 0.0
        
        # Simple diversity metric
        return len(unique_dest) / len(destinations)

    def _get_path_passenger_count(self, path: Path) -> int:
        """Get total passengers on path (stations + metros)"""
        total = 0
        for station in path.stations:
            total += len(station.passengers)
        for metro in path.metros:
            total += len(metro.passengers)
        return total

    def _get_path_efficiency(self, path: Path) -> float:
        """Calculate path efficiency (passengers moved vs capacity)"""
        if not path.metros:
            return 0.0
        
        total_capacity = sum(metro.capacity for metro in path.metros)
        total_passengers = sum(len(metro.passengers) for metro in path.metros)
        
        if total_capacity == 0:
            return 0.0
        
        return total_passengers / total_capacity

    def _get_system_efficiency(self) -> float:
        """Calculate overall system efficiency"""
        if not self.passengers:
            return 1.0
        
        # Ratio of passengers being transported vs waiting
        waiting = sum(len(s.passengers) for s in self.stations)
        traveling = sum(len(m.passengers) for m in self.metros)
        
        total = waiting + traveling
        if total == 0:
            return 1.0
        
        return traveling / total

    def _get_connectivity_score(self) -> float:
        """Calculate how well stations are connected"""
        if not self.stations:
            return 0.0
        connected_stations = set()
        for path in self.paths:
            connected_stations.update(path.stations)
        return len(connected_stations) / len(self.stations)
    
    def get_network_reachability(self) -> dict:
        if not self.stations or not self.paths:
            return {
                'is_fully_connected': False,
                'reachability_matrix': {},
                'unreachable_pairs': [],
                'connected_components': [[s] for s in self.stations],
                'connectivity_ratio': 0.0,
                'isolated_stations': self.stations[:],
                'reachable_from_any': set(),
                'num_components': len(self.stations),
                'largest_component_size': 1 if self.stations else 0
            }
        # Build station nodes graph
        station_nodes_dict = build_station_nodes_dict(self.stations, self.paths)
        # Find isolated stations (not on any path)
        isolated_stations = []
        for station in self.stations:
            if station not in station_nodes_dict or not station_nodes_dict[station].neighbors:
                isolated_stations.append(station)
        # Build reachability matrix using BFS from each station
        reachability_matrix = {}
        for start_station in self.stations:
            if start_station in station_nodes_dict:
                start_node = station_nodes_dict[start_station]
                reachable = set()
                # BFS to find all reachable stations
                visited = set()
                queue = [start_node]
                visited.add(start_node)
                while queue:
                    current_node = queue.pop(0)
                    reachable.add(current_node.station)
                    for neighbor in current_node.neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                reachability_matrix[start_station] = reachable
            else:
                # Isolated station
                reachability_matrix[start_station] = {start_station}
        # Find unreachable pairs
        unreachable_pairs = []
        reachable_pairs_count = 0
        total_pairs = 0
        for i, station1 in enumerate(self.stations):
            for station2 in self.stations[i+1:]:
                total_pairs += 1
                if station2 not in reachability_matrix.get(station1, set()):
                    unreachable_pairs.append((station1, station2))
                else:
                    reachable_pairs_count += 1
        # Calculate connectivity ratio
        connectivity_ratio = reachable_pairs_count / total_pairs if total_pairs > 0 else 0.0
        # Find connected components (using Union-Find approach)
        connected_components = self._find_connected_components(reachability_matrix)
        # Check if fully connected
        is_fully_connected = len(connected_components) == 1 and len(isolated_stations) == 0
        # Find stations reachable from at least one other station
        reachable_from_any = set()
        for reachable_set in reachability_matrix.values():
            reachable_from_any.update(reachable_set)
        return {
            'is_fully_connected': is_fully_connected,
            'reachability_matrix': reachability_matrix,
            'unreachable_pairs': unreachable_pairs,
            'connected_components': connected_components,
            'connectivity_ratio': connectivity_ratio,
            'isolated_stations': isolated_stations,
            'reachable_from_any': reachable_from_any,
            'num_components': len(connected_components),
            'largest_component_size': max(len(comp) for comp in connected_components) if connected_components else 0
        }
    
    def _find_connected_components(self, reachability_matrix: dict) -> list:
        """
        Find connected components using the reachability matrix.
        Returns list of components, where each component is a list of stations.
        """
        if not reachability_matrix:
            return []
        
        visited = set()
        components = []
        
        for station in reachability_matrix.keys():
            if station not in visited:
                # Start a new component
                component = set()
                queue = [station]
                
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        # Add all reachable stations from current
                        reachable = reachability_matrix.get(current, set())
                        for reachable_station in reachable:
                            if reachable_station not in visited:
                                queue.append(reachable_station)
                
                components.append(list(component))
        
        return components
    
    def get_network_completeness_score(self) -> float:
        """
        Calculate a comprehensive network completeness score
        """
        if not self.stations:
            return 1.0
        if not self.paths:
            return 0.0
        reachability_info = self.get_network_reachability()
        connectivity_score = reachability_info['connectivity_ratio'] * 0.5
        num_components = reachability_info['num_components']
        component_score = (1.0 - (num_components - 1) / len(self.stations)) * 0.3
        num_isolated = len(reachability_info['isolated_stations'])
        isolation_score = (1.0 - num_isolated / len(self.stations)) * 0.2
        total_score = connectivity_score + component_score + isolation_score
        return max(0.0, min(1.0, total_score))

    def set_rl_config(self, num_stations: int = None, num_paths: int = None, num_metros: int = None, num_station_types: int = None):
        """Configure mediator for RL training"""
        if num_stations is not None:
            self.num_stations = num_stations
        if num_paths is not None:
            self.num_paths = num_paths
            self.path_buttons = get_path_buttons(self.num_paths)
            self.buttons = [*self.path_buttons]
        if num_metros is not None:
            self.num_metros = num_metros
        if num_station_types is not None:
            self.num_station_types = min(max(num_station_types, 2), 4)
        else:
            self.num_station_types = 3
        
        # Reinitialize with new config
        self.stations = self._get_random_stations_with_types(self.num_stations, self.num_station_types)
        self.metros = []
        self.paths = []
        self.passengers = []
        
        # Reset path colors
        self.path_colors = {}
        for i in range(self.num_paths):
            color = hue_to_rgb(i / (self.num_paths + 1))
            self.path_colors[color] = False
        self.path_to_color = {}
        
        # Reset state
        self.time_ms = 0
        self.steps = 0
        self.steps_since_last_spawn = self.passenger_spawning_interval_step + 1
        self.score = 0
        self.travel_plans = {}
        self.path_to_button = {}  # Clear button mapping for RL mode

    def _get_random_stations_with_types(self, num_stations: int, num_station_types: int) -> List[Station]:
        """Generate random stations using only the specified number of station types"""
        from config import station_shape_type_list, station_color, station_size, station_min_distance
        from utils import get_random_position, get_shape_from_type
        from geometry.type import ShapeType
        from entity.get_entity import is_valid_station_position
        
        # Use only the first num_station_types from the list
        available_types = station_shape_type_list[:num_station_types]
        
        stations = []
        max_attempts_per_station = 100  # Prevent infinite loops
        
        # Ensure at least one station of each type (if we have enough stations)
        for i in range(min(num_stations, num_station_types)):
            shape_type = available_types[i]
            attempts = 0
            station = None
            
            while attempts < max_attempts_per_station:
                shape = get_shape_from_type(shape_type, station_color, station_size)
                position = get_random_position(screen_width, screen_height)
                candidate_station = Station(shape, position)
                
                # Check if position is valid (far enough from existing stations)
                if is_valid_station_position(candidate_station.position, stations, station_min_distance):
                    station = candidate_station
                    break
                
                attempts += 1
            
            if station:
                stations.append(station)
            else:
                # Fallback: add the last candidate even if it's too close
                print(f"Warning: Could not find valid position for station type {shape_type} after {max_attempts_per_station} attempts")
                stations.append(candidate_station)
        
        # Fill remaining stations randomly from available types
        for _ in range(num_stations - len(stations)):
            shape_type = random.choice(available_types)
            attempts = 0
            station = None
            
            while attempts < max_attempts_per_station:
                shape = get_shape_from_type(shape_type, station_color, station_size)
                position = get_random_position(screen_width, screen_height)
                candidate_station = Station(shape, position)
                
                # Check if position is valid (far enough from existing stations)
                if is_valid_station_position(candidate_station.position, stations, station_min_distance):
                    station = candidate_station
                    break
                
                attempts += 1
            
            if station:
                stations.append(station)
            else:
                # Fallback: add the last candidate even if it's too close
                print(f"Warning: Could not find valid position for station after {max_attempts_per_station} attempts")
                stations.append(candidate_station)
        
        # Shuffle to randomize positions of guaranteed types
        random.shuffle(stations)
        
        return stations
