from typing import List

from config import screen_height, screen_width, station_min_distance
from entity.metro import Metro
from entity.station import Station
from utils import get_random_position, get_random_station_shape


def get_random_station() -> Station:
    shape = get_random_station_shape()
    position = get_random_position(screen_width, screen_height)
    return Station(shape, position)


def is_valid_station_position(position, existing_stations: List[Station], min_distance: float) -> bool:
    """Check if a position is far enough from all existing stations."""
    for station in existing_stations:
        if position.distance_to(station.position) < min_distance:
            return False
    return True


def get_random_stations(num: int) -> List[Station]:
    stations: List[Station] = []
    max_attempts_per_station = 100  # Prevent infinite loops
    
    for _ in range(num):
        attempts = 0
        station = None
        
        while attempts < max_attempts_per_station:
            candidate_station = get_random_station()
            
            # Check if position is valid (far enough from existing stations)
            if is_valid_station_position(candidate_station.position, stations, station_min_distance):
                station = candidate_station
                break
            
            attempts += 1
        
        # If we found a valid position, add it; otherwise add anyway to avoid infinite loop
        if station:
            stations.append(station)
        else:
            # Fallback: add the last candidate even if it's too close
            # This prevents the game from failing to start if we can't find enough valid positions
            print(f"Warning: Could not find valid position for station after {max_attempts_per_station} attempts")
            stations.append(candidate_station)
    
    return stations


def get_metros(num: int) -> List[Metro]:
    metros: List[Metro] = []
    for _ in range(num):
        metros.append(Metro())
    return metros
