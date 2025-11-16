import pygame
from shortuuid import uuid  # type: ignore

from config import passenger_max_waiting_time
from geometry.point import Point
from geometry.shape import Shape


class Passenger:
    def __init__(self, destination_shape: Shape) -> None:
        self.id = f"Passenger-{uuid()}"
        self.position = Point(0, 0)
        self.destination_shape = destination_shape
        self.is_at_destination = False
        self.waiting_time = 0  # in milliseconds
        self.max_waiting_time = passenger_max_waiting_time

    def __repr__(self) -> str:
        return f"{self.id}-{self.destination_shape.type}"

    def __hash__(self) -> int:
        return hash(self.id)
    
    def increment_waiting_time(self, dt_ms: int) -> None:
        """Increment the waiting time for this passenger."""
        if not self.is_at_destination:
            self.waiting_time += dt_ms
    
    def get_waiting_indicator_color(self) -> tuple:
        """Get the color indicator based on waiting time (green -> yellow -> red)."""
        # Normalize waiting time to 0-1 range
        ratio = min(self.waiting_time / self.max_waiting_time, 1.0)
        
        # Green (0, 255, 0) -> Yellow (255, 255, 0) -> Red (255, 0, 0)
        if ratio < 0.5:
            # Green to Yellow
            green_to_yellow = ratio * 2
            r = int(255 * green_to_yellow)
            g = 255
            b = 0
        else:
            # Yellow to Red
            yellow_to_red = (ratio - 0.5) * 2
            r = 255
            g = int(255 * (1 - yellow_to_red))
            b = 0
        
        return (r, g, b)

    def draw(self, surface: pygame.surface.Surface):
        # Update the passenger's color based on waiting time
        waiting_color = self.get_waiting_indicator_color()
        self.destination_shape.color = waiting_color
        
        # Draw the destination shape with the updated color
        self.destination_shape.draw(surface, self.position)
