# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:13:42 2025

@author: manuw
"""

import time
import random
import json

def generate_random_coordinates():
    num_points = random.randint(2, 5)
    coordinates = []
    for _ in range(num_points):
        lat = random.uniform(35.0, 60.0)   # Breitengrad
        lon = random.uniform(-10.0, 30.0)  # Längengrad
        coordinates.append({'latitude': lat, 'longitude': lon})
    return coordinates

def write_coordinates_to_file():
    while True:
        coords = generate_random_coordinates()
        with open('live_coords.txt', 'w') as f:
            json.dump(coords, f)
        print(f"{len(coords)} Koordinaten gespeichert.")
        time.sleep(10)

if __name__ == "__main__":
    write_coordinates_to_file()
