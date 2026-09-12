"""Snaps the spectator to the ego's RGB camera transform every tick.

Usage:
    .venv/bin/python scripts/follow_ego_cam.py
"""
import time
import carla

c = carla.Client("127.0.0.1", 2000)
c.set_timeout(10.0)

while True:
    world = c.get_world()  # re-fetch: picks up a map switch on any client
    spectator = world.get_spectator()
    cams = world.get_actors().filter("sensor.camera.rgb")
    if not cams:
        print("no ego RGB camera found yet, waiting...")
        time.sleep(1.0)
        continue
    cam = cams[0]
    spectator.set_transform(cam.get_transform())
    time.sleep(0.05)
