import gymnasium as gym
import pygame
import numpy as np

env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

obs, info = env.reset()
clock = pygame.time.Clock()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    steering = 0.0
    gas = 0.0
    brake = 0.0

    if keys[pygame.K_LEFT]:
        steering = -0.4
    elif keys[pygame.K_RIGHT]:
        steering = 0.4

    if keys[pygame.K_UP]:
        gas = 1.0

    if keys[pygame.K_DOWN]:
        brake = 0.8

    action = np.array(
        [steering, gas, brake],
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(50)

env.close()
pygame.quit()