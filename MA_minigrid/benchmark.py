
#!/usr/bin/env python3

from __future__ import annotations

import time

import gymnasium as gym

from MA_minigrid.wrappers import SingleAgentWrapper, RGBImgObsWrapper

def benchmark(env_id, num_resets, num_frames):
    env = gym.make(env_id, render_mode="rgb_array")
    # Benchmark env.reset
    t0 = time.time()
    for i in range(num_resets):
        env.reset()
    t1 = time.time()
    dt = t1 - t0
    reset_time = (1000 * dt) / num_resets

    # Benchmark rendering
    t0 = time.time()
    for i in range(num_frames):
        env.render()
    t1 = time.time()
    dt = t1 - t0
    frames_per_sec = num_frames / dt

    # Create an environment with an RGB agent observation
    env = gym.make(env_id, render_mode="rgb_array")
    env = SingleAgentWrapper(env)

    env.reset()
    # Benchmark rendering in agent view
    t0 = time.time()
    for i in range(num_frames):
        _ = env.step(0)
    t1 = time.time()
    dt = t1 - t0
    agent_view_fps = num_frames / dt

    print(f"Env reset time: {reset_time:.1f} ms")
    print(f"Rendering FPS : {frames_per_sec:.0f}")
    print(f"Agent view FPS: {agent_view_fps:.0f}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        dest="env_id",
        help="gym environment to load",
        default="MiniGrid-LavaGapS7-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        help="number of times to reset the environment for benchmarking",
        default=100,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames to test rendering for",
        default=5000,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )

    args = parser.parse_args()
    benchmark(args.env_id, args.num_resets, args.num_frames)
