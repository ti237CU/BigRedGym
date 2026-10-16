import argparse
import sys

import numpy as np
import torch

from play import get_play_args, setup


def get_eval_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hold_s", type=float, default=4.0, help="seconds per level")
    parser.add_argument("--cycles", type=int, default=2, help="low-high repeats")
    parser.add_argument("--settle_s", type=float, default=2.0, help="initial settle")
    parser.add_argument("--out", type=str, default="height_log.npz")
    eval_args, rest = parser.parse_known_args()
    return eval_args, get_play_args(rest)


def main():
    eval_args, play_args = get_eval_args()
    with torch.no_grad():
        env, runner = setup(play_args)
        if env.commands.shape[1] < 4:
            sys.exit(f"task {play_args.task!r} has no height command (commands[:, 3])")

        env.hold_height_command = True  # resets must not resample it
        env.commands[:, :3] = 0.0  # trot in place: grade height on its own
        low, high = env.command_ranges["base_height"]
        mid = 0.5 * (low + high)

        steps = lambda s: int(round(s / env.dt))  # noqa: E731
        schedule = [mid] * steps(eval_args.settle_s)
        for _ in range(eval_args.cycles):
            schedule += [low] * steps(eval_args.hold_s)
            schedule += [high] * steps(eval_args.hold_s)

        target, height, done = [], [], []
        for value in schedule:
            env.commands[:, 3] = value
            runner.set_actions(
                runner.actor_cfg["actions"],
                runner.get_inference_actions(),
                runner.actor_cfg["disable_actions"],
            )
            env.step()
            target.append(value)
            height.append(env.base_height[:, 0].cpu().numpy().copy())
            done.append((env.terminated | env.timed_out).cpu().numpy().copy())
            env.check_exit()

        t = np.arange(len(schedule)) * env.dt
        np.savez(
            eval_args.out,
            t=t,
            target=np.asarray(target),
            height=np.stack(height),  # (T, num_envs)
            done=np.stack(done),
            settle_s=eval_args.settle_s,
        )
        n_fail = int(np.stack(done).any(axis=0).sum())
        print(f"saved {eval_args.out}: {len(t)} steps x {env.num_envs} envs, "
              f"{n_fail} env(s) terminated during the run")


if __name__ == "__main__":
    main()