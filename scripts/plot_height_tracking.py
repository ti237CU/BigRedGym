import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRANSIENT_S = 1.5  # excluded from the steady-state error after each step


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "height_log.npz"
    out = sys.argv[2] if len(sys.argv) > 2 else "height_tracking.pdf"
    d = np.load(path)
    t, target, height = d["t"], d["target"], d["height"]
    if height.ndim == 1:
        height = height[:, None]
    dt = t[1] - t[0]

    mean, std = height.mean(axis=1), height.std(axis=1)
    error = height - target[:, None]

    # Steady state: skip the initial settle and the transient after each step.
    steady = t >= float(d["settle_s"]) if "settle_s" in d else np.ones_like(t, bool)
    steps_at = np.flatnonzero(np.diff(target) != 0) + 1
    shortest = np.diff(np.r_[steps_at, len(t)]).min() * dt if len(steps_at) else 1e9
    transient = min(TRANSIENT_S, 0.5 * shortest)  # never blank a whole level
    for i in steps_at:
        steady[i : i + int(transient / dt)] = False
    rmse_ss = np.sqrt(np.mean(error[steady] ** 2))

    levels = np.unique(target[steady])
    level_lines = []
    for level in levels:
        sel = steady & (target == level)
        level_lines.append(
            f"cmd {level:.2f} m -> {height[sel].mean():.3f} "
            f"± {height[sel].std():.3f} m"
        )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    ax1.fill_between(t, mean - std, mean + std, color="#2563eb", alpha=0.2, lw=0)
    ax1.plot(t, mean, color="#2563eb", lw=1.8,
             label=f"actual (mean ± std, {height.shape[1]} envs)")
    ax1.plot(t, height[:, 0], color="#2563eb", lw=0.7, alpha=0.6, label="env 0")
    ax1.step(t, target, where="post", color="0.3", ls="--", lw=1.5, label="commanded")
    ax1.set_ylabel("base height [m]")
    ax1.set_title(
        f"Go2 base-height tracking | steady-state RMSE {100 * rmse_ss:.1f} cm",
        fontsize=10,
    )
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.text(0.01, 0.03, "\n".join(level_lines), transform=ax1.transAxes,
             fontsize=8, family="monospace", va="bottom")
    ax1.grid(alpha=0.3)

    ax2.plot(t, 100 * error.mean(axis=1), color="#dc2626", lw=1.2)
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_ylabel("mean error [cm]")
    ax2.set_xlabel("time [s]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out} (steady-state RMSE {rmse_ss:.4f} m)")
    for line in level_lines:
        print("  " + line)


if __name__ == "__main__":
    main()