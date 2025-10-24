import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

def br_map_all_states(policy_model, firm_id, actions, own_grid=None, comp_grid=None):
    """
    Compute RL best response for ALL state pairs:
      s = (own_state = o, comp_state = c) for o in own_grid, c in comp_grid,
    and BR(s) = argmax_a Q(s, a).

    Returns:
      br_idx    : (len(comp_grid), len(own_grid)) int matrix of action indices
      br_wages  : same shape, chosen wages (actions[br_idx])
      actions_used : possibly truncated action grid (align with model's output)
      own_grid, comp_grid : the grids actually used (np.array)
    """
    import Globals

    actions   = np.asarray(actions, dtype=float)
    own_grid  = np.asarray(own_grid if own_grid is not None else actions, dtype=float)
    comp_grid = np.asarray(comp_grid if comp_grid is not None else actions, dtype=float)

    # State layout (model_type==0: vector of firm wages ordered by firm_id)
    if getattr(Globals, "model_type", 0) == 0:
        num_firms = getattr(Globals, "num_firms", 2)
        own_idx   = int(firm_id)
        comp_idx  = (own_idx + 1) % num_firms
        inp_dim   = max(own_idx, comp_idx) + 1
    else:
        own_idx, comp_idx, inp_dim = 0, None, 1

    #Build the model once to infer output dimension
    if not getattr(policy_model, "built", False):
        _ = policy_model(tf.zeros((1, inp_dim), dtype=tf.float32))
    out_dim = int(policy_model(tf.zeros((1, inp_dim), dtype=tf.float32)).shape[-1])

    # Align to model's output dimension
    mA = min(out_dim, len(actions))
    actions_used = actions[:mA]

    # Allocate result
    H, W = len(comp_grid), len(own_grid)
    br_idx   = np.zeros((H, W), dtype=int)
    br_wages = np.zeros((H, W), dtype=float)

    # Build once and reuse base state
    base = np.zeros((inp_dim,), dtype=np.float32)

    # Loop all state pairs (competitor row, own-state column)
    for r, pc in enumerate(comp_grid):
        for c, own_state in enumerate(own_grid):
            s = base.copy()
            if comp_idx is not None:
                s[comp_idx] = pc
            if own_idx is not None:
                s[own_idx] = own_state

            q = policy_model(tf.convert_to_tensor(s[None, :], dtype=tf.float32)).numpy()[0]
            q = q[:mA]  # clamp to actions_used length
            a_star = int(np.argmax(q))
            br_idx[r, c]   = a_star
            br_wages[r, c] = actions_used[a_star]

    return br_idx, br_wages, actions_used, own_grid, comp_grid


def plot_br_map(br_wages, own_grid, comp_grid, outdir,iteration, fname="firm", fmt="pdf"):
    """
    Plot the BR map as a discrete heatmap where color = chosen wage (continuous legend, discrete steps).
    Rows = competitor state, Cols = own state.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        br_wages, origin="lower", aspect="auto",
        extent=[own_grid[0], own_grid[-1], comp_grid[0], comp_grid[-1]],
        interpolation="nearest",
          vmin=4, vmax=22    # no smoothing -> discreteness is visible
    )
    ax.set_xlabel("Own state")
    ax.set_ylabel("Competitor state")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Best-response wage ")
    fig.tight_layout()
    out_path = os.path.join(outdir, f"{fname}_rl_br_map_{iteration}.{fmt}")
    fig.savefig(out_path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    #print(f"[BR MAP] Saved {out_path}")
