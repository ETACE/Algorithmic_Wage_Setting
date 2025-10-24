# ABM Wage Competition – DQN & Q‑Table Baselines

This repository contains a compact agent‑based simulation of firms competing for workers on a circular space. 
Agents learn **wage‑setting policies** either with a **Deep Q‑Network (DQN)** or a **tabular Q‑learning** baseline.
The code is organized to let you switch between models and predefined **scenarios (1–8)** that toggle replay, target‑net sync, and a bidding vs. take‑it‑or‑leave‑it mechanism.

> **At a glance**
> - Environment: workers are placed on a circle; moving farther is costly (\(\textit{effort}\)).  
> - Firms post wages; workers choose based on payoff (wage minus travel effort).  
> - Learning: either DQN (Keras/TensorFlow) or a pure Q‑table.  
> - Rich diagnostics: best‑response maps, per‑tick Q‑values, firm performance logs.

---

## Project structure

```
.
├── Model.py             # Entry point – CLI, simulation loop & scenario wiring
├── Globals.py           # Global hyperparameters & defaults
├── Neural_Network.py    # Keras policy/target network builders & helpers (DQN)
├── QTable_Agent.py      # Tabular Q-learning agent + policy shim
├── Firm.py              # Firm agent logic (act, learn, memory)
├── Worker.py            # Worker agent logic & choice
├── Space.py             # Minimal container for agents (iterable)
├── DataHandling.py      # CSV logging: Q-values over time, firm performance
└── PlotUtils.py         # Best-response (BR) map & plotting helpers
```

---

## Scenarios (1–8)

Two economic interaction modes:
- **model_type = 0**: *Take‑it‑or‑leave‑it (TIOLI)* wage posting
- **model_type = 1**: *Bidding*

Replay & target‑network toggles define eight scenarios:

1. **TIOLI + replay** (symmetric)  
2. **TIOLI, no replay** (symmetric)  
3. **Bidding + replay** (symmetric)  
4. **Bidding, no replay** (symmetric)  
5. **TIOLI, asymmetric fast‑sync firm 0** – firm **0** uses online updates (no replay, mini‑batch=1) and **forces target‑net sync every iteration**; others use replay/sync.  
6. **Bidding, asymmetric fast‑sync firm 0** – as in 5, but bidding.  
7. **TIOLI, asymmetric fast‑sync firm 1** – firm **1** is the fast‑sync/online learner.  
8. **Bidding, asymmetric fast‑sync firm 1** – as in 7, but bidding.

> These correspond to the comment block in `Model.py`. See the function that wires `set_simulation_scenario` for exact parameterizations in your copy.

---

## Installation

**Python** 3.9+ recommended.

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install dependencies
pip install -U pip wheel
pip install numpy pandas matplotlib tensorflow keras
```
> If you only run the Q‑table baseline, `tensorflow/keras` is still imported for a small policy shim; 
> however, you can strip that dependency if desired (see `QTable_Agent.py`).

---

## Quick start

Run a symmetric DQN TIOLI scenario (1):

```bash
python Model.py --set_simulation_scenario 1
```

Use the tabular baseline (set the flag to 1) and plot the best‑response map occasionally:

```bash
python Model.py --set_simulation_scenario 2 --qtable 1 --plot_br 10000
```

Try an **asymmetric** setup where firm 0 learns online with per‑iteration target sync (scenario 5):

```bash
python Model.py --set_simulation_scenario 5
```

> Use `-h/--help` for the full list of flags and their defaults in your version.

---

## Key CLI flags (typical)

> Names and defaults come from `Model.py`; your copy may differ slightly.

- `--set_simulation_scenario INT` – select scenario (1–8).
- `--qtable {0,1}` – `1` uses the tabular baseline, `0` uses DQN.
- `--learning_rate FLOAT` – optimizer step size (DQN).
- `--beta FLOAT` – exploration decay parameter for ε‑greedy.
- `--effort FLOAT` – travel cost per unit distance on the circle.
- `--random_productivity {0,1}` – draw ±`delta_productivity` shocks each period if 1.
- `--delta_productivity FLOAT` – amplitude for productivity shock/asymmetry.
- `--num_firms INT` – number of competing firms.
- `--plot_br INT` – if >0, render the best‑response map every N iterations.
- Additional flags may exist (e.g., seed, plotting on/off); see `Model.py`.

---

## Outputs

`DataHandling.py` writes CSVs to the working directory:

- `q_values_over_time.csv` – per‑tick Q‑values by firm (for diagnostics/plots)
- `firm_performance.csv` – rewards, wages, and other performance metrics

`PlotUtils.py` can render **best‑response (BR) maps** and other visuals.

---

## How it works (high level)

1. **Environment**  
   Workers live on a circle. Accepting a job implies a travel cost proportional to distance (controlled by `--effort`).

2. **Firms**  
   Each firm sets a wage from a discrete grid. In TIOLI, workers accept if net utility is highest; in Bidding, firms compete more explicitly.

3. **Learning**  
   - **DQN**: policy/target networks (`Neural_Network.py`), replay memory, target syncs.  
   - **Tabular**: standard Q‑learning update with ε‑greedy; a small policy shim makes it compatible with the same logging and plotting code.

4. **Logging**  
   Q‑values and firm performance are persisted every tick (append mode after the first tick).

---

## Reproducibility tips

- Set a fixed random seed.
- Keep the wage grid and environment parameters constant while comparing models.
- For DQN sensitivity checks: vary replay size, target‑sync frequency, and learning rate.

---

## Troubleshooting

- **No plots appear**: ensure `--plot_br` is set to a positive number and `matplotlib` is installed.
- **TensorFlow errors (Q‑table only)**: if you want to avoid TF altogether, you can replace the policy shim tensor return with NumPy arrays and adjust calls in `DataHandling.py` accordingly.
- **CSV not updating**: the files append after the first iteration; delete them if you want a fresh run.

---


