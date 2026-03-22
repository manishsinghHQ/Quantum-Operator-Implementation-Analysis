import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
import time

st.set_page_config(page_title="QGA vs GA Simulator", layout="wide")

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Controls")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENS = st.sidebar.slider("Generations", 50, 300, 150)
MUT_RATE = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01)

dataset_type = st.sidebar.selectbox("Dataset", ["Easy", "Hard"])
algo_choice = st.sidebar.selectbox(
    "Algorithm",
    ["QGA Rotation", "QGA + Crossover", "Classical GA"]
)

# =========================
# Knapsack Data
# =========================
def generate_knapsack(n=50, hard=False):
    np.random.seed(42 if not hard else 99)
    weights = np.random.randint(1, 20, n)
    values = np.random.randint(10, 100, n)
    capacity = int(np.sum(weights) * (0.5 if not hard else 0.3))
    return weights, values, capacity

weights, values, capacity = generate_knapsack(50, dataset_type == "Hard")

st.sidebar.write(f"Capacity: {capacity}")
st.sidebar.write(f"Items: {len(weights)}")

# =========================
# Fitness
# =========================
def fitness(sol):
    w = np.sum(sol * weights)
    if w > capacity:
        return 0
    return np.sum(sol * values)

# =========================
# QGA FUNCTIONS
# =========================
def init_theta():
    return np.full((POP_SIZE, len(weights)), np.pi / 4)

def measure(theta):
    probs = np.sin(theta) ** 2
    return (np.random.rand(*probs.shape) < probs).astype(int)

def q_rotation(theta, x, best):
    delta = 0.03
    for i in range(len(theta)):
        if x[i] == 0 and best[i] == 1:
            theta[i] += delta
        elif x[i] == 1 and best[i] == 0:
            theta[i] -= delta
    return theta

def q_mutation(theta):
    mask = np.random.rand(len(theta)) < MUT_RATE
    theta[mask] += np.random.uniform(-0.1, 0.1, np.sum(mask))
    return theta

def q_crossover_swap(p1, p2):
    point = np.random.randint(len(p1))
    return (
        np.concatenate([p1[:point], p2[point:]]),
        np.concatenate([p2[:point], p1[point:]])
    )

def q_crossover_avg(p1, p2):
    avg = (p1 + p2) / 2
    return avg.copy(), avg.copy()

# =========================
# Classical GA
# =========================
def init_population():
    return np.random.randint(0, 2, (POP_SIZE, len(weights)))

def crossover(p1, p2):
    point = np.random.randint(len(p1))
    return (
        np.concatenate([p1[:point], p2[point:]]),
        np.concatenate([p2[:point], p1[point:]])
    )

def mutate(sol):
    mask = np.random.rand(len(sol)) < MUT_RATE
    sol[mask] = 1 - sol[mask]
    return sol

# =========================
# Visualization
# =========================
def plot_circle(theta):
    x = np.cos(theta.flatten())
    y = np.sin(theta.flatten())

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=5)

    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_patch(circle)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Q-bit States (Unit Circle)")
    return fig

# =========================
# Metrics
# =========================
def convergence_point(history):
    best = max(history)
    for i, v in enumerate(history):
        if v == best:
            return i

def stability(history):
    window = min(20, len(history))
    return np.std(history[-window:])

def efficiency(best, t):
    return round(best / (t + 1e-6), 2)

# =========================
# QGA Simulation (FIXED)
# =========================
def run_qga(use_crossover=False, generate_gif=True):
    theta = init_theta()

    best_fit = -1   # FIX
    best_sol = None

    fitness_history = []
    avg_history = []

    frames = []
    progress = st.progress(0)

    for g in range(GENS):
        population = measure(theta)
        fits = np.array([fitness(ind) for ind in population])

        best_idx = np.argmax(fits)

        # FIX: always update safely
        if best_sol is None or fits[best_idx] > best_fit:
            best_fit = fits[best_idx]
            best_sol = population[best_idx]

        fitness_history.append(best_fit)
        avg_history.append(np.mean(fits))

        for i in range(POP_SIZE):
            if best_sol is not None:  # FIX
                theta[i] = q_rotation(theta[i], population[i], best_sol)
            theta[i] = q_mutation(theta[i])
            theta[i] = np.clip(theta[i], 0, np.pi)

        if use_crossover:
            for i in range(0, POP_SIZE, 2):
                if i + 1 < POP_SIZE:
                    if np.random.rand() < 0.5:
                        theta[i], theta[i+1] = q_crossover_swap(theta[i], theta[i+1])
                    else:
                        theta[i], theta[i+1] = q_crossover_avg(theta[i], theta[i+1])

        if generate_gif and g % 3 == 0:
            fig = plot_circle(theta)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)

        progress.progress((g + 1) / GENS)

    gif_bytes = None
    if generate_gif and len(frames) > 0:
        gif_bytes = io.BytesIO()
        imageio.mimsave(gif_bytes, frames, format="GIF", duration=0.08)
        gif_bytes.seek(0)

    return fitness_history, avg_history, best_fit, best_sol, gif_bytes

# =========================
# Classical GA
# =========================
def run_ga():
    pop = init_population()
    best_fit = -1
    best_sol = None
    fitness_history = []
    avg_history = []

    for g in range(GENS):
        fits = np.array([fitness(ind) for ind in pop])

        idx = np.argmax(fits)
        if best_sol is None or fits[idx] > best_fit:
            best_fit = fits[idx]
            best_sol = pop[idx]

        fitness_history.append(best_fit)
        avg_history.append(np.mean(fits))

        new_pop = []
        for _ in range(POP_SIZE // 2):
            p1, p2 = pop[np.random.choice(POP_SIZE, 2)]
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            new_pop.append(mutate(c2))

        pop = np.array(new_pop)

    return fitness_history, avg_history, best_fit, best_sol

# =========================
# UI
# =========================
st.title("⚛️ Quantum Genetic Algorithm Simulator")

if st.button("🚀 Run Simulation"):

    if algo_choice == "Classical GA":
        hist, avg_hist, best, sol = run_ga()
        gif = None
    else:
        hist, avg_hist, best, sol, gif = run_qga("Crossover" in algo_choice, True)

    st.success(f"Best Fitness: {best}")

    st.subheader("📈 Convergence Plot")
    st.line_chart({
        "Best Fitness": hist,
        "Average Fitness": avg_hist
    })

    if gif:
        st.subheader("🎥 Q-bit Evolution")
        st.image(gif)

    st.subheader("🎯 Best Solution")
    selected = np.where(sol == 1)[0]
    st.write(f"Selected Items: {selected}")
    st.write(f"Items Selected: {np.sum(sol)} / {len(sol)}")
    st.write(f"Total Weight: {np.sum(sol * weights)} / {capacity}")
    st.write(f"Total Value: {np.sum(sol * values)}")

# =========================
# Compare All
# =========================
if st.button("📊 Compare All Algorithms"):

    start = time.time()
    h1, a1, b1, _, _ = run_qga(False, generate_gif=False)
    t1 = time.time() - start

    start = time.time()
    h2, a2, b2, _, _ = run_qga(True, generate_gif=False)
    t2 = time.time() - start

    start = time.time()
    h3, a3, b3, _ = run_ga()
    t3 = time.time() - start

    st.subheader("📈 Comparison Graph")
    st.line_chart({
        "QGA Rotation": h1,
        "QGA + Crossover": h2,
        "Classical GA": h3
    })

    st.subheader("📋 Comparison Table")
    st.table({
        "Algorithm": ["QGA Rotation", "QGA + Crossover", "Classical GA"],
        "Best Fitness": [b1, b2, b3],
        "Convergence Gen": [
            convergence_point(h1),
            convergence_point(h2),
            convergence_point(h3)
        ],
        "Stability": [
            round(stability(h1), 2),
            round(stability(h2), 2),
            round(stability(h3), 2)
        ],
        "Time (s)": [
            round(t1, 2),
            round(t2, 2),
            round(t3, 2)
        ],
        "Efficiency": [
            efficiency(b1, t1),
            efficiency(b2, t2),
            efficiency(b3, t3)
        ]
    })

    st.subheader("🧠 Insights")

    if b2 >= b1 and b2 >= b3:
        st.success("QGA + Crossover performed best overall 🚀")
    elif b1 >= b2 and b1 >= b3:
        st.success("QGA Rotation performed best ⚛️")
    else:
        st.success("Classical GA performed best 🧬")
