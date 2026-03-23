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
GENS = st.sidebar.slider("Generations", 50, 300, 200)
MUT_RATE = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01)

dataset_type = st.sidebar.selectbox("Dataset", ["Easy", "Hard"])
algo_choice = st.sidebar.selectbox(
    "Algorithm",
    ["QGA Rotation", "QGA + Crossover", "Classical GA"]
)

# =========================
# Knapsack
# =========================
def generate_knapsack(n=50, hard=False):
    np.random.seed(42 if not hard else 99)
    weights = np.random.randint(1, 20, n)
    values = np.random.randint(10, 100, n)
    capacity = int(np.sum(weights) * (0.5 if not hard else 0.3))
    return weights, values, capacity

weights, values, capacity = generate_knapsack(50, dataset_type == "Hard")

# =========================
# Fitness
# =========================
def fitness(sol):
    w = np.sum(sol * weights)
    if w > capacity:
        return 0
    return np.sum(sol * values)

# =========================
# QBIT REPRESENTATION
# =========================
def init_qbit():
    alpha = np.full((POP_SIZE, len(weights)), 1/np.sqrt(2))
    beta = np.full((POP_SIZE, len(weights)), 1/np.sqrt(2))
    return alpha, beta

def measure(alpha, beta):
    probs = beta ** 2
    return (np.random.rand(*probs.shape) < probs).astype(int)

# =========================
# Q-ROTATION (CORE)
# =========================
def q_rotation(alpha, beta, x, best):
    delta = 0.02

    for i in range(len(alpha)):
        if x[i] == best[i]:
            continue

        direction = +1 if (x[i] == 0 and best[i] == 1) else -1

        a, b = alpha[i], beta[i]

        alpha[i] = a * np.cos(direction * delta) - b * np.sin(direction * delta)
        beta[i]  = a * np.sin(direction * delta) + b * np.cos(direction * delta)

    return alpha, beta

# =========================
# Q-MUTATION
# =========================
def q_mutation(alpha, beta):
    mask = np.random.rand(len(alpha)) < MUT_RATE
    noise = np.random.uniform(-0.05, 0.05, np.sum(mask))

    for idx, n in zip(np.where(mask)[0], noise):
        a, b = alpha[idx], beta[idx]
        alpha[idx] = a * np.cos(n) - b * np.sin(n)
        beta[idx]  = a * np.sin(n) + b * np.cos(n)

    return alpha, beta

# =========================
# Q-CROSSOVER
# =========================
def q_crossover_swap(a1, b1, a2, b2):
    point = np.random.randint(len(a1))
    c1_a = np.concatenate([a1[:point], a2[point:]])
    c1_b = np.concatenate([b1[:point], b2[point:]])
    c2_a = np.concatenate([a2[:point], a1[point:]])
    c2_b = np.concatenate([b2[:point], b1[point:]])
    return c1_a, c1_b, c2_a, c2_b

def q_crossover_avg(a1, b1, a2, b2):
    c_a = (a1 + a2) / 2
    c_b = (b1 + b2) / 2
    norm = np.sqrt(c_a**2 + c_b**2)
    return c_a/norm, c_b/norm, c_a/norm, c_b/norm

def q_crossover_interference(a1, b1, a2, b2):
    c_a = (a1 + a2) / 2
    c_b = (b1 + b2) / 2
    norm = np.sqrt(c_a**2 + c_b**2)
    return c_a/norm, c_b/norm, c_a/norm, c_b/norm

# =========================
# BLOCH VISUALIZATION
# =========================
def plot_bloch(alpha, beta):
    x = 2 * alpha.flatten() * beta.flatten()
    z = alpha.flatten()**2 - beta.flatten()**2

    fig, ax = plt.subplots()
    ax.scatter(x, z, s=5)

    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_patch(circle)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("Bloch Projection (X-Z Plane)")
    return fig

# =========================
# METRICS
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
# QGA RUN
# =========================
def run_qga(use_crossover=False, generate_gif=True):
    alpha, beta = init_qbit()

    best_fit = -1
    best_sol = None

    fitness_history = []
    avg_history = []
    frames = []

    progress = st.progress(0)

    for g in range(GENS):
        population = measure(alpha, beta)
        fits = np.array([fitness(ind) for ind in population])

        idx = np.argmax(fits)

        if best_sol is None or fits[idx] > best_fit:
            best_fit = fits[idx]
            best_sol = population[idx]

        fitness_history.append(best_fit)
        avg_history.append(np.mean(fits))

        for i in range(POP_SIZE):
            alpha[i], beta[i] = q_rotation(alpha[i], beta[i], population[i], best_sol)
            alpha[i], beta[i] = q_mutation(alpha[i], beta[i])

        if use_crossover:
            for i in range(0, POP_SIZE, 2):
                if i + 1 < POP_SIZE:
                    r = np.random.rand()
                    if r < 0.33:
                        alpha[i], beta[i], alpha[i+1], beta[i+1] = q_crossover_swap(
                            alpha[i], beta[i], alpha[i+1], beta[i+1])
                    elif r < 0.66:
                        alpha[i], beta[i], alpha[i+1], beta[i+1] = q_crossover_avg(
                            alpha[i], beta[i], alpha[i+1], beta[i+1])
                    else:
                        alpha[i], beta[i], alpha[i+1], beta[i+1] = q_crossover_interference(
                            alpha[i], beta[i], alpha[i+1], beta[i+1])

        if generate_gif and g % 3 == 0:
            fig = plot_bloch(alpha, beta)
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
# CLASSICAL GA
# =========================
def run_ga():
    pop = np.random.randint(0, 2, (POP_SIZE, len(weights)))

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
            point = np.random.randint(len(p1))

            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])

            mask1 = np.random.rand(len(c1)) < MUT_RATE
            mask2 = np.random.rand(len(c2)) < MUT_RATE

            c1[mask1] = 1 - c1[mask1]
            c2[mask2] = 1 - c2[mask2]

            new_pop.append(c1)
            new_pop.append(c2)

        pop = np.array(new_pop)

    return fitness_history, avg_history, best_fit, best_sol

# =========================
# UI
# =========================
st.title("⚛️ Quantum Genetic Algorithm Simulator (Research Grade)")

if st.button("🚀 Run Simulation"):

    if algo_choice == "Classical GA":
        hist, avg_hist, best, sol = run_ga()
        gif = None
    else:
        hist, avg_hist, best, sol, gif = run_qga("Crossover" in algo_choice)

    st.success(f"Best Fitness: {best}")

    st.line_chart({
        "Best Fitness": hist,
        "Average Fitness": avg_hist
    })

    if gif:
        st.image(gif)

# =========================
# COMPARE
# =========================
if st.button("📊 Compare All Algorithms"):

    start = time.time()
    h1, _, b1, _, _ = run_qga(False, False)
    t1 = time.time() - start

    start = time.time()
    h2, _, b2, _, _ = run_qga(True, False)
    t2 = time.time() - start

    start = time.time()
    h3, _, b3, _ = run_ga()
    t3 = time.time() - start

    st.line_chart({
        "QGA Rotation": h1,
        "QGA + Crossover": h2,
        "Classical GA": h3
    })

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
