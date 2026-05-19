import numpy as np
import pennylane as qml

from model import nQubits, nSteps, nFeatures, get_partition

DARK_BG = '#0d1117'
DARK_PANEL = '#161b22'
DARK_FG = '#c9d1d9'
DARK_FG_DIM = '#8b949e'
DARK_BTN = '#21262d'
DARK_BTN_HOV = '#30363d'
ACCENT_BLUE = '#58a6ff'
ACCENT_ORANGE = '#f78166'
ACCENT_GREEN = '#56d364'
GRID_COLOR = '#30363d'

CLASS_COLORS = [
    '#ff6b6b', '#5fd068', '#ffd93d', '#54a0ff', '#ff9f43',
    '#a55eea', '#26c6da', '#ec407a', '#9ccc65', '#ab47bc'
]


def build_gate_program():
    program = []
    for i in range(nQubits):
        program.append(('RY', i))
    for step in range(nSteps):
        for i in range(nQubits):
            program.append(('RY', i))
        for c, t in get_partition(step):
            program.append(('CNOT', c, t))
    return program


gate_program = build_gate_program()
n_total_gates = len(gate_program)


def _build_gate_layer():
    layers = []
    for i, op in enumerate(gate_program):
        if op[0] != 'RY':
            layers.append(-1)
            continue
        if i < nQubits:
            layers.append(0)
        else:
            macro_id = (i - nQubits) // (nQubits + nQubits // 2)
            layers.append(macro_id + 1)
    return layers


gate_layer = _build_gate_layer()

gate_dev = qml.device("lightning.qubit", wires=nQubits, shots=None)


def make_gate_snapshot_qnode(gate_count):
    @qml.qnode(gate_dev, interface="numpy")
    def circ(features, weights):
        for i in range(min(gate_count, n_total_gates)):
            op = gate_program[i]
            if op[0] == 'RY':
                layer = gate_layer[i]
                qml.RY(np.pi * features[op[1] % nFeatures] + weights[layer, op[1]],
                       wires=op[1])
            else:
                qml.CNOT(wires=[op[1], op[2]])
        return qml.state()
    return circ


gate_snapshot_qnodes = [make_gate_snapshot_qnode(g) for g in range(n_total_gates + 1)]


def partial_trace(sv, n_q, keep):
    rho = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            for cfg in range(2 ** (n_q - 1)):
                idx1 = 0
                idx2 = 0
                bp = 0
                for q in range(n_q):
                    if q == keep:
                        idx1 = idx1 * 2 + a
                        idx2 = idx2 * 2 + b
                    else:
                        bit = (cfg >> bp) & 1
                        idx1 = idx1 * 2 + bit
                        idx2 = idx2 * 2 + bit
                        bp += 1
                rho[a, b] += sv[idx1] * sv[idx2].conj()
    return rho


def bloch_xyz(sv, n_q, qubit):
    rho = partial_trace(sv, n_q, qubit)
    x = 2.0 * rho[0, 1].real
    y = 2.0 * rho[1, 0].imag
    z = (rho[0, 0] - rho[1, 1]).real
    return float(x), float(y), float(z)


def basis_state_probs(sv):
    return np.abs(sv) ** 2


def basis_state_labels(n_q):
    out = []
    for i in range(2 ** n_q):
        out.append("|" + bin(i)[2:].zfill(n_q) + ">")
    return out


def draw_circuit_figure(features=None, weights=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    if features is None:
        features = np.zeros(nFeatures)
    features = np.asarray(features, dtype=np.float64).flatten()
    if weights is None:
        weights = np.zeros((nSteps + 1, nQubits))
    weights = np.asarray(weights, dtype=np.float64)

    gates = []
    col = 0
    for i in range(nQubits):
        gates.append((col, 'RY', i, 0))
        col += 1
    for step in range(nSteps):
        for i in range(nQubits):
            gates.append((col, 'RY', i, step + 1))
            col += 1
        for c, t in get_partition(step):
            gates.append((col, 'CNOT', c, t))
            col += 1
    n_cols = col

    fig, ax = plt.subplots(figsize=(max(18, n_cols * 0.7), 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    for q in range(nQubits):
        y = nQubits - 1 - q
        ax.plot([-0.5, n_cols - 0.5], [y, y], color=DARK_FG_DIM, lw=1.0, zorder=1)
        ax.text(-0.85, y, "q" + str(q), ha='right', va='center',
                color=ACCENT_BLUE, fontsize=12, weight='bold', family='monospace')

    for entry in gates:
        col_idx, gate_type, a, b = entry
        if gate_type == 'RY':
            qubit = a
            layer = b
            y = nQubits - 1 - qubit
            f_val = float(features[qubit % nFeatures])
            w_val = float(weights[layer, qubit])
            data_part = np.pi * f_val
            angle = data_part + w_val
            if abs(w_val) > 1e-3:
                bias_color = ACCENT_GREEN
            else:
                bias_color = DARK_FG_DIM

            box_w = 0.92
            box_h = 0.82
            rect = Rectangle((col_idx - box_w / 2, y - box_h / 2), box_w, box_h,
                             facecolor=DARK_PANEL, edgecolor=ACCENT_ORANGE,
                             linewidth=1.6, zorder=3)
            ax.add_patch(rect)
            ax.text(col_idx, y + 0.24, "RY", ha='center', va='center',
                    color=ACCENT_ORANGE, fontsize=9, weight='bold',
                    family='monospace', zorder=4)
            ax.text(col_idx, y + 0.05, f"{angle:+.2f}", ha='center', va='center',
                    color=DARK_FG, fontsize=8, weight='bold',
                    family='monospace', zorder=4)
            ax.text(col_idx, y - 0.13, f"d={data_part:+.2f}", ha='center', va='center',
                    color=ACCENT_BLUE, fontsize=6, family='monospace', zorder=4)
            ax.text(col_idx, y - 0.27, f"t={w_val:+.2f}", ha='center', va='center',
                    color=bias_color, fontsize=6, family='monospace', zorder=4)
        else:
            ctrl = a
            tgt = b
            y_c = nQubits - 1 - ctrl
            y_t = nQubits - 1 - tgt
            ax.plot([col_idx, col_idx], [min(y_c, y_t), max(y_c, y_t)],
                    color=ACCENT_BLUE, lw=1.8, zorder=2)
            ax.add_patch(Circle((col_idx, y_c), 0.13,
                                facecolor=ACCENT_BLUE, edgecolor=ACCENT_BLUE, zorder=4))
            ax.add_patch(Circle((col_idx, y_t), 0.27,
                                facecolor=DARK_BG, edgecolor=ACCENT_BLUE,
                                linewidth=1.8, zorder=4))
            ax.plot([col_idx - 0.20, col_idx + 0.20], [y_t, y_t],
                    color=ACCENT_BLUE, lw=1.4, zorder=5)
            ax.plot([col_idx, col_idx], [y_t - 0.20, y_t + 0.20],
                    color=ACCENT_BLUE, lw=1.4, zorder=5)

    block_w_macro = nQubits + nQubits // 2
    pos = nQubits
    boundaries = [pos]
    for s in range(nSteps):
        pos += block_w_macro
        boundaries.append(pos)
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color=GRID_COLOR, linestyle=':', linewidth=0.9, zorder=0)

    top_y = nQubits - 0.10
    ax.text(nQubits / 2 - 0.5, top_y, "init layer (L=0)",
            ha='center', va='bottom', color=DARK_FG_DIM,
            fontsize=10, style='italic')
    pos = nQubits
    for s in range(nSteps):
        ax.text(pos + block_w_macro / 2 - 0.5, top_y,
                "macro step " + str(s + 1) + " (L=" + str(s + 1) + ")",
                ha='center', va='bottom', color=DARK_FG_DIM,
                fontsize=10, style='italic')
        pos += block_w_macro

    legend = ("RY angle = d + t    "
              "d = pi * features[q mod 8] (blue)    "
              "t = trained bias (green)    "
              "CNOT: control dot -> target circle")
    ax.text(n_cols / 2, -1.15, legend, ha='center', va='top',
            color=DARK_FG_DIM, fontsize=9)

    ax.set_xlim(-1.7, n_cols)
    ax.set_ylim(-1.4, nQubits + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    return fig
