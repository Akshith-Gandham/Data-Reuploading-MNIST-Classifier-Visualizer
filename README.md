# QCA MNIST Visualizer

An MNIST handwritten-digit classifier whose core nonlinearity is a **partitioned Quantum Cellular Automaton (QCA)** with data re-uploading, paired with a **bona fide Turing Machine** simulator and an interactive dark-mode visualizer.

Built as an honors contract for CSE 355 (Theory of Computation) at ASU.

---

## What it does

Draw a digit → hit **Predict** → watch two computational models process it in real time:

- **TM mode** — animates a formal Turing Machine scanning the raw binary image tape (784 bits), executing transitions from the δ table one step at a time, and committing encoder features at each region boundary.
- **QCA mode** — steps through all 28 elementary quantum gates (RY rotations + CNOT entanglers), showing the live basis-state probability histogram, the gate timeline, LDA and tSNE projections of the quantum state at each macro step, and a full circuit popup.

---

## Architecture

```
image (1×28×28)
    │
    ▼
Encoder — Linear(784→64) → ReLU → Linear(64→8) → tanh
    │  (features ∈ [-1, 1]⁸)
    ▼
QCA Layer — 4 qubits, 4 macro steps, data re-uploading
    │         28 gates: 4 init RY + 4×(4 RY + 2 CNOT)
    │         outputs: [⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩] ∈ [-1, 1]⁴
    ▼
Classifier — Linear(4→10) → logits
```

**Trainable parameters:** 50,818 total (50,768 encoder + 50 classifier). The QCA has **zero** variational parameters — all learning happens in the encoder that produces the RY angles.

### QCA structure

Each macro step re-uploads the 8 encoder features as `RY(π·fᵢ)` rotations, then applies a CNOT block on an alternating partition:

| Step | Partition |
|------|-----------|
| 0 | (q0,q1) (q2,q3) |
| 1 | (q1,q2) (q3,q0) |
| 2 | (q0,q1) (q2,q3) |
| 3 | (q1,q2) (q3,q0) |

Alternating partitions ensure the global update remains unitary and information propagates across the register (partitioned QCA formalism of Schumacher & Werner, 2004).

### Turing Machine

Tape alphabet `{0, 1, B, F, #, _}` over 8 regions of 98 bits each, with boundary markers `B`. States: `INIT → SCAN → REWIND → DONE`. The machine executes ~1,561 atomic steps per inference. The committed output features at each `F` (boundary flip) transition are pulled from the trained encoder so the TM's output trace exactly matches what flows into the QCA.

---

## Setup

```bash
git clone https://github.com/Akshith-Gandham/Data-Reuploading-MNIST-Classifier-Visualizer
cd Data-Reuploading-MNIST-Classifier-Visualizer

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

**Python 3.10+** required. Training runs on CPU; a GPU is not needed.

---

## Usage

### Train

```bash
python train.py
```

Downloads MNIST automatically to `./data/`. Saves a checkpoint to `./weights/model.pt` including the trained model, fitted LDA models, tSNE embeddings, and raw macro snapshots. Expected wall time: **30–90 min on CPU** depending on hardware.

### Run the visualizer

```bash
python main.py
```

Requires `./weights/model.pt` (run training first).

**Controls:**

| Control | Action |
|---------|--------|
| Draw canvas | Paint digit pixels (3×3 brush) |
| Clear | Reset canvas and state |
| Predict | Run inference, precompute TM history + QCA gate snapshots |
| Play / Pause | Animate the active mode |
| TM / QCA buttons | Switch visualization mode |
| Speed slider | TM steps per animation frame (1–50) |
| Show Circuit | Open QCA gate diagram popup |
| Click gate box | Show gate detail popup (angle, unitary, layer) |
| Click TM tape | Open TM state diagram + encoder NN popup |

---

## File structure

```
main.py          — Tkinter VisualApp: UI, rendering, animation loop
model.py         — Encoder, QCALayer, FullModel, snapshot QNodes
tm.py            — GranularTM, TMStep, TMState, δ table
viz.py           — gate program, per-gate snapshot QNodes, color theme
train.py         — training loop, dataloaders, LDA/tSNE projection fitting
requirements.txt — pinned dependencies
weights/         — checkpoint saved here after training
data/            — MNIST downloaded here automatically
```

---

## Visualizer panels

| Panel | Contents |
|-------|----------|
| Top bar | Basis-state probabilities `|c_b|²` for all 16 states of the 4-qubit register |
| Mid left (QCA) | Gate timeline — 28 boxes, current gate = orange, completed = blue |
| Mid left (TM) | Tape cells with head triangle; current δ transition shown in title |
| Mid right | 8-dim encoder feature bar chart |
| Bottom left | LDA projection at current macro step (10-class background scatter + white star = current sample) |
| Bottom right | tSNE embedding at current macro step (nearest-neighbour lookup for current sample) |

---

## Dependencies

| Package | Role |
|---------|------|
| `torch >= 2.2` | Encoder + classifier training, autograd |
| `pennylane 0.42` + `pennylane-lightning 0.42` | QCA simulation, adjoint differentiation |
| `torchvision >= 0.17` | MNIST dataset loader |
| `scikit-learn >= 1.4` | LDA fitting, tSNE |
| `matplotlib >= 3.8` | All plot panels, circuit diagram |
| `numpy` | Statevector math |
| `tkinter` | GUI (Python stdlib) |

---

## Theory references

1. Schumacher, B., Werner, R. F. *Reversible quantum cellular automata.* arXiv:quant-ph/0405174 (2004).
2. Arrighi, P. *An overview of quantum cellular automata.* Natural Computing 18, 885–899 (2019).
3. Pérez-Salinas, A. et al. *Data re-uploading for a universal quantum classifier.* Quantum 4, 226 (2020).
4. Sipser, M. *Introduction to the Theory of Computation*, 3rd ed. Cengage (2012).
5. Bergholm, V. et al. *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* arXiv:1811.04968 (2018).

---

## Notes

- All quantum simulation runs on a classical CPU via PennyLane's `lightning.qubit` backend. No quantum hardware, no quantum advantage claims.
- The QCA acts as a fixed nonlinear kernel; representational power comes from the encoder learning useful RY angles.
- The TM does not participate in training or inference — it is a pedagogical companion showing how a formally-defined automaton consumes the same image data.
