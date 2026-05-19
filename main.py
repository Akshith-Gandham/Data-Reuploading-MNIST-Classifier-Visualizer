import numpy as np
import torch
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from typing import List, Optional

from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch

from model import FullModel, nQubits, nSteps, nFeatures, load_checkpoint
from tm import (
    GranularTM, TMStep, TMState,
    SYM_CHAR, SYM_0, SYM_1, SYM_B, SYM_F, SYM_LEFT, SYM_BLANK
)
from viz import (
    gate_program, gate_snapshot_qnodes, n_total_gates, bloch_xyz,
    basis_state_probs, basis_state_labels, draw_circuit_figure,
    DARK_BG, DARK_PANEL, DARK_FG, DARK_FG_DIM,
    DARK_BTN, DARK_BTN_HOV, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN,
    GRID_COLOR, CLASS_COLORS
)


class VisualApp(tk.Tk):
    def __init__(self, checkpoint_path='./weights/qca_model.pt'):
        super().__init__()
        self.title("QCA MNIST Visualizer")
        self.configure(bg=DARK_BG)
        self.geometry('1700x950')

        self._model = FullModel()
        self._projections = load_checkpoint(self._model, checkpoint_path)
        self._model.eval()

        self._image = np.zeros((28, 28), dtype=np.float32)
        self._drawing = False
        self._mode = 'TM'
        self._playing = False
        self._tm_speed = 8

        self._tm_history = []
        self._tm_idx = 0
        self._qca_gates = []
        self._qca_idx = 0
        self._features = None
        self._prediction = -1
        self._confidence = 0.0

        self._build_ui()
        self._set_mode('TM')
        self._render()
        self.after(120, self._tick)

    def _build_ui(self):
        topbar = tk.Frame(self, bg=DARK_BG)
        topbar.pack(side='top', fill='x', padx=14, pady=(10, 4))
        tk.Label(topbar, text="QCA MNIST Visualizer", font=('Helvetica', 18, 'bold'),
                 bg=DARK_BG, fg=ACCENT_BLUE).pack(side='left')
        self._pred_label = tk.Label(topbar, text="Prediction: --",
                                   font=('Helvetica', 14, 'bold'), bg=DARK_BG, fg=DARK_FG)
        self._pred_label.pack(side='right', padx=20)

        main = tk.Frame(self, bg=DARK_BG)
        main.pack(side='top', fill='both', expand=True, padx=14, pady=4)

        left = tk.Frame(main, bg=DARK_BG, width=340)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Draw a digit", bg=DARK_BG, fg=DARK_FG_DIM,
                 font=('Helvetica', 11)).pack(side='top', anchor='w')
        canvas_frame = tk.Frame(left, bg=DARK_PANEL, padx=2, pady=2)
        canvas_frame.pack(side='top', pady=(2, 12))
        self._draw_canvas = tk.Canvas(canvas_frame, width=280, height=280,
                                     bg='black', highlightthickness=0)
        self._draw_canvas.pack()
        self._draw_canvas.bind('<Button-1>', self._on_canvas_press)
        self._draw_canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self._draw_canvas.bind('<ButtonRelease-1>', self._on_canvas_release)

        def make_btn(parent, text, cmd, **extra):
            return tk.Button(parent, text=text, command=cmd,
                             bg=DARK_BTN, fg=DARK_FG,
                             activebackground=DARK_BTN_HOV, activeforeground=DARK_FG,
                             relief='flat', borderwidth=0, font=('Helvetica', 11),
                             cursor='hand2', padx=10, pady=6, **extra)

        bf = tk.Frame(left, bg=DARK_BG)
        bf.pack(side='top', fill='x', pady=4)
        make_btn(bf, "Clear", self._on_clear).pack(side='left', expand=True, fill='x', padx=(0, 4))
        make_btn(bf, "Predict", self._on_predict).pack(side='left', expand=True, fill='x', padx=(4, 0))

        self._btn_play = make_btn(left, "Play / Pause", self._on_play_toggle)
        self._btn_play.pack(side='top', fill='x', pady=4)

        self._btn_circuit = make_btn(left, "Show Circuit", self._on_show_circuit)
        self._btn_circuit.pack(side='top', fill='x', pady=4)

        tk.Label(left, text="Mode", bg=DARK_BG, fg=DARK_FG_DIM,
                 font=('Helvetica', 11)).pack(side='top', anchor='w', pady=(10, 2))
        mb = tk.Frame(left, bg=DARK_BG)
        mb.pack(side='top', fill='x')
        self._btn_tm = make_btn(mb, "TM", lambda: self._set_mode('TM'))
        self._btn_tm.pack(side='left', expand=True, fill='x', padx=(0, 4))
        self._btn_qca = make_btn(mb, "QCA", lambda: self._set_mode('QCA'))
        self._btn_qca.pack(side='left', expand=True, fill='x', padx=(4, 0))

        tk.Label(left, text="TM speed (steps/frame)", bg=DARK_BG, fg=DARK_FG_DIM,
                 font=('Helvetica', 11)).pack(side='top', anchor='w', pady=(12, 2))
        self._speed_var = tk.IntVar(value=self._tm_speed)
        tk.Scale(left, from_=1, to=50, orient='horizontal',
                 bg=DARK_BG, fg=DARK_FG, troughcolor=DARK_BTN,
                 activebackground=ACCENT_BLUE, highlightthickness=0,
                 showvalue=True, variable=self._speed_var,
                 command=lambda v: setattr(self, '_tm_speed', int(v))
                 ).pack(side='top', fill='x')

        tk.Label(left, text="Status", bg=DARK_BG, fg=DARK_FG_DIM,
                 font=('Helvetica', 11)).pack(side='top', anchor='w', pady=(12, 2))
        self._state_text = tk.Text(left, height=10, bg=DARK_PANEL, fg=DARK_FG,
                                  font=('Courier', 10), relief='flat',
                                  wrap='word', borderwidth=0, padx=8, pady=6,
                                  insertbackground=DARK_FG)
        self._state_text.pack(side='top', fill='x')

        right = tk.Frame(main, bg=DARK_BG)
        right.pack(side='left', fill='both', expand=True)

        self._fig = Figure(figsize=(13, 8), facecolor=DARK_BG)
        self._fig_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._fig_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self._build_plots()
        self._fig.canvas.mpl_connect('button_press_event', self._on_fig_click)

    def _build_plots(self):
        gs = GridSpec(3, 4, figure=self._fig,
                      hspace=0.55, wspace=0.35,
                      left=0.06, right=0.97, top=0.93, bottom=0.07)
        self._ax_basis = self._fig.add_subplot(gs[0, :], facecolor=DARK_BG)
        self._ax_main = self._fig.add_subplot(gs[1, :3], facecolor=DARK_BG)
        self._ax_features = self._fig.add_subplot(gs[1, 3], facecolor=DARK_BG)
        self._ax_lda = self._fig.add_subplot(gs[2, :2], facecolor=DARK_BG)
        self._ax_tsne = self._fig.add_subplot(gs[2, 2:], facecolor=DARK_BG)
        for ax in [self._ax_basis, self._ax_main, self._ax_features,
                   self._ax_lda, self._ax_tsne]:
            self._style_2d_axis(ax)

    def _style_2d_axis(self, ax):
        for s in ax.spines.values():
            s.set_color(GRID_COLOR)
        ax.tick_params(colors=DARK_FG_DIM, labelsize=8)

    def _on_canvas_press(self, event):
        self._drawing = True
        self._draw_at(event.x, event.y)

    def _on_canvas_drag(self, event):
        if self._drawing:
            self._draw_at(event.x, event.y)

    def _on_canvas_release(self, event):
        self._drawing = False

    def _draw_at(self, px, py):
        cx, cy = px // 10, py // 10
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 28 and 0 <= ny < 28 and self._image[ny, nx] < 0.99:
                    self._image[ny, nx] = 1.0
                    x0, y0 = nx * 10, ny * 10
                    self._draw_canvas.create_rectangle(
                        x0, y0, x0 + 10, y0 + 10,
                        fill='white', outline='', tags='pixel'
                    )

    def _on_clear(self):
        self._image[:] = 0.0
        self._draw_canvas.delete('pixel')
        self._tm_history = []
        self._tm_idx = 0
        self._qca_gates = []
        self._qca_idx = 0
        self._features = None
        self._prediction = -1
        self._playing = False
        self._pred_label.config(text="Prediction: --", fg=DARK_FG)
        self._render()

    def _on_predict(self):
        img = self._image.copy()
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        t = (t - 0.1307) / 0.3081
        with torch.no_grad():
            features = self._model.encoder(t).squeeze(0).cpu().numpy()
            qca_out = self._model.qca(self._model.encoder(t))
            logits = self._model.classifier(qca_out).squeeze(0)

        pred = int(logits.argmax().item())
        conf = float(torch.softmax(logits, dim=0)[pred])

        feat64 = features.astype(np.float64)
        weights64 = self._model.qca.weights.detach().cpu().numpy().astype(np.float64)
        gates = [q(feat64, weights64) for q in gate_snapshot_qnodes]

        img_bits = (self._image > 0.5).astype(int).flatten()
        history = GranularTM(img_bits, features).run_to_done()

        self._features = features
        self._qca_gates = gates
        self._qca_idx = 0
        self._tm_history = history
        self._tm_idx = 0
        self._prediction = pred
        self._confidence = conf

        self._pred_label.config(
            text=f"Prediction: {pred}  ({conf*100:.1f}%)", fg=ACCENT_GREEN
        )
        self._playing = True
        self._render()

    def _on_play_toggle(self):
        self._playing = not self._playing

    def _set_mode(self, mode):
        self._mode = mode
        self._btn_tm.config(
            bg=ACCENT_BLUE if mode == 'TM' else DARK_BTN,
            fg='black' if mode == 'TM' else DARK_FG
        )
        self._btn_qca.config(
            bg=ACCENT_BLUE if mode == 'QCA' else DARK_BTN,
            fg='black' if mode == 'QCA' else DARK_FG
        )
        if mode == 'TM' and self._tm_history:
            self._tm_idx = 0
            self._playing = True
        elif mode == 'QCA' and self._qca_gates:
            self._qca_idx = 0
            self._playing = True
        self._render()

    def _tick(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._playing:
            advanced = False
            if self._mode == 'TM' and self._tm_history:
                if self._tm_idx < len(self._tm_history) - 1:
                    self._tm_idx = min(self._tm_idx + self._tm_speed, len(self._tm_history) - 1)
                    advanced = True
                else:
                    self._playing = False
            elif self._mode == 'QCA' and self._qca_gates:
                if self._qca_idx < len(self._qca_gates) - 1:
                    self._qca_idx += 1
                    advanced = True
                else:
                    self._playing = False
            if advanced or not self._playing:
                self._render()
        try:
            self.after(180, self._tick)
        except tk.TclError:
            pass

    def _render(self):
        if self._mode == 'TM':
            self._render_tm()
        else:
            self._render_qca()
        self._render_state_text()
        self._fig_canvas.draw_idle()

    def _draw_basis_probs(self, sv, title="Basis state probabilities"):
        ax = self._ax_basis
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color=DARK_FG, fontsize=11, loc='left')
        if sv is not None:
            probs = basis_state_probs(sv)
            labels = basis_state_labels(nQubits)
            colors = [ACCENT_BLUE] * len(probs)
            max_idx = int(np.argmax(probs))
            colors[max_idx] = ACCENT_ORANGE
            ax.bar(range(len(probs)), probs, color=colors, edgecolor='none')
            ax.set_xticks(range(len(probs)))
            ax.set_xticklabels(labels, fontsize=8, color=DARK_FG_DIM, rotation=0,
                               family='monospace')
            for i, p in enumerate(probs):
                if p > 0.05:
                    ax.text(i, p + 0.02, f"{p:.2f}",
                            ha='center', va='bottom', color=DARK_FG, fontsize=8)
        else:
            ax.text(0.5, 0.5, "Predict to see basis-state probabilities",
                    ha='center', va='center', color=DARK_FG_DIM, fontsize=11,
                    transform=ax.transAxes)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P", color=DARK_FG_DIM, fontsize=10)
        ax.tick_params(axis='y', colors=DARK_FG_DIM, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID_COLOR)

    def _render_qca(self):
        sv = self._qca_gates[self._qca_idx] if self._qca_gates else None
        if sv is not None:
            title = "Basis state probabilities -- gate " + str(self._qca_idx) + "/" + str(n_total_gates)
        else:
            title = "Basis state probabilities"
        self._draw_basis_probs(sv, title=title)

        ax = self._ax_main
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_title("QCA gate program (current = orange)", color=DARK_FG, fontsize=11, loc='left')
        for i, op in enumerate(gate_program):
            if op[0] == 'RY':
                lbl = "RY q" + str(op[1])
            else:
                lbl = "CX q" + str(op[1]) + "->q" + str(op[2])
            done = i < self._qca_idx
            current = i == self._qca_idx - 1
            if current:
                fc, ec, tcol, weight = DARK_PANEL, ACCENT_ORANGE, ACCENT_ORANGE, 'bold'
            elif done:
                fc, ec, tcol, weight = DARK_PANEL, ACCENT_BLUE, ACCENT_BLUE, 'normal'
            else:
                fc, ec, tcol, weight = DARK_BG, GRID_COLOR, DARK_FG_DIM, 'normal'
            ax.text(i, 0.5, lbl, color=tcol, fontsize=8, ha='center', va='center',
                    weight=weight, fontfamily='monospace',
                    bbox=dict(facecolor=fc, edgecolor=ec, linewidth=1.0, boxstyle='round,pad=0.3'))
        ax.set_xlim(-0.7, len(gate_program) - 0.3)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        self._draw_features(self._features, "Encoder features (8-d)")
        macro = self._gate_to_macro(self._qca_idx)
        self._draw_lda(macro, title="LDA -- macro step " + str(macro) + "/" + str(nSteps))
        self._draw_tsne(macro, title="tSNE -- macro step " + str(macro) + "/" + str(nSteps))

    def _render_tm(self):
        sv = self._qca_gates[-1] if self._qca_gates else None
        self._draw_basis_probs(sv, title="Basis state probabilities (final QCA state)")

        ax = self._ax_main
        ax.clear()
        ax.set_facecolor(DARK_BG)

        if self._tm_history:
            cur = self._tm_history[self._tm_idx]
            self._draw_tape(ax, cur.tape, cur.head, cur)
        else:
            ax.text(0.5, 0.5, "Draw + Predict to start TM",
                    transform=ax.transAxes, ha='center', va='center',
                    color=DARK_FG_DIM, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        if self._tm_history:
            cur = self._tm_history[self._tm_idx]
            vals = list(cur.output_features) + [0.0] * (8 - len(cur.output_features))
            self._draw_features(np.array(vals), "TM output features")
        else:
            self._draw_features(self._features, "TM output features")

        self._draw_lda(nSteps, title="LDA -- final state")
        self._draw_tsne(nSteps, title="tSNE -- final state")

    def _draw_tape(self, ax, tape, head, step):
        ax.clear()
        ax.set_facecolor(DARK_BG)
        for s in ax.spines.values():
            s.set_visible(False)

        n_visible = 31
        start = max(0, head - n_visible // 2)
        end = min(len(tape), start + n_visible)
        if end - start < n_visible:
            start = max(0, end - n_visible)

        cell_w = 1.0
        cell_h = 1.0
        base_y = 0.0

        sym_color = {
            SYM_0: (DARK_PANEL, GRID_COLOR, DARK_FG),
            SYM_1: (ACCENT_BLUE, ACCENT_BLUE, 'black'),
            SYM_B: (ACCENT_ORANGE, ACCENT_ORANGE, 'black'),
            SYM_F: (ACCENT_GREEN, ACCENT_GREEN, 'black'),
            SYM_LEFT: ('#a855f7', '#a855f7', 'white'),
            SYM_BLANK: ('#1c2128', GRID_COLOR, DARK_FG_DIM),
        }

        for i in range(start, end):
            local = i - start
            x = local * cell_w
            sym = tape[i]
            is_head = (i == head)

            if is_head:
                fc = '#fbbf24'
                ec = '#fbbf24'
                txt_col = 'black'
            else:
                fc, ec, txt_col = sym_color.get(sym, (DARK_PANEL, GRID_COLOR, DARK_FG))

            rect = Rectangle((x, base_y), cell_w, cell_h,
                             facecolor=fc, edgecolor=ec, linewidth=1.2)
            ax.add_patch(rect)
            ax.text(x + cell_w / 2, base_y + cell_h / 2, SYM_CHAR.get(sym, '?'),
                    ha='center', va='center', color=txt_col,
                    fontsize=13, weight='bold', family='monospace')
            ax.text(x + cell_w / 2, base_y - 0.25, str(i),
                    ha='center', va='top', color=DARK_FG_DIM,
                    fontsize=7, family='monospace')

        head_local = head - start
        if 0 <= head_local < (end - start):
            hx = head_local * cell_w + cell_w / 2
            tip = (hx, base_y + cell_h + 0.05)
            base1 = (hx - 0.42, base_y + cell_h + 0.85)
            base2 = (hx + 0.42, base_y + cell_h + 0.85)
            triangle = Polygon([tip, base1, base2], closed=True,
                               facecolor='#fbbf24', edgecolor='#fbbf24')
            ax.add_patch(triangle)
            ax.text(hx, base_y + cell_h + 1.15, 'HEAD',
                    ha='center', va='bottom',
                    color='#fbbf24', fontsize=9, weight='bold')

        ax.text(0, base_y - 0.85,
                "showing cells " + str(start) + "-" + str(end - 1) + " of " + str(len(tape))
                + "    (B = boundary, F = visited, # = left-end, _ = blank)",
                ha='left', va='top', color=DARK_FG_DIM, fontsize=8)

        dir_str = {-1: 'L', 0: 'S', +1: 'R'}[step.direction]
        title = ("delta(" + step.state.label() + ", " + SYM_CHAR[step.read_sym] + ")"
                 + " = (" + step.next_state.label() + ", " + SYM_CHAR[step.write_sym]
                 + ", " + dir_str + ")     -- transition: " + step.transition_label())
        ax.set_title(title, color=DARK_FG, fontsize=11, loc='left',
                     family='monospace')

        ax.set_xlim(-0.6, (end - start) * cell_w + 0.4)
        ax.set_ylim(-1.4, base_y + cell_h + 1.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    def _draw_features(self, vals, title):
        ax = self._ax_features
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color=DARK_FG, fontsize=10, loc='left')
        if vals is not None and len(vals) > 0:
            colors = [ACCENT_ORANGE if v > 0 else ACCENT_BLUE for v in vals]
            ax.bar(range(len(vals)), vals, color=colors, edgecolor='none')
        ax.set_ylim(-1.1, 1.1)
        ax.set_xticks(range(8))
        ax.tick_params(colors=DARK_FG_DIM, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID_COLOR)
        ax.axhline(0, color=GRID_COLOR, linewidth=0.5)

    def _draw_lda(self, macro, title):
        ax = self._ax_lda
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color=DARK_FG, fontsize=11, loc='left')
        try:
            bg = self._projections['ldaTestProjections'][macro]
            labs = self._projections['tsneLabels']
            for cls in range(10):
                mask = labs == cls
                if mask.any():
                    ax.scatter(bg[mask, 0], bg[mask, 1], s=14, alpha=0.6,
                               c=CLASS_COLORS[cls], label=str(cls), edgecolors='none')
            if self._qca_gates:
                macro_sv = self._snapshot_at_macro(macro)
                sv_flat = np.concatenate([macro_sv.real, macro_sv.imag]).reshape(1, -1)
                pt = self._projections['ldaModels'][macro].transform(sv_flat)[0]
                ax.scatter([pt[0]], [pt[1]], s=350, marker='*',
                           c='white', edgecolors=ACCENT_ORANGE, linewidth=2, zorder=10)
            ax.legend(loc='upper right', fontsize=7, ncol=5, frameon=False, labelcolor=DARK_FG)
        except Exception as e:
            ax.text(0.5, 0.5, "LDA unavailable: " + str(e),
                    transform=ax.transAxes, ha='center', va='center',
                    color=DARK_FG_DIM, fontsize=10)
        for s in ax.spines.values():
            s.set_color(GRID_COLOR)
        ax.tick_params(colors=DARK_FG_DIM, labelsize=8)

    def _draw_tsne(self, macro, title):
        ax = self._ax_tsne
        ax.clear()
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color=DARK_FG, fontsize=11, loc='left')
        try:
            bg = self._projections['tsneEmbeddings'][macro]
            labs = self._projections['tsneLabels']
            for cls in range(10):
                mask = labs == cls
                if mask.any():
                    ax.scatter(bg[mask, 0], bg[mask, 1], s=14, alpha=0.6,
                               c=CLASS_COLORS[cls], label=str(cls), edgecolors='none')
            snaps = self._projections.get('macroSnapshotsRaw')
            if self._qca_gates and snaps is not None and macro < len(snaps):
                macro_sv = self._snapshot_at_macro(macro)
                sv_flat = np.concatenate([macro_sv.real, macro_sv.imag])
                dists = np.linalg.norm(snaps[macro] - sv_flat, axis=1)
                nn = int(np.argmin(dists))
                pt = bg[nn]
                ax.scatter([pt[0]], [pt[1]], s=350, marker='*',
                           c='white', edgecolors=ACCENT_ORANGE, linewidth=2, zorder=10)
            ax.legend(loc='upper right', fontsize=7, ncol=5, frameon=False, labelcolor=DARK_FG)
        except Exception as e:
            ax.text(0.5, 0.5, "tSNE unavailable: " + str(e),
                    transform=ax.transAxes, ha='center', va='center',
                    color=DARK_FG_DIM, fontsize=10)
        for s in ax.spines.values():
            s.set_color(GRID_COLOR)
        ax.tick_params(colors=DARK_FG_DIM, labelsize=8)

    def _on_show_circuit(self):
        if self._features is not None:
            feats = self._features.astype(np.float64)
        else:
            feats = np.zeros(8, dtype=np.float64)
        weights = self._model.qca.weights.detach().cpu().numpy().astype(np.float64)
        win = tk.Toplevel(self)
        win.title("QCA quantum circuit")
        win.configure(bg=DARK_BG)
        win.geometry('1300x520')
        try:
            fig = draw_circuit_figure(feats, weights)
        except Exception as e:
            tk.Label(win, text="Circuit render failed: " + str(e),
                     bg=DARK_BG, fg=DARK_FG, font=('Helvetica', 12)).pack()
            return
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _on_fig_click(self, event):
        if event.inaxes is not self._ax_main:
            return
        if event.xdata is None:
            return
        if self._mode == 'QCA' and self._qca_gates:
            idx = int(round(event.xdata))
            if 0 <= idx < n_total_gates:
                self._show_gate_info(idx)
        elif self._mode == 'TM' and self._tm_history:
            self._show_dfa()

    def _show_gate_info(self, idx):
        op = gate_program[idx]
        if op[0] == 'RY':
            from viz import gate_layer
            wire = op[1]
            feat_idx = wire % nFeatures
            layer = gate_layer[idx]
            wval = float(self._model.qca.weights[layer, wire].detach().cpu().item())
            if self._features is not None:
                fval = float(self._features[feat_idx])
                angle = np.pi * fval + wval
                deg = np.degrees(angle)
                param_lines = [
                    "angle = pi * features[" + str(feat_idx) + "] + t[layer=" + str(layer) + ", q" + str(wire) + "]",
                    "features[" + str(feat_idx) + "] = " + f"{fval:+.4f}",
                    "t[" + str(layer) + "," + str(wire) + "] (trainable) = " + f"{wval:+.4f}",
                    "angle = " + f"{angle:+.4f}" + " rad  =  " + f"{deg:+.2f}" + " deg",
                ]
            else:
                param_lines = [
                    "angle = pi * features[" + str(feat_idx) + "] + t[layer=" + str(layer) + ", q" + str(wire) + "]",
                    "t[" + str(layer) + "," + str(wire) + "] (trainable) = " + f"{wval:+.4f}",
                    "(predict to see feature value)"
                ]
            unitary = ("U = [[ cos(t/2), -sin(t/2) ],\n"
                       "     [ sin(t/2),  cos(t/2) ]]")
            gate_name = "RY (Pauli-Y rotation, trainable bias)"
            wire_str = "target wire: q" + str(wire) + "    layer: " + str(layer)
        else:
            c, t = op[1], op[2]
            param_lines = ["no parameters (fixed unitary)"]
            unitary = ("U = [[1,0,0,0],\n"
                       "     [0,1,0,0],\n"
                       "     [0,0,0,1],\n"
                       "     [0,0,1,0]]")
            gate_name = "CNOT (controlled-X)"
            wire_str = "control: q" + str(c) + "    target: q" + str(t)

        macro = self._gate_to_macro(idx + 1)

        win = tk.Toplevel(self)
        win.title("Gate " + str(idx + 1) + " info")
        win.configure(bg=DARK_BG)
        win.geometry('540x420')

        tk.Label(win, text="Gate #" + str(idx + 1) + " of " + str(n_total_gates),
                 font=('Helvetica', 14, 'bold'),
                 bg=DARK_BG, fg=ACCENT_ORANGE).pack(pady=(14, 4), anchor='w', padx=20)
        tk.Label(win, text=gate_name, font=('Helvetica', 12),
                 bg=DARK_BG, fg=ACCENT_BLUE).pack(anchor='w', padx=20)
        tk.Label(win, text=wire_str, font=('Courier', 11),
                 bg=DARK_BG, fg=DARK_FG).pack(anchor='w', padx=20, pady=(8, 4))

        for line in param_lines:
            tk.Label(win, text=line, font=('Courier', 11),
                     bg=DARK_BG, fg=DARK_FG, justify='left').pack(anchor='w', padx=20)

        tk.Label(win, text="Unitary:", font=('Helvetica', 11, 'bold'),
                 bg=DARK_BG, fg=DARK_FG_DIM).pack(anchor='w', padx=20, pady=(12, 2))
        tk.Label(win, text=unitary, font=('Courier', 10),
                 bg=DARK_PANEL, fg=DARK_FG, justify='left',
                 padx=10, pady=8).pack(anchor='w', padx=20, fill='x')

        tk.Label(win, text="belongs to macro step: " + str(macro) + " / " + str(nSteps),
                 font=('Helvetica', 10),
                 bg=DARK_BG, fg=DARK_FG_DIM).pack(anchor='w', padx=20, pady=(12, 8))

    def _show_dfa(self):
        cur = self._tm_history[self._tm_idx].state if self._tm_history else None
        win = tk.Toplevel(self)
        win.title("TM state diagram + Encoder NN")
        win.configure(bg=DARK_BG)
        win.geometry('1500x950')

        fig = Figure(figsize=(15, 9.5), facecolor=DARK_BG)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 4], hspace=0.25,
                      left=0.04, right=0.97, top=0.95, bottom=0.04)
        ax_dfa = fig.add_subplot(gs[0], facecolor=DARK_BG)
        ax_enc = fig.add_subplot(gs[1], facecolor=DARK_BG)
        self._draw_dfa(ax_dfa, cur)
        self._draw_encoder(ax_enc)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _draw_encoder(self, ax):
        ax.clear()
        ax.set_facecolor(DARK_BG)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(
            "Encoder feed-forward NN:  784 (image) -> 64 (ReLU) -> 8 (tanh) -> QCA",
            color=DARK_FG, fontsize=12, loc='left'
        )

        img_t = torch.from_numpy(self._image).float().unsqueeze(0).unsqueeze(0)
        img_t = (img_t - 0.1307) / 0.3081
        flat = img_t.view(1, -1)
        with torch.no_grad():
            h_pre = self._model.encoder.fc1(flat)
            h = torch.relu(h_pre).squeeze(0).cpu().numpy()
            o = torch.tanh(self._model.encoder.fc2(torch.relu(h_pre))).squeeze(0).cpu().numpy()

        img_x, img_y = 0.4, 0.5
        img_w, img_h = 2.6, 2.6
        ax.imshow(self._image, cmap='gray', vmin=0, vmax=1,
                  extent=[img_x, img_x + img_w, img_y, img_y + img_h],
                  origin='upper', aspect='auto', zorder=2)
        rect = Rectangle((img_x, img_y), img_w, img_h,
                         fill=False, edgecolor=ACCENT_BLUE, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(img_x + img_w / 2, img_y - 0.2,
                "input  (28x28 = 784)",
                ha='center', va='top', color=DARK_FG, fontsize=10)

        h_x0 = 5.0
        h_y0 = 0.5
        cell_w = 0.30
        cell_h = 0.62
        h_max = float(np.max(np.abs(h))) + 1e-6
        for idx in range(64):
            r = idx // 16
            c = idx % 16
            cx = h_x0 + c * cell_w
            cy = h_y0 + (3 - r) * cell_h
            v = h[idx]
            if v > 0:
                t_val = min(1.0, v / h_max)
            else:
                t_val = 0.0
            color = (0.13 + t_val * 0.84, 0.18 + t_val * 0.32, 0.18 + t_val * 0.10)
            r_patch = Rectangle((cx, cy), cell_w * 0.94, cell_h * 0.94,
                                facecolor=color, edgecolor=GRID_COLOR,
                                linewidth=0.3)
            ax.add_patch(r_patch)
        h_w = 16 * cell_w
        ax.text(h_x0 + h_w / 2, h_y0 - 0.2,
                "hidden  (64, ReLU)",
                ha='center', va='top', color=DARK_FG, fontsize=10)
        ax.text(h_x0 + h_w / 2, h_y0 + 4 * cell_h + 0.05,
                "max activation = " + f"{h_max:.2f}",
                ha='center', va='bottom', color=DARK_FG_DIM, fontsize=8)

        o_x0 = 11.5
        o_y_mid = 1.7
        bar_w = 0.5
        bar_max = 1.4
        o_w = 8 * (bar_w + 0.12)
        for idx in range(8):
            bx = o_x0 + idx * (bar_w + 0.12)
            v = float(o[idx])
            bh = abs(v) * bar_max
            if v >= 0:
                rect = Rectangle((bx, o_y_mid), bar_w, bh,
                                 facecolor=ACCENT_ORANGE, edgecolor='none')
            else:
                rect = Rectangle((bx, o_y_mid - bh), bar_w, bh,
                                 facecolor=ACCENT_BLUE, edgecolor='none')
            ax.add_patch(rect)
            if v >= 0:
                ty = o_y_mid + bh + 0.08
                va = 'bottom'
            else:
                ty = o_y_mid - bh - 0.08
                va = 'top'
            ax.text(bx + bar_w / 2, ty,
                    f"{v:+.2f}", ha='center', va=va,
                    color=DARK_FG, fontsize=8)
        ax.plot([o_x0 - 0.05, o_x0 + o_w], [o_y_mid, o_y_mid],
                color=GRID_COLOR, lw=0.6)
        ax.text(o_x0 + o_w / 2, 0.3,
                "output  (8, tanh) -> QCA features",
                ha='center', va='top', color=DARK_FG, fontsize=10)

        ax.annotate('', xy=(h_x0 - 0.1, img_y + img_h / 2),
                    xytext=(img_x + img_w + 0.1, img_y + img_h / 2),
                    arrowprops=dict(arrowstyle='-|>', color=ACCENT_BLUE,
                                    lw=2.0, mutation_scale=18))
        ax.text((img_x + img_w + h_x0) / 2, img_y + img_h / 2 + 0.4,
                "Linear(784, 64)\n+ ReLU",
                ha='center', va='bottom', color=DARK_FG_DIM,
                fontsize=10, style='italic')

        h_end = h_x0 + h_w
        ax.annotate('', xy=(o_x0 - 0.1, o_y_mid),
                    xytext=(h_end + 0.1, h_y0 + 2 * cell_h),
                    arrowprops=dict(arrowstyle='-|>', color=ACCENT_ORANGE,
                                    lw=2.0, mutation_scale=18))
        ax.text((h_end + o_x0) / 2, h_y0 + 3 * cell_h + 0.1,
                "Linear(64, 8)\n+ tanh",
                ha='center', va='bottom', color=DARK_FG_DIM,
                fontsize=10, style='italic')

        ax.annotate('', xy=(o_x0 + o_w + 1.3, o_y_mid),
                    xytext=(o_x0 + o_w + 0.1, o_y_mid),
                    arrowprops=dict(arrowstyle='-|>', color=ACCENT_GREEN,
                                    lw=2.0, mutation_scale=18))
        ax.text(o_x0 + o_w + 0.7, o_y_mid + 0.2,
                "to QCA",
                ha='center', va='bottom', color=ACCENT_GREEN,
                fontsize=10, style='italic')

        ax.set_xlim(0, 18.5)
        ax.set_ylim(-0.1, 4.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')

    def _draw_dfa(self, ax, current):
        PURPLE = '#a855f7'
        NODE_R = 0.7

        positions = {
            TMState.INIT:   (0.0,  0.0),
            TMState.SCAN:   (5.0,  0.0),
            TMState.REWIND: (10.0, 0.0),
            TMState.DONE:   (15.0, 0.0),
        }

        scan_loops = ['0 -> 0,R', '1 -> 1,R', 'B -> F,R']
        rewind_loops = ['0 -> 0,L', '1 -> 1,L', 'F -> F,L']

        for st, labels in [(TMState.SCAN, scan_loops), (TMState.REWIND, rewind_loops)]:
            x, y = positions[st]
            loop = FancyArrowPatch(
                (x - 0.30, y + NODE_R + 0.05),
                (x + 0.30, y + NODE_R + 0.05),
                connectionstyle="arc3,rad=-1.6",
                arrowstyle="-|>", color=DARK_FG_DIM, lw=1.4, mutation_scale=14
            )
            ax.add_patch(loop)
            for i, lbl in enumerate(labels):
                if 'R' in lbl:
                    color = ACCENT_BLUE
                else:
                    color = ACCENT_ORANGE
                ax.text(x, y + NODE_R + 1.4 + i * 0.45, lbl,
                        ha='center', va='bottom',
                        color=color,
                        fontsize=11, weight='bold', family='monospace')

        def fwd(s1, s2, label, color, y_offset=-0.45):
            x1, y1 = positions[s1]
            x2, y2 = positions[s2]
            ax.annotate('', xy=(x2 - NODE_R, y2), xytext=(x1 + NODE_R, y1),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=1.7, mutation_scale=18))
            ax.text((x1 + x2) / 2, y1 + y_offset, label,
                    ha='center', va='top', color=color,
                    fontsize=11, weight='bold', family='monospace')

        fwd(TMState.INIT, TMState.SCAN, '# -> #,R', ACCENT_BLUE)
        fwd(TMState.SCAN, TMState.REWIND, '_ -> _,L', ACCENT_ORANGE)
        fwd(TMState.REWIND, TMState.DONE, '# -> #,S', ACCENT_GREEN)

        for st, (x, y) in positions.items():
            is_current = (st == current)
            if is_current:
                fc, ec, tc, lw = PURPLE, 'white', 'white', 3.0
            elif st == TMState.DONE:
                fc, ec, tc, lw = DARK_PANEL, ACCENT_GREEN, ACCENT_GREEN, 2.0
            elif st == TMState.INIT:
                fc, ec, tc, lw = DARK_PANEL, ACCENT_BLUE, ACCENT_BLUE, 2.0
            else:
                fc, ec, tc, lw = DARK_PANEL, ACCENT_BLUE, DARK_FG, 1.6

            circle = Circle((x, y), NODE_R, facecolor=fc, edgecolor=ec,
                            linewidth=lw, zorder=10)
            ax.add_patch(circle)

            ax.text(x, y, st.label(), ha='center', va='center',
                    color=tc, fontsize=12, weight='bold', zorder=11,
                    family='monospace')

            if st == TMState.DONE:
                outer = Circle((x, y), NODE_R + 0.10, fill=False,
                               edgecolor=ACCENT_GREEN, linewidth=lw, zorder=10)
                ax.add_patch(outer)

            if st == TMState.INIT:
                ax.annotate('', xy=(x - NODE_R - 0.05, y),
                            xytext=(x - NODE_R - 1.0, y),
                            arrowprops=dict(arrowstyle='-|>', color=DARK_FG,
                                            lw=1.5, mutation_scale=16))
                ax.text(x - NODE_R - 1.05, y + 0.30, 'start',
                        ha='right', va='bottom', color=DARK_FG_DIM,
                        fontsize=10, style='italic')

        legend = ("Delta table notation:  read -> write,move    |    "
                  "Tape Alphabet = {0, 1, B, F, #, _}    |    "
                  "B -> F transition commits an encoder feature")
        ax.text(0.0, -2.6, legend, color=DARK_FG_DIM, fontsize=10)

        if current:
            cur_label = current.label()
        else:
            cur_label = '--'
        ax.set_title(
            "TM finite-state diagram   |   current state: " + cur_label,
            color=DARK_FG, fontsize=13, loc='left'
        )

        ax.set_xlim(-2.0, 17.0)
        ax.set_ylim(-3.0, 4.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect('equal')

    def _render_state_text(self):
        self._state_text.delete('1.0', 'end')
        if self._mode == 'TM':
            if self._tm_history:
                cur = self._tm_history[self._tm_idx]
                move_str = {-1: 'L', 0: 'S', 1: 'R'}.get(cur.direction, '?')
                txt = (
                    "TM step " + str(self._tm_idx + 1) + "/" + str(len(self._tm_history)) + "\n"
                    + "state   : " + cur.state.label() + "\n"
                    + "read    : " + SYM_CHAR[cur.read_sym] + "\n"
                    + "write   : " + SYM_CHAR[cur.write_sym] + "\n"
                    + "move    : " + move_str + "\n"
                    + "trans   : " + cur.transition_label() + "\n"
                    + "head    : " + str(cur.head) + "\n"
                    + "feature : " + str(cur.feature_idx) + "/8\n"
                    + "accum   : " + f"{cur.accumulator:.2f}" + "\n"
                    + "speed   : " + str(self._tm_speed) + " steps/frame"
                )
            else:
                txt = "TM idle.\nDraw + Predict to start."
        else:
            if self._qca_gates:
                if self._qca_idx == 0:
                    op_str = "INIT (no gate)"
                else:
                    g = gate_program[self._qca_idx - 1]
                    if g[0] == 'RY':
                        op_str = "RY q" + str(g[1])
                    else:
                        op_str = "CNOT q" + str(g[1]) + "->q" + str(g[2])
                macro = self._gate_to_macro(self._qca_idx)
                txt = (
                    "QCA gate " + str(self._qca_idx) + "/" + str(n_total_gates) + "\n"
                    + "op      : " + op_str + "\n"
                    + "macro   : " + str(macro) + "/" + str(nSteps) + "\n"
                    + "qubits  : " + str(nQubits) + "\n"
                    + "playing : " + str(self._playing)
                )
            else:
                txt = "QCA idle.\nDraw + Predict to start."
        self._state_text.insert('1.0', txt)

    def _gate_to_macro(self, gate_idx):
        init = nQubits
        per = nQubits + (nQubits // 2)
        if gate_idx <= init:
            return 0
        return min((gate_idx - init) // per, nSteps)

    def _snapshot_at_macro(self, macro):
        gate_idx = nQubits + macro * (nQubits + nQubits // 2)
        gate_idx = min(gate_idx, len(self._qca_gates) - 1)
        return self._qca_gates[gate_idx]


if __name__ == '__main__':
    app = VisualApp("./weights/model.pt")
    app.mainloop()
