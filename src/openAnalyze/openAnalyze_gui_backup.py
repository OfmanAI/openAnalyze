#!/usr/bin/env python3
"""
openAnalyze_gui – desktop front-end for the openAnalyze detector pipeline
-----------------------------------------------------------------------
Features:
• Sample‐size spinbox
• “Normalize classes” checkbox
• Quick Confusion‐Matrix mode (computes best‐F1 threshold per sample)
• Full pipeline (R analytics + all plots)
• “View Graph” dropdown to display any generated plot
• Saves quick confusion output under “quick_cms” folder
• Aesthetic improvements: separators and bold labels
• Buttons respond anywhere inside their area

Run after installation:
    openAnalyze-gui
"""

from __future__ import annotations
import threading, time, datetime, sys
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import ttkbootstrap as ttkb                   # pip install ttkbootstrap
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from rich.console import Console
from PIL import Image, ImageTk                # pip install pillow
import plotly.graph_objects as go             # pip install plotly
import kaleido                                # ensures fig.to_image works

import tkinter as tk
from tkinter import filedialog

# Import existing helpers
from openAnalyze.openAnalyze import ingest, analyse, summarise_results  # type: ignore

APP_TITLE = "openAnalyze-gui"
console   = Console()

class DetectorApp(ttkb.Window):
    def __init__(self):
        super().__init__(title=APP_TITLE, themename="flatly", size=(1200, 750))
        self.style.configure("TButton")
        self._build_left_panel()
        self._build_right_panel()
        ttkb.Separator(self, bootstyle="secondary").pack(fill='x', pady=(5,0))
        self.status = ttkb.Label(self, text="Idle", bootstyle=SECONDARY)
        self.status.pack(side=BOTTOM, fill='x', pady=(0,10))
        self._img_refs = []  # keep image references alive
        # tracking all available graphs
        self.available_graphs: dict[str, Path] = {}

    # ── Left-hand controls ─────────────────────────────────────────────
    def _build_left_panel(self):
        frm = ttkb.Frame(self, padding=10, bootstyle="light")
        frm.pack(side=LEFT, fill='y')

        # CSV paths
        self.fake_var = ttkb.StringVar()
        self.real_var = ttkb.StringVar()
        self.det_var  = ttkb.StringVar(value="my_detector")
        self.out_var  = ttkb.StringVar(value=str(Path.cwd()))

        # Sample-size & normalize
        self.sample_var    = ttkb.IntVar(value=200)
        self.normalize_var = ttkb.BooleanVar(value=True)

        lbl_font = ("Segoe UI", 10, "bold")
        entry_font = ("Segoe UI", 10)

        # Row 0: Fake CSV
        ttkb.Label(frm, text="Fake CSV", font=lbl_font).grid(row=0, column=0, sticky='w', pady=4)
        ttkb.Entry(frm, textvariable=self.fake_var, width=32, font=entry_font).grid(row=0, column=1, pady=4)
        ttkb.Button(frm, text="📂", command=self.browse_fake, width=3, bootstyle=SECONDARY)\
            .grid(row=0, column=2, padx=4)

        # Row 1: Real CSV
        ttkb.Label(frm, text="Real CSV", font=lbl_font).grid(row=1, column=0, sticky='w', pady=4)
        ttkb.Entry(frm, textvariable=self.real_var, width=32, font=entry_font).grid(row=1, column=1, pady=4)
        ttkb.Button(frm, text="📂", command=self.browse_real, width=3, bootstyle=SECONDARY)\
            .grid(row=1, column=2, padx=4)

        # Row 2: Detector name
        ttkb.Label(frm, text="Detector name", font=lbl_font).grid(row=2, column=0, sticky='w', pady=(15,4))
        ttkb.Entry(frm, textvariable=self.det_var, width=20, font=entry_font).grid(row=2, column=1, sticky='w', pady=(15,4))

        # Row 3: Output folder
        ttkb.Label(frm, text="Output folder", font=lbl_font).grid(row=3, column=0, sticky='w', pady=4)
        ttkb.Entry(frm, textvariable=self.out_var, width=32, font=entry_font).grid(row=3, column=1, pady=4)
        ttkb.Button(frm, text="📁", command=self.browse_out, width=3, bootstyle=SECONDARY)\
            .grid(row=3, column=2, padx=4)

        # Row 4: Sample size
        ttkb.Label(frm, text="Sample size", font=lbl_font).grid(row=4, column=0, sticky='w', pady=(15,4))
        ttkb.Spinbox(frm,
                     from_=10, to=10000,
                     increment=10,
                     textvariable=self.sample_var,
                     width=10,
                     bootstyle="info"
        ).grid(row=4, column=1, sticky='w', pady=(15,4))

        # Row 5: Normalize classes
        ttkb.Checkbutton(
            frm,
            text="Normalize classes",
            variable=self.normalize_var,
            bootstyle="success"
        ).grid(row=5, column=1, sticky='w', pady=4)

        # Row 6: Buttons (full width clickable)
        btn_q = ttkb.Button(
            frm, text="▶ Quick Confusion", bootstyle=WARNING, width=30,
            command=self.start_quick_confusion
        )
        btn_q.grid(row=6, column=0, columnspan=3, pady=(15,6), sticky='ew')

        btn_f = ttkb.Button(
            frm, text="▶ Full pipeline", bootstyle=SUCCESS, width=30,
            command=self.start_full_pipeline
        )
        btn_f.grid(row=7, column=0, columnspan=3, pady=6, sticky='ew')

        # Row 8: Log pane
        ttkb.Label(frm, text="Log", font=lbl_font).grid(row=8, column=0, sticky='w', pady=(15,4))
        self.log = ttkb.ScrolledText(frm, height=10, width=50, font=("Consolas", 9))
        self.log.grid(row=9, column=0, columnspan=3, sticky='ew')

        # Make columns expand
        frm.columnconfigure(1, weight=1)

    # ── Right-hand panel (dropdown + image display) ──────────────────────
    def _build_right_panel(self):
        right = ttkb.Frame(self, padding=10)
        right.pack(side=LEFT, fill='both', expand=True)

        ttkb.Label(right, text="View Graph", font=("Segoe UI", 10, "bold")).pack(anchor='nw')
        self.graph_var = ttkb.StringVar()
        self.graph_cb = ttkb.Combobox(
            right, textvariable=self.graph_var, state="readonly", width=40,
            bootstyle="info"
        )
        self.graph_cb.pack(anchor='nw', pady=(2, 10))
        self.graph_cb.bind("<<ComboboxSelected>>", self.on_graph_selected)

        self.img_label = ttkb.Label(right)
        self.img_label.pack(fill='both', expand=True)

    def on_graph_selected(self, event=None):
        key = self.graph_var.get()
        path = self.available_graphs.get(key)
        if not path or not path.exists():
            return
        pil = Image.open(path)
        photo = ImageTk.PhotoImage(pil)
        self._img_refs = [photo]
        self.img_label.configure(image=photo)

    # ── File pickers via tkinter.filedialog ─────────────────────────────
    def browse_fake(self):
        path = filedialog.askopenfilename(
            title="Select Fake CSV",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(Path.cwd())
        )
        if path:
            self.fake_var.set(path)

    def browse_real(self):
        path = filedialog.askopenfilename(
            title="Select Real CSV",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(Path.cwd())
        )
        if path:
            self.real_var.set(path)

    def browse_out(self):
        path = filedialog.askdirectory(
            title="Choose Output Folder",
            initialdir=str(Path.cwd())
        )
        if path:
            self.out_var.set(path)

    # ── Threading helpers ───────────────────────────────────────────────────
    def start_quick_confusion(self):
        if not (self.fake_var.get() and self.real_var.get()):
            Messagebox.show_warning("Please select both Fake and Real CSVs.")
            return
        threading.Thread(target=self.run_quick_confusion, daemon=True).start()

    def start_full_pipeline(self):
        if not (self.fake_var.get() and self.real_var.get()):
            Messagebox.show_warning("Please select both Fake and Real CSVs.")
            return
        threading.Thread(target=self.run_full_pipeline, daemon=True).start()

    # ── Quick confusion only (reuse ingest → combined CSV) ──────────────────
    def run_quick_confusion(self):
        self.after(0, lambda: self.status.configure(text="Running Quick Confusion…", bootstyle=INFO))
        start_time = time.time()
        try:
            # Build combined CSV using ingest exactly
            out_root = Path(self.out_var.get())
            quick_csv_dir = out_root / "quick_csv"
            combined_csv = ingest(
                [self.fake_var.get(), self.real_var.get()],
                self.det_var.get(),
                quick_csv_dir
            )
            df = pd.read_csv(combined_csv)  # has 'score' and 'true_label'
            self.before_log(f"Loaded combined CSV with {len(df)} rows.")

            # Sample according to settings
            n = self.sample_var.get()
            if self.normalize_var.get():
                self.before_log(f"Sampling {n} from each class.")
                df_f = df[df.true_label == 1]
                df_r = df[df.true_label == 0]
                df_sampled = pd.concat([
                    df_f.sample(n=min(n, len(df_f)), replace=False),
                    df_r.sample(n=min(n, len(df_r)), replace=False)
                ], ignore_index=True)
            else:
                self.before_log(f"Sampling {n} records from combined pool.")
                df_sampled = df.sample(n=min(n, len(df)), replace=False)

            # Find best threshold by grid search on sampled data
            scores = df_sampled.score.values
            labels = df_sampled.true_label.values
            thresholds = np.arange(0.0, 1.0001, 0.01)
            best_thr = 0.0
            best_f1 = -1.0
            for t in thresholds:
                preds = (scores >= t).astype(int)
                tp = int(((preds == 1) & (labels == 1)).sum())
                fp = int(((preds == 1) & (labels == 0)).sum())
                fn = int(((preds == 0) & (labels == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                if f1 > best_f1 or (f1 == best_f1 and t < best_thr):
                    best_f1 = f1
                    best_thr = t

            self.before_log(f"Best‐F1 threshold = {best_thr:.2f} (F1={best_f1:.3f})")

            # Apply best_thr to sampled data
            df_sampled["pred"] = (df_sampled.score >= best_thr).astype(int)

            cm = pd.crosstab(
                df_sampled.true_label, df_sampled.pred,
                rownames=["Actual"], colnames=["Predicted"], dropna=False
            ).reindex(index=[0,1], columns=[0,1], fill_value=0)

            self.before_log("Computed confusion matrix at best threshold.")
            # Ensure quick_cms folder exists
            quick_cms_dir = out_root / "quick_cms"
            quick_cms_dir.mkdir(exist_ok=True)

            # Save and display
            save_path = quick_cms_dir / f"confusion_quick_thr{best_thr:.2f}.png"
            self._show_confusion_matrix(cm, best_thr, save_path)

            # Update dropdown
            key = f"Quick thr={best_thr:.2f}"
            self.available_graphs[key] = save_path
            self._refresh_dropdown()

            elapsed = time.time() - start_time
            self.before_log(f"Quick confusion done in {elapsed:.1f}s.")
            self.after(0, lambda: self.status.configure(text="Quick Confusion ✓", bootstyle=SUCCESS))

        except Exception as exc:
            err_msg = f"Error in Quick Confusion:\n{exc}"
            self.after(0, lambda: self.status.configure(text="Error ✗", bootstyle=DANGER))
            self.after(0, lambda: Messagebox.show_error(err_msg))

    # Refresh dropdown options from available_graphs
    def _refresh_dropdown(self):
        names = list(self.available_graphs.keys())
        self.graph_cb["values"] = names
        if names:
            self.graph_cb.current(len(names)-1)
            self.on_graph_selected()

    def before_log(self, text: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.after(0, lambda: self._append_log(f"[{ts}] {text}\n"))

    def _show_confusion_matrix(self, cm: pd.DataFrame, thr: float, save_path: Path):
        """Convert Plotly heatmap to PNG via Kaleido, save, and display with numbers."""
        fig = go.Figure(data=go.Heatmap(
            z=cm.values,
            x=[str(c) for c in cm.columns],
            y=[str(r) for r in cm.index],
            colorscale="Blues",
            text=cm.values,
            texttemplate="%{text}",
            textfont={"color": "black"},
            hovertemplate="Actual %{y}<br>Pred %{x}: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title=f"Confusion Matrix (Quick, thr={thr:.2f})",
            margin=dict(t=50,l=30,r=30,b=30),
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )

        # Save to quick_cms
        fig.write_image(str(save_path), format="png", engine="kaleido")

        # Display inside GUI
        png_bytes = fig.to_image(format="png", engine="kaleido")
        buf = BytesIO(png_bytes)
        pil = Image.open(buf)
        photo = ImageTk.PhotoImage(pil)

        def _add_to_display():
            self._img_refs = [photo]
            self.img_label.configure(image=photo)

        self.after(0, _add_to_display)

    # ── Full pipeline (ingest + R + display all plots) ─────────────────
    def run_full_pipeline(self):
        self.after(0, lambda: self.status.configure(text="Running Full Pipeline…", bootstyle=INFO))
        start_time = time.time()
        try:
            out_root  = Path(self.out_var.get())
            csv_ready = out_root / "csv_ready"
            results   = out_root / f"results_{self.det_var.get()}"

            self.before_log(f"Merging CSVs → {csv_ready}")
            csv_path = ingest([self.fake_var.get(), self.real_var.get()],
                              self.det_var.get(), csv_ready)

            self.before_log("Running R analytics … this may take a moment")
            analyse(csv_path, results)

            self.before_log("Analysis complete – loading plots")
            self._populate_plots(results)
            self._show_full_confusion(results)

            elapsed = time.time() - start_time
            self.before_log(f"Completed in {elapsed:.1f}s. Artefacts in {results}")
            self.after(0, lambda: self.status.configure(text="Full Pipeline ✓", bootstyle=SUCCESS))

        except Exception as exc:
            err_msg = f"Pipeline error:\n{exc}"
            self.after(0, lambda: self.status.configure(text="Error ✗", bootstyle=DANGER))
            self.after(0, lambda: Messagebox.show_error(err_msg))

    def _show_full_confusion(self, results: Path):
        cm_csv = next(results.glob("confusion_matrix_modalThr*.csv"), None)
        if cm_csv is None:
            return
        df_full = pd.read_csv(cm_csv, index_col=0).astype(int)
        # Derive threshold from filename
        stem = cm_csv.stem
        thr_str = stem.split("Thr")[-1]
        best_thr = float(thr_str)
        # Save path under results
        save_path = results / f"confusion_full_thr{best_thr:.2f}.png"
        self._show_confusion_matrix(df_full, best_thr, save_path)
        # Add to dropdown
        key = f"Full thr={best_thr:.2f}"
        self.available_graphs[key] = save_path
        self._refresh_dropdown()

    # Called when user selects from dropdown
    def on_graph_selected(self, event=None):
        key = self.graph_var.get()
        path = self.available_graphs.get(key)
        if not path or not path.exists():
            return
        pil = Image.open(path)
        photo = ImageTk.PhotoImage(pil)
        self._img_refs = [photo]
        self.img_label.configure(image=photo)

    # ── GUI threading safety ─────────────────────────────────────────────
    def _append_log(self, line: str):
        self.log.insert('end', line)
        self.log.see('end')

    # ── shared plot population (for full pipeline) ─────────────────────────
    def _populate_plots(self, results: Path):
        # After R finishes, collect all PNGs under results and add to dropdown
        for img in sorted(results.glob("*.png")):
            # Use stem for dropdown
            key = img.stem.replace("_", " ")
            self.available_graphs[key] = img
        self._refresh_dropdown()

# ─────────────────────────────────────────────────────────────
def launch():
    app = DetectorApp()
    app.place_window_center()
    app.mainloop()

if __name__ == "__main__":
    launch()
