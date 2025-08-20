#!/usr/bin/env python3
"""
openAnalyze_gui – Enhanced desktop front-end for the openAnalyze detector pipeline
---------------------------------------------------------------------------------

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

APP_TITLE = "openAnalyze - Quick Machine Learning Statistics"
console   = Console()

class DetectorApp(ttkb.Window):
    def __init__(self):
        super().__init__(title=APP_TITLE, themename="cosmo", size=(1400, 900))
        
        # Configure custom styles
        self._configure_styles()
        
        # Initialize state
        self._img_refs = []  # keep image references alive
        self.available_graphs: dict[str, Path] = {}
        self._is_processing = False
        
        # Build interface
        self._build_interface()
        self._setup_bindings()
        
        # Initial status
        self._update_status("Ready - Select your CSV files to begin analysis", SUCCESS)

    def _configure_styles(self):
        """Configure custom styles for enhanced aesthetics"""
        style = self.style
        
        # Custom button styles
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(20, 12)
        )
        
        style.configure(
            "Secondary.TButton", 
            font=("Segoe UI", 10),
            padding=(15, 8)
        )
        
        # Enhanced label styles
        style.configure(
            "Heading.TLabel",
            font=("Segoe UI", 12, "bold"),
            foreground="#2c3e50"
        )
        
        style.configure(
            "Subheading.TLabel",
            font=("Segoe UI", 10, "bold"),
            foreground="#34495e"
        )
        
        # Entry styles
        style.configure(
            "Custom.TEntry",
            font=("Segoe UI", 10),
            fieldbackground="#ffffff",
            borderwidth=2
        )

    def _build_interface(self):
        """Build the main interface with enhanced layout"""
        # Main container with padding
        main_container = ttkb.Frame(self, padding=20)
        main_container.pack(fill='both', expand=True)
        
        # Title section
        self._build_title_section(main_container)
        
        # Main content area
        content_frame = ttkb.Frame(main_container)
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Left panel (controls)
        self._build_enhanced_left_panel(content_frame)
        
        # Separator
        separator = ttkb.Separator(content_frame, orient='vertical', bootstyle="secondary")
        separator.pack(side=LEFT, fill='y', padx=20)
        
        # Right panel (visualization)
        self._build_enhanced_right_panel(content_frame)
        
        # Status bar
        self._build_status_bar(main_container)

    def _build_title_section(self, parent):
        """Build the title section with branding"""
        title_frame = ttkb.Frame(parent, bootstyle="light")
        title_frame.pack(fill='x', pady=(0, 10))
        
        # Main title
        title_label = ttkb.Label(
            title_frame, 
            text="openAnalyze", 
            font=("Segoe UI", 24, "bold"),
            bootstyle="primary"
        )
        title_label.pack(side=LEFT)
        
        # Subtitle
        subtitle_label = ttkb.Label(
            title_frame,
            text="Quick ML Statistics",
            font=("Segoe UI", 12),
            bootstyle="secondary"
        )
        subtitle_label.pack(side=LEFT, padx=(15, 0), pady=(8, 0))
        
        # Version info
        version_label = ttkb.Label(
            title_frame,
            text="v2.0",
            font=("Segoe UI", 10),
            bootstyle="info"
        )
        version_label.pack(side=RIGHT, pady=(8, 0))

    def _build_enhanced_left_panel(self, parent):
        """Build enhanced left control panel"""
        left_frame = ttkb.LabelFrame(
            parent, 
            text=" Configuration & Controls ",
            padding=25,
            bootstyle="primary"
        )
        left_frame.pack(side=LEFT, fill='y', padx=(0, 10))
        
        # Initialize variables
        self.fake_var = ttkb.StringVar()
        self.real_var = ttkb.StringVar()
        self.det_var = ttkb.StringVar(value="my_detector")
        self.out_var = ttkb.StringVar(value=str(Path.cwd()))
        self.sample_var = ttkb.IntVar(value=200)
        self.normalize_var = ttkb.BooleanVar(value=True)
        
        # File Selection Section
        self._build_file_section(left_frame)
        
        # Configuration Section
        self._build_config_section(left_frame)
        
        # Action Buttons Section
        self._build_action_section(left_frame)
        
        # Log Section
        self._build_log_section(left_frame)

    def _build_file_section(self, parent):
        """Build file selection section"""
        file_frame = ttkb.LabelFrame(parent, text=" Data Files ", padding=15, bootstyle="info")
        file_frame.pack(fill='x', pady=(0, 20))
        
        # Fake CSV
        fake_frame = ttkb.Frame(file_frame)
        fake_frame.pack(fill='x', pady=(0, 10))
        
        ttkb.Label(fake_frame, text="Fake Dataset CSV:", style="Subheading.TLabel").pack(anchor='w')
        fake_input_frame = ttkb.Frame(fake_frame)
        fake_input_frame.pack(fill='x', pady=(5, 0))
        
        self.fake_entry = ttkb.Entry(
            fake_input_frame, 
            textvariable=self.fake_var, 
            style="Custom.TEntry"
        )
        self.fake_entry.pack(side=LEFT, fill='x', expand=True)
        
        ttkb.Button(
            fake_input_frame, 
            text="Browse", 
            command=self.browse_fake,
            bootstyle="outline-secondary"
        ).pack(side=RIGHT, padx=(10, 0))
        
        # Real CSV
        real_frame = ttkb.Frame(file_frame)
        real_frame.pack(fill='x')
        
        ttkb.Label(real_frame, text="Real Dataset CSV:", style="Subheading.TLabel").pack(anchor='w')
        real_input_frame = ttkb.Frame(real_frame)
        real_input_frame.pack(fill='x', pady=(5, 0))
        
        self.real_entry = ttkb.Entry(
            real_input_frame, 
            textvariable=self.real_var, 
            style="Custom.TEntry"
        )
        self.real_entry.pack(side=LEFT, fill='x', expand=True)
        
        ttkb.Button(
            real_input_frame, 
            text="Browse", 
            command=self.browse_real,
            bootstyle="outline-secondary"
        ).pack(side=RIGHT, padx=(10, 0))

    def _build_config_section(self, parent):
        """Build configuration section"""
        config_frame = ttkb.LabelFrame(parent, text=" Analysis Configuration ", padding=15, bootstyle="success")
        config_frame.pack(fill='x', pady=(0, 20))
        
        # Detector name
        det_frame = ttkb.Frame(config_frame)
        det_frame.pack(fill='x', pady=(0, 15))
        
        ttkb.Label(det_frame, text="Detector Name:", style="Subheading.TLabel").pack(anchor='w')
        ttkb.Entry(
            det_frame, 
            textvariable=self.det_var, 
            style="Custom.TEntry"
        ).pack(fill='x', pady=(5, 0))
        
        # Output folder
        out_frame = ttkb.Frame(config_frame)
        out_frame.pack(fill='x', pady=(0, 15))
        
        ttkb.Label(out_frame, text="Output Directory:", style="Subheading.TLabel").pack(anchor='w')
        out_input_frame = ttkb.Frame(out_frame)
        out_input_frame.pack(fill='x', pady=(5, 0))
        
        ttkb.Entry(
            out_input_frame, 
            textvariable=self.out_var, 
            style="Custom.TEntry"
        ).pack(side=LEFT, fill='x', expand=True)
        
        ttkb.Button(
            out_input_frame, 
            text="Browse", 
            command=self.browse_out,
            bootstyle="outline-secondary"
        ).pack(side=RIGHT, padx=(10, 0))
        
        # Sample size and normalize in a row
        options_frame = ttkb.Frame(config_frame)
        options_frame.pack(fill='x')
        
        # Sample size
        sample_frame = ttkb.Frame(options_frame)
        sample_frame.pack(side=LEFT, fill='x', expand=True)
        
        ttkb.Label(sample_frame, text="Sample Size:", style="Subheading.TLabel").pack(anchor='w')
        self.sample_spinbox = ttkb.Spinbox(
            sample_frame,
            from_=2, to=10000,
            increment=1,
            textvariable=self.sample_var,
            bootstyle="info",
            width=12
        )
        self.sample_spinbox.pack(anchor='w', pady=(5, 0))
        
        # Normalize checkbox
        normalize_frame = ttkb.Frame(options_frame)
        normalize_frame.pack(side=RIGHT, padx=(20, 0))
        
        self.normalize_cb = ttkb.Checkbutton(
            normalize_frame,
            text="Normalize Classes",
            variable=self.normalize_var,
            bootstyle="success-round-toggle"
        )
        self.normalize_cb.pack(pady=(25, 0))

    def _build_action_section(self, parent):
        """Build action buttons section"""
        action_frame = ttkb.LabelFrame(parent, text=" Analysis Actions ", padding=15, bootstyle="warning")
        action_frame.pack(fill='x', pady=(0, 20))
        
        # Quick confusion button
        self.quick_btn = ttkb.Button(
            action_frame,
            text="🔍 Quick Confusion Matrix Analysis",
            command=self.start_quick_confusion,
            bootstyle="warning",
            style="Primary.TButton"
        )
        self.quick_btn.pack(fill='x', pady=(0, 10))
        
        # Full pipeline button
        self.full_btn = ttkb.Button(
            action_frame,
            text="🚀 Complete Pipeline Analysis",
            command=self.start_full_pipeline,
            bootstyle="success",
            style="Primary.TButton"
        )
        self.full_btn.pack(fill='x')
        
        # Progress bar (initially hidden)
        self.progress = ttkb.Progressbar(
            action_frame,
            mode='indeterminate',
            bootstyle="success-striped"
        )

    def _build_log_section(self, parent):
        """Build enhanced log section"""
        log_frame = ttkb.LabelFrame(parent, text=" Analysis Log ", padding=15, bootstyle="secondary")
        log_frame.pack(fill='both', expand=True)
        
        # Log text widget with better styling
        self.log = ttkb.ScrolledText(
            log_frame,
            height=12,
            width=50,
            font=("Consolas", 9),
            wrap='word'
        )
        self.log.pack(fill='both', expand=True)
        
        # Clear log button
        clear_btn = ttkb.Button(
            log_frame,
            text="Clear Log",
            command=self._clear_log,
            bootstyle="outline-secondary",
            style="Secondary.TButton"
        )
        clear_btn.pack(pady=(10, 0))

    def _build_enhanced_right_panel(self, parent):
        """Build enhanced right visualization panel"""
        right_frame = ttkb.LabelFrame(
            parent, 
            text=" Visualization Dashboard ",
            padding=25,
            bootstyle="info"
        )
        right_frame.pack(side=LEFT, fill='both', expand=True)
        
        # Graph selection
        graph_control_frame = ttkb.Frame(right_frame)
        graph_control_frame.pack(fill='x', pady=(0, 20))
        
        ttkb.Label(
            graph_control_frame, 
            text="Select Graph:", 
            style="Subheading.TLabel"
        ).pack(side=LEFT)
        
        self.graph_var = ttkb.StringVar()
        self.graph_cb = ttkb.Combobox(
            graph_control_frame,
            textvariable=self.graph_var,
            state="readonly",
            bootstyle="info",
            font=("Segoe UI", 10),
            width=40
        )
        self.graph_cb.pack(side=RIGHT)
        self.graph_cb.bind("<<ComboboxSelected>>", self.on_graph_selected)
        
        # Image display area with border
        img_frame = ttkb.Frame(right_frame, bootstyle="light", padding=10)
        img_frame.pack(fill='both', expand=True)
        
        self.img_label = ttkb.Label(
            img_frame,
            text="🔬 Analysis results will appear here\n\nSelect your datasets and run an analysis to begin",
            font=("Segoe UI", 12),
            bootstyle="secondary",
            anchor="center"
        )
        self.img_label.pack(fill='both', expand=True)

    def _build_status_bar(self, parent):
        """Build enhanced status bar"""
        status_frame = ttkb.Frame(parent, bootstyle="dark")
        status_frame.pack(fill='x', pady=(20, 0))
        
        self.status = ttkb.Label(
            status_frame, 
            text="Ready",
            font=("Segoe UI", 10),
            bootstyle="inverse-secondary"
        )
        self.status.pack(side=LEFT, padx=10, pady=8)
        
        # Time display
        self.time_label = ttkb.Label(
            status_frame,
            text="",
            font=("Segoe UI", 10),
            bootstyle="inverse-info"
        )
        self.time_label.pack(side=RIGHT, padx=10, pady=8)
        
        # Update time every second
        self._update_time()

    def _setup_bindings(self):
        """Setup keyboard bindings and tooltips"""
        self.bind('<Control-o>', lambda e: self.browse_fake())
        self.bind('<Control-r>', lambda e: self.browse_real())
        self.bind('<F5>', lambda e: self.start_quick_confusion() if not self._is_processing else None)
        self.bind('<F6>', lambda e: self.start_full_pipeline() if not self._is_processing else None)
        
        self.sample_spinbox.bind('<KeyRelease>', self._validate_sample_size)

    def _validate_sample_size(self, event=None):
        """Validate sample size input"""
        try:
            value = int(self.sample_var.get())
            if value < 2: self.sample_var.set(2)
            elif value > 10000: self.sample_var.set(10000)
        except ValueError:
            self.sample_var.set(200)

    def _update_time(self):
        """Update time display"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=f"Time: {current_time}")
        self.after(1000, self._update_time)

    def _update_status(self, message: str, style: str = INFO):
        """Update status bar with message and style"""
        self.status.configure(text=message, bootstyle=f"inverse-{style.lower()}")

    def _clear_log(self):
        """Clear the log area"""
        self.log.delete('1.0', 'end')
        self.before_log("Log cleared")

    def _set_processing_state(self, processing: bool):
        """Set processing state and update UI accordingly"""
        self._is_processing = processing
        if processing:
            self.quick_btn.configure(state='disabled')
            self.full_btn.configure(state='disabled')
            self.progress.pack(fill='x', pady=(10, 0))
            self.progress.start()
        else:
            self.quick_btn.configure(state='normal')
            self.full_btn.configure(state='normal')
            self.progress.stop()
            self.progress.pack_forget()

    def browse_fake(self):
        path = filedialog.askopenfilename(
            title="Select Fake Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path.cwd())
        )
        if path: self.fake_var.set(path)

    def browse_real(self):
        path = filedialog.askopenfilename(
            title="Select Real Dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path.cwd())
        )
        if path: self.real_var.set(path)

    def browse_out(self):
        path = filedialog.askdirectory(
            title="Choose Output Directory",
            initialdir=str(Path.cwd())
        )
        if path: self.out_var.set(path)

    def start_quick_confusion(self):
        if not self._validate_inputs(): return
        self._set_processing_state(True)
        self.before_log("🔍 Starting Quick Confusion Matrix Analysis...")
        threading.Thread(target=self.run_quick_confusion, daemon=True).start()

    def start_full_pipeline(self):
        if not self._validate_inputs(): return
        self._set_processing_state(True)
        self.before_log("🚀 Starting Complete Pipeline Analysis...")
        threading.Thread(target=self.run_full_pipeline, daemon=True).start()

    def _validate_inputs(self):
        if not self.fake_var.get() or not self.real_var.get():
            Messagebox.show_error("Please select both a Fake and a Real Dataset CSV file.")
            return False
        if not Path(self.fake_var.get()).exists() or not Path(self.real_var.get()).exists():
            Messagebox.show_error("One or both CSV files do not exist.")
            return False
        if not self.det_var.get().strip():
            Messagebox.show_error("Please enter a detector name.")
            return False
        return True

    def run_quick_confusion(self):
        try:
            self.after(0, lambda: self._update_status("Processing datasets...", INFO))
            start_time = time.time()
            
            # Create a unique timestamped directory for this run's output
            out_root = Path(self.out_var.get())
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir_name = f"quick_analysis_{self.det_var.get()}_{timestamp}"
            run_dir = out_root / run_dir_name
            quick_csv_dir = run_dir / "csv_ready" # Store intermediate CSV here

            combined_csv = ingest(
                [self.fake_var.get(), self.real_var.get()],
                self.det_var.get(),
                quick_csv_dir
            )
            df = pd.read_csv(combined_csv)
            self.before_log(f"✅ Loaded combined CSV with {len(df):,} rows")

            self.after(0, lambda: self._update_status("Sampling data...", INFO))
            n = self.sample_var.get()
            if self.normalize_var.get():
                self.before_log(f"📊 Sampling up to {n:,} records from each class")
                df_f = df[df.true_label == 1]
                df_r = df[df.true_label == 0]
                df_sampled = pd.concat([
                    df_f.sample(n=min(n, len(df_f)), replace=False),
                    df_r.sample(n=min(n, len(df_r)), replace=False)
                ], ignore_index=True)
            else:
                self.before_log(f"📊 Sampling up to {n:,} records from combined pool")
                df_sampled = df.sample(n=min(n, len(df)), replace=False)

            self.after(0, lambda: self._update_status("Optimizing threshold...", INFO))
            self.before_log("🎯 Performing grid search for optimal F1 threshold...")
            
            scores, labels = df_sampled.score.values, df_sampled.true_label.values
            thresholds = np.arange(0.0, 1.0001, 0.01)
            best_thr, best_f1 = 0.0, -1.0
            
            for t in thresholds:
                preds = (scores >= t).astype(int)
                tp = int(((preds == 1) & (labels == 1)).sum())
                fp = int(((preds == 1) & (labels == 0)).sum())
                fn = int(((preds == 0) & (labels == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_thr = f1, t

            self.before_log(f"🎯 Optimal threshold: {best_thr:.3f} (F1-Score: {best_f1:.3f})")

            self.after(0, lambda: self._update_status("Generating confusion matrix...", INFO))
            df_sampled["pred"] = (df_sampled.score >= best_thr).astype(int)
            cm = pd.crosstab(
                df_sampled.true_label, df_sampled.pred,
                rownames=["Actual"], colnames=["Predicted"], dropna=False
            ).reindex(index=[0,1], columns=[0,1], fill_value=0)
            self.before_log("📊 Generated confusion matrix")
            
            quick_cms_dir = run_dir / "quick_cms"
            quick_cms_dir.mkdir(exist_ok=True, parents=True)
            save_path = quick_cms_dir / f"confusion_quick_thr{best_thr:.3f}.png"
            self._show_confusion_matrix(cm, best_thr, save_path, "Quick Analysis")

            key = f"Quick Analysis (threshold={best_thr:.3f})"
            self.available_graphs[key] = save_path

            elapsed = time.time() - start_time
            self.before_log(f"✅ Quick confusion analysis completed in {elapsed:.1f}s")
            self.before_log(f"📁 Artifacts saved to: {run_dir}")
            self.after(0, lambda: self._update_status("Quick Analysis Complete ✅", SUCCESS))

        except Exception as exc:
            err_msg = f"Analysis Error:\n\n{str(exc)}"
            self.after(0, lambda: self.status.configure(text="Error ✗", bootstyle=DANGER))
            self.after(0, lambda exc=exc: Messagebox.show_error(f"Analysis Error:\n\n{str(exc)}"))
        finally:
            self.after(0, lambda: self._set_processing_state(False))

    def _show_confusion_matrix(self, cm: pd.DataFrame, thr: float, save_path: Path, title_prefix: str = ""):
        tn, fp, fn, tp = cm.values.ravel()
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        fig = go.Figure(data=go.Heatmap(
            z=cm.values,
            x=["Predicted Real", "Predicted Fake"],
            y=["Actual Real", "Actual Fake"],
            colorscale="RdYlBu_r",
            text=cm.values, texttemplate="%{text}", textfont={"size": 16, "color": "white"}
        ))

        title_text = f"{title_prefix} Confusion Matrix<br><sub>Threshold: {thr:.3f} | F1: {f1:.3f} | Accuracy: {accuracy:.3f}</sub>"
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=16), x=0.5),
            xaxis=dict(title="Predicted Class"), yaxis=dict(title="Actual Class"),
            width=600, height=500, margin=dict(t=100)
        )

        fig.write_image(str(save_path), format="png", engine="kaleido", scale=2)
        pil = Image.open(save_path)
        pil.thumbnail((600,500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.after(0, lambda: self.img_label.configure(image=photo, text=""))
        self._img_refs = [photo]

    def _refresh_dropdown(self):
        names = sorted(list(self.available_graphs.keys()))
        self.graph_cb["values"] = names
        if names:
            self.graph_cb.set(names[-1])
            self.on_graph_selected()

    def on_graph_selected(self, event=None):
        key = self.graph_var.get()
        if not key: return
        path = self.available_graphs.get(key)
        if not path or not path.exists():
            self.before_log(f"⚠️ Graph file not found: {key}")
            return
        try:
            pil = Image.open(path)
            pil.thumbnail((800, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            self.img_label.configure(image=photo, text="")
            self._img_refs = [photo]
            self.before_log(f"📊 Displaying: {key}")
        except Exception as e:
            self.before_log(f"❌ Error loading graph: {str(e)}")

    def before_log(self, text: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.after(0, lambda: self._append_log(f"[{timestamp}] {text}\n"))

    def _append_log(self, line: str):
        self.log.insert('end', line)
        self.log.see('end')
        if int(self.log.index('end-1c').split('.')[0]) > 1000:
            self.log.delete('1.0', '100.0')

    def run_full_pipeline(self):
        try:
            self.after(0, lambda: self._update_status("Initializing full pipeline...", INFO))
            start_time = time.time()
            
            # Create a unique timestamped directory for this run's output
            out_root = Path(self.out_var.get())
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir_name = f"{self.det_var.get()}_{timestamp}"
            run_dir = out_root / run_dir_name

            csv_ready = run_dir / "csv_ready"
            results = run_dir / "results"

            self.after(0, lambda: self._update_status("Preparing datasets...", INFO))
            self.before_log(f"🔄 Merging CSV files into {run_dir_name}")
            csv_path = ingest([self.fake_var.get(), self.real_var.get()],
                              self.det_var.get(), csv_ready)

            # Get the state of the checkbox
            use_balanced_sampling = self.normalize_var.get()
            
            self.after(0, lambda: self._update_status("Running R analytics engine...", INFO))
            self.before_log("⚙️ Executing R analytics engine (this may take several minutes)...")
            analyse(csv_path, results, use_balanced_sampling) # Pass the flag

            self.after(0, lambda: self._update_status("Loading analysis results...", INFO))
            self.before_log("📊 Analysis complete - loading generated plots...")
            self._populate_plots(results)
            self._show_full_confusion(results)

            elapsed = time.time() - start_time
            self.before_log(f"🎉 Full pipeline completed successfully in {elapsed:.1f}s")
            self.before_log(f"📁 All artifacts saved to: {results}")
            self.after(0, lambda: self._update_status("Full Pipeline Complete ✅", SUCCESS))

        except Exception as exc:
            err_msg = f"Analysis Error:\n\n{str(exc)}"
            self.after(0, lambda: self.status.configure(text="Error ✗", bootstyle=DANGER))
            self.after(0, lambda exc=exc: Messagebox.show_error(f"Analysis Error:\n\n{str(exc)}"))
        finally:
            self.after(0, lambda: self._set_processing_state(False))

    def _show_full_confusion(self, results: Path):
        try:
            cm_csv = next(results.glob("confusion_matrix_modalThr*.csv"), None)
            if cm_csv is None:
                self.before_log("⚠️ No confusion matrix found in results")
                return
            df_full = pd.read_csv(cm_csv, index_col=0).astype(int)
            thr_str = cm_csv.stem.split("Thr")[-1]
            best_thr = float(thr_str)
            save_path = results / f"confusion_full_thr{best_thr:.3f}.png"
            self._show_confusion_matrix(df_full, best_thr, save_path, "Full Pipeline")
            
            key = f"Full Pipeline (threshold={best_thr:.3f})"
            self.available_graphs[key] = save_path
            self._refresh_dropdown()
            self.before_log(f"📊 Full pipeline confusion matrix generated (threshold: {best_thr:.3f})")
        except Exception as e:
            self.before_log(f"⚠️ Error processing full confusion matrix: {str(e)}")

    def _populate_plots(self, results: Path):
        try:
            plot_count = 0
            for img in sorted(results.glob("*.png")):
                stem = img.stem.replace("_", " ").title()
                if "roc" in img.stem.lower():           friendly_name = "📈 " + stem
                elif "precision" in img.stem.lower():   friendly_name = "📊 " + stem
                elif "confusion" in img.stem.lower():   friendly_name = "🎯 " + stem
                elif "distribution" in img.stem.lower():friendly_name = "🧮 " + stem
                else:                                   friendly_name = "📋 " + stem
                self.available_graphs[friendly_name] = img
                plot_count += 1
            self._refresh_dropdown()
            self.before_log(f"📊 Loaded {plot_count} visualization plots")
        except Exception as e:
            self.before_log(f"⚠️ Error loading plots: {str(e)}")

def launch():
    """Launch the enhanced openAnalyze GUI application"""
    try:
        app = DetectorApp()
        app.place_window_center()
        app.before_log("🚀 openAnalyze initialized")
        app.before_log("📋 Select your CSV files and configure analysis parameters")
        app.before_log("💡 Use Ctrl+O for fake CSV, Ctrl+R for real CSV")
        app.before_log("⚡ Press F5 for Quick Analysis, F6 for Full Pipeline")
        app.mainloop()
    except Exception as e:
        print(f"Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch()