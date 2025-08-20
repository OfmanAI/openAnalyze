#!/usr/bin/env python3
"""
openAnalyze.py – One‑shot DeepMedia detector analytics CLI (FINAL, WORKING VERSION)
───────────────────────────────────────────────────────────
"""

from __future__ import annotations
import subprocess
import sys
import datetime
import re
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich.text import Text

THIS = Path(__file__).resolve()
ROOT = THIS.parent
console = Console()

# ───────────────────────── helpers ──────────────────────────

def stamp() -> str:
    """Returns a non-conflicting timestamp string like '14:35:01 |'."""
    return datetime.datetime.now().strftime("%H:%M:%S |")

def run_quiet(cmd: list[str], **kw) -> None:
    """Run subprocess, echo command, and stream its output."""
    console.print(f"{stamp()}$ {' '.join(cmd)}", style="cyan")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if not re.match(r"^\s*(null device|dev\.off\(\))\s*$", line.strip()):
                     console.out(line, highlight=False)
            process.stdout.close()
        
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)

    except subprocess.CalledProcessError as e:
        console.print(f"{stamp()}▶ Command failed with exit code {e.returncode}", style="red")
        raise

def read_csv_autodetect_sep(path: Path) -> pd.DataFrame:
    """Reads a CSV, trying comma and then semicolon as a separator."""
    try:
        df = pd.read_csv(path, sep=',')
        if len(df.columns) <= 1:
            raise pd.errors.ParserError("Only one column found with comma delimiter, trying another.")
        console.print(f"{stamp()}Successfully read CSV using comma (',') delimiter.")
        return df
    except (pd.errors.ParserError, UnicodeDecodeError):
        try:
            df = pd.read_csv(path, sep=';')
            console.print(f"{stamp()}Successfully read CSV using semicolon (';') delimiter.")
            return df
        except Exception as e:
            console.print(f"Error: Could not parse CSV file '{path}'.", style="bold red")
            console.print("The file could not be read with either a comma (,) or a semicolon (;) delimiter.", style="red")
            raise e

def find_score_column(df: pd.DataFrame) -> str:
    """Automatically find the score column or ask the user."""
    common_names = ["score", "final_ensemble_score", "confidence"]
    for name in common_names:
        if name in df.columns:
            return name
            
    console.print("Could not automatically identify the score column.", style="yellow")
    col_map = {str(i+1): col for i, col in enumerate(df.columns)}
    
    for i, col in col_map.items():
        console.print(Text(f"  {i}: ", style="bold"), col)
        
    choice = Prompt.ask("Please enter the number for the column representing the detection score", choices=col_map.keys())
    return col_map[choice]

def prompt_for_file(prompt_text: str, optional: bool = False) -> Path | None:
    """Prompts user for a file path and validates it."""
    while True:
        path_str = Prompt.ask(Text(prompt_text, style="bold"))
        if not path_str and optional:
            return None
        
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            return path
        console.print(f"Error: File not found at '{path_str}'. Please try again.", style="red")


# ───────────────────── ingest (now takes dataframes) ─────────────────────

def ingest(fake_df: pd.DataFrame, real_df: pd.DataFrame | None, output_dir: Path) -> Path:
    """Combines dataframes and saves a single CSV for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "combined_data_for_analysis.csv"

    fake_df['true_label'] = 1
    
    if real_df is not None:
        real_df['true_label'] = 0
        combined = pd.concat([fake_df, real_df], ignore_index=True)
    else:
        combined = fake_df
        
    combined.to_csv(out_csv, index=False)
    console.print(f"{stamp()}✓ Wrote combined data to {out_csv} ({len(combined):,} rows)")
    return out_csv


# ─────────────────── analyse (R balanced) ───────────────────

def analyse(csv_path: Path, results_dir: Path, filename_suffix: str, threshold: float | None = None) -> None:
    """Calls the R analysis script with optional user-defined threshold and filename suffix."""
    r_script = ROOT / "targeted_Diagnostic.R"
    if not r_script.exists():
        console.print(f"Error: Analysis script not found at {r_script}", style="red")
        sys.exit(1)
        
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "Rscript", "--vanilla", str(r_script),
        "--single", str(csv_path.resolve()),
        "--out", str(results_dir.resolve()),
        "--suffix", filename_suffix
    ]
    
    if threshold is not None:
        cmd.extend(["--use-threshold", str(threshold)])

    run_quiet(cmd)


# ───────────────── New Interactive Main Function ─────────────────

def main():
    """The new interactive command-line interface for the openAnalyze pipeline."""
    console.print("\nWelcome to the openAnalyze Interactive Analysis Pipeline", style="bold cyan")
    console.print("This wizard will guide you through the analysis process.")

    try:
        # 1. Get fake results
        fake_path = prompt_for_file("Enter the path to the FAKE results CSV")
        if not fake_path: return
        
        fake_df_raw = read_csv_autodetect_sep(fake_path)
        
        score_col = find_score_column(fake_df_raw)

        console.print(
            stamp(),
            "Using column ",
            Text(score_col, style="bold green"),
            " as the score for fakes."
        )
        
        if 'filename' not in fake_df_raw.columns:
            console.print("Error: Input CSV must contain a 'filename' column.", style="bold red")
            sys.exit(1)
            
        fake_df = fake_df_raw[['filename', score_col]].rename(columns={score_col: "score"})

        # 2. Get real results (optional)
        real_path = prompt_for_file("Enter the path to the REAL results CSV (or press Enter to skip)", optional=True)
        real_df = None
        if real_path:
            real_df_raw = read_csv_autodetect_sep(real_path)
            real_score_col = find_score_column(real_df_raw)
            
            if 'filename' not in real_df_raw.columns:
                console.print("Error: Input CSV for real data must contain a 'filename' column.", style="bold red")
                sys.exit(1)
                
            real_df = real_df_raw[['filename', real_score_col]].rename(columns={real_score_col: "score"})
            console.print(f"{stamp()}Loaded and filtered real data.")

            # --- THIS IS THE NEW NORMALIZATION LOGIC ---
            if len(real_df) > len(fake_df):
                num_fakes = len(fake_df)
                num_reals = len(real_df)
                
                prompt_text = Text(f"\nYour 'real' dataset ({num_reals} rows) is larger than your 'fake' dataset ({num_fakes} rows).\nDo you want to randomly downsample the real data to match the fake data count? This can help balance the analysis.")
                prompt_text.stylize("bold")

                normalize = Confirm.ask(prompt_text, default=True)

                if normalize:
                    real_df = real_df.sample(n=num_fakes, random_state=42)
                    console.print(f"{stamp()}Normalized 'real' dataset by sampling down to {len(real_df)} rows.")


        # 3. Determine thresholding method
        custom_threshold = None
        while True:
            prompt_text = Text("\nEnter a custom threshold (e.g., 0.5), or press Enter to find the optimal one:")
            prompt_text.stylize("bold")
            
            threshold_str = Prompt.ask(prompt_text, default="")
            
            if threshold_str == "":
                custom_threshold = None
                console.print(f"{stamp()}No threshold entered. The optimal F1 threshold will be calculated.")
                break
            else:
                try:
                    custom_threshold = float(threshold_str)
                    if 0.0 <= custom_threshold <= 1.0:
                        console.print(f"{stamp()}Using custom threshold: {custom_threshold}")
                        break
                    else:
                        console.print("Error: Threshold must be between 0.0 and 1.0.", style="red")
                except ValueError:
                    console.print("Error: Invalid input. Please enter a number or press Enter.", style="red")

        # 4. Get output location
        default_output = Path.cwd() / f"analysis_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        output_str = Prompt.ask(Text("\nWhere would you like the analysis output saved?", style="bold"), default=str(default_output))
        output_dir = Path(output_str).expanduser()

        # 5. Run pipeline
        console.print("\nConfiguration complete. Starting analysis...", style="bold green")
        
        temp_dir = output_dir / "temp_data"
        combined_csv_path = ingest(fake_df, real_df, temp_dir)
        
        fake_filename_stem = fake_path.stem
        analyse(combined_csv_path, output_dir, filename_suffix=fake_filename_stem, threshold=custom_threshold)

        console.print("\n✓ Analysis Complete!", style="bold bright_green")
        console.print("All artifacts have been saved to: ", Text(str(output_dir.resolve()), style="underline"))

    except (KeyboardInterrupt, EOFError):
        console.print("\nAnalysis cancelled by user.", style="yellow")
    except Exception:
        console.print("\nAn unexpected error occurred:", style="bold red")
        console.print_exception(show_locals=False)


if __name__ == "__main__":
    main()