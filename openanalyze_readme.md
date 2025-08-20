# openAnalyze: DeepMedia Detector Analytics

openAnalyze is a comprehensive toolkit for performance analysis of deepfake and manipulated media detectors. It provides both a command-line interface (CLI) and a graphical user interface (GUI) to streamline the process of evaluating detector models, from data ingestion to generating detailed statistical reports and visualizations.

This toolkit is designed for researchers and developers who need to rigorously assess the performance of their detection algorithms. By providing standardized metrics and visualizations, openAnalyze facilitates robust comparisons and deeper insights into a model's behavior.

## Key Features

- **Dual Interface**: Choose between an interactive command-line wizard (`openAnalyze`) for automated workflows or a user-friendly graphical interface (`openAnalyze-gui`) for visual analysis.

- **Flexible Data Ingestion**: A powerful data conversion script (`convert_to_analyzer_csv.py`) parses various detector output formats (CSV, TSV, TXT, JSON Lines, and DeepMedia results.json) into a standardized format required for analysis. It can infer "real" or "fake" labels directly from file paths.

- **Comprehensive Statistical Analysis**: The backend leverages a robust R script to perform a suite of statistical tests and generate key metrics, including:
  - Precision-Recall and F1-Score Analysis
  - Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC)
  - Detection Error Tradeoff (DET) Curve
  - Kolmogorov-Smirnov (KS) and Wilcoxon statistical tests
  - Equal Error Rate (EER)
  - Logistic Regression model summary

- **Balanced or Unbalanced Evaluation**: The R analytics script supports both balanced subsampling to prevent class imbalance from skewing results, and evaluation on the full, potentially unbalanced dataset. The GUI's "Normalize Classes" checkbox controls this feature.

- **"Quick Confusion Matrix" Mode**: The GUI offers a rapid analysis mode to quickly find the optimal F1-score threshold on a subsample of the data and visualize the resulting confusion matrix without running the full statistical suite.

- **Rich Visualizations**: Automatically generates and saves a variety of plots:
  - ROC Curve
  - Precision-Recall Curve
  - DET Curve
  - Score Distribution Violins/Boxplots
  - Calibration Curve
  - Confusion Matrix Heatmaps

- **Project Management**: The GUI organizes outputs into timestamped directories for clear and manageable experiment tracking.

## Installation

The tool is designed to be installed as a Python package.

### Prerequisites

- Python 3.9 or higher
- R environment with the following packages installed: `optparse`, `pROC`, `ROCR`, `ggplot2`, and `ResourceSelection`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/OfmanAI/openAnalyze.git
   cd openAnalyze
   ```

2. Install Python dependencies: The project dependencies are specified in `pyproject.toml` and `requirements.txt`. Install them using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   or install the project in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

openAnalyze provides two primary entry points: a command-line interface and a graphical user interface.

### Graphical User Interface (GUI)

The GUI provides an intuitive, visual way to perform analyses.

To launch the GUI, run:
```bash
openAnalyze-gui
```

#### Workflow:

1. **Load Data**: Use the "Browse" buttons to select your "Fake" and "Real" dataset CSV files. These files should contain at least `filename` and `score` columns.

2. **Configure Analysis**:
   - **Detector Name**: Assign a name for your analysis run.
   - **Output Directory**: Choose where the results will be saved.
   - **Sample Size**: Specify the number of records to sample for the "Quick Confusion Matrix" analysis.
   - **Normalize Classes**: Check this box (default) to perform balanced sampling, ensuring an equal number of real and fake samples are used in the R analysis backend. Uncheck it to use the full dataset.

3. **Run Analysis**:
   - **Quick Confusion Matrix Analysis**: Click for a fast analysis that identifies the best F1-score threshold and displays a confusion matrix. Results are saved in timestamped `quick_analysis_*` folders.
   - **Complete Pipeline Analysis**: Click to execute the full R statistical analysis suite. All generated plots and data files will be saved to a timestamped folder within your chosen output directory.

4. **View Results**: The "Visualization Dashboard" on the right will populate with generated plots. Select any graph from the dropdown menu to view it. The log panel provides real-time updates on the analysis progress.

### Command-Line Interface (CLI)

The CLI is an interactive wizard that guides you through the analysis process step-by-step.

To launch the CLI, run:
```bash
openAnalyze
```

#### Workflow:

1. You will be prompted to enter the file path for your FAKE results CSV.

2. Next, provide the path for your REAL results CSV. This step is optional; you can press Enter to proceed with only fake data.

3. The tool automatically detects the score column (e.g., `score`, `confidence`). If it cannot, it will ask you to select the correct column from a list.

4. If the real dataset is larger than the fake dataset, you will be asked if you want to downsample the real data to create a balanced dataset for analysis.

5. Choose whether to specify a custom detection threshold (e.g., 0.5) or have the tool automatically calculate the optimal F1 threshold.

6. Finally, specify an output directory for the results.

7. The analysis will run, and all artifacts (plots, summary files, confusion matrix) will be saved to the specified directory.

### Data Preparation Utility

If your detector outputs are not in a simple `filename,score` CSV format, use the `convert_to_analyzer_csv.py` script to standardize them.

#### Usage Example:

```bash
python3 convert_to_analyzer_csv.py \
        --input /path/to/my/fakes_dump.txt /path/to/reals_dump.txt \
        --detector my_detector_v1 \
        --out_dir csv_for_analysis \
        --combine
```

This command processes `fakes_dump.txt` and `reals_dump.txt`, infers the true labels from the file names, and outputs a single combined file `csv_for_analysis/my_detector_v1.csv` with `filename`, `score`, and `true_label` columns, ready for use in the openAnalyze tools.

## Project Structure

```
openAnalyze/
├── src/
│   ├── openAnalyze.py                              # The core CLI application logic
│   ├── openAnalyze_gui.py                          # The main GUI application window and logic
│   ├── deepfake_detection_threshold_analytics_balanced.R  # R script for statistical analysis
│   └── convert_to_analyzer_csv.py                  # Standalone utility for parsing detector outputs
├── pyproject.toml                                  # Project metadata and build specifications
├── setup.py                                        # Setup script
├── requirements.txt                                # Python package dependencies
├── dummy_reals.csv                                 # Sample data file
└── README.md                                       # This file
```

## Requirements

Create a `requirements.txt` file with the necessary Python dependencies:

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tkinter
argparse
json
csv
os
sys
subprocess
datetime
```

## R Dependencies

Ensure the following R packages are installed:

```r
install.packages(c("optparse", "pROC", "ROCR", "ggplot2", "ResourceSelection"))
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

## Citation

If you use openAnalyze in your research, please cite:

```bibtex
@software{openanalyze2024,
  title={openAnalyze: DeepMedia Detector Analytics},
  author={OfmanAI},
  year={2024},
  url={https://github.com/OfmanAI/openAnalyze}
}
```