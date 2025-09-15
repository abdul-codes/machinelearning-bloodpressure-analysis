# Uncovering Behavioural Patterns in High Blood Pressure Awareness and Prevention Among Nigerian Students Using Machine Learning Methods

## About The Project

This project analyzes survey data on high blood pressure awareness among students at the Federal University of Technology, Akure (FUTA). It uses Python and machine learning to identify behavioral patterns in students' knowledge, attitudes, and practices.

The main goal is to find meaningful groups within the student population to help create targeted health campaigns.

## Getting Started

To get a local copy up and running, follow these steps.

### Prerequisites

*   Python 3
*   pip

### Installation

1.  Clone the repo.
2.  It's recommended to use a virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Analysis

To run the full analysis, simply run the main script:

```sh
python pearsoncorrelation.py
```

The script will print key findings to your console and save all generated charts and data tables into the `outputdata/` folder.

## How It Works

The `pearsoncorrelation.py` script does the following:

1.  **Loads and Cleans Data:** It reads the `bloodpressure.csv` survey data and handles its unique two-line header format.
2.  **Calculates Scores:** It computes composite scores for four main sections: Knowledge, Attitude, Risk Perception, and Practices.
3.  **Finds Correlations:** It analyzes the relationships between these sections.
4.  **Identifies Patterns:** It uses PCA and K-Means clustering to group students with similar behaviors and perceptions.
5.  **Generates Outputs:** It saves all results, including charts and summary tables, to the `outputdata/` directory.

## Outputs

The analysis generates several files in the `outputdata/` directory, including:

*   **Charts:** Heatmaps, histograms, and scatter plots visualizing the data.
*   **CSV Tables:** Detailed statistics for survey questions, cluster profiles, and demographic breakdowns.