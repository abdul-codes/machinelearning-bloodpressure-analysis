from __future__ import annotations
# File handling and Management
import dataframe_image as dfi
import os
# Regular expression operations
import re
# Handle string as file-like objects
from io import StringIO
# Provide type hints
from typing import Dict, List
# CSV file reading and writing
import csv
# Numerical operations and arrays
import numpy as np
# Data manipulation and analysis
import pandas as pd
# Data Visualization and plots
import matplotlib.pyplot as plt
# Data Visualization
import seaborn as sns
# Scale features
from sklearn.preprocessing import StandardScaler
# Principal Component Analysis
from sklearn.decomposition import PCA
# K-Means clustering algorithm
from sklearn.cluster import KMeans
# Evaluate clustering performance
from sklearn.metrics import silhouette_score
# Statistical tests
from scipy.stats import kruskal

# Configuration
PROJECT_ROOT = "/home/abdul/Documents/finalProject/collab"
DATA_FILE = os.path.join(PROJECT_ROOT, "newexpanded_data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputdata")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Likert Scale
LIKERT_MAP = {"Yes": 3, "Maybe": 2, "Not sure": 1, "No": 0}


DEMOGRAPHIC_COLUMNS = [
    "Age",
    "Gender",
    "Faculty/School",
    "Year of Study",
]

# Helpers
# function to normalize text, remove brackets, quotes and strip whitespaces
def normalize_text(value):
    if value is None:
        return value
    text = str(value).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    # Remove square brackets that appear in headers
    text = text.replace("[", "").replace("]", "")
    # Remove stray quotes
    text = text.replace('"', "").replace("'", "'")
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_clean_csv(file_path: str) -> pd.DataFrame:
    """Load CSV where the header spans two lines; clean and return DataFrame.
    Uses CSV-aware parsing and constructs header as:
    [first 4 fields of line 1] + [all fields of line 2].
    Remaining lines are the data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        raise ValueError("CSV seems incomplete. Expecting at least 3 lines (2 header lines + data)")
    # Parse first two lines with csv reader to respect commas/quotes
    l1 = next(csv.reader([lines[0]]))
    l2 = next(csv.reader([lines[1]]))
    # Construct header: first 4 from line1 (demographics) + all from line2 (questions)
    header_fields = (l1[:4] if len(l1) >= 4 else l1) + l2
    # Normalize header fields
    cleaned_columns = [normalize_text(col) for col in header_fields]
    cleaned_columns = [c for c in cleaned_columns if c != ""]
    # Read remaining lines as data (no header)
    data_str = "\n".join(lines[2:])
    df = pd.read_csv(StringIO(data_str), header=None, engine="python")
    # Align number of columns if mismatch by trimming/padding (defensive)
    if df.shape[1] > len(cleaned_columns):
        df = df.iloc[:, : len(cleaned_columns)]
    elif df.shape[1] < len(cleaned_columns):
        for _ in range(len(cleaned_columns) - df.shape[1]):
            df[df.shape[1]] = np.nan
    df.columns = cleaned_columns
    # Final normalization pass on column names (collapse spaces, remove brackets)
    df.columns = [normalize_text(c) for c in df.columns]
    return df

def compute_cronbach_alpha(df_subset: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of items (columns)."""
    # Ensure numeric
    data = df_subset.apply(pd.to_numeric, errors="coerce")
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    k = data.shape[1]
    if k <= 1 or total_var == 0 or np.isnan(total_var):
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)

# Load data

df = load_and_clean_csv(DATA_FILE)
# Map Likert responses to ordinal values
# for column_name in df.columns:
#     # Standardize cell strings before mapping
#     if df[column_name].dtype == object:
#         df[column_name] = df[column_name].astype(str).str.strip()
# ... existing code ...
# Apply the mapping more explicitly to avoid FutureWarning
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = df[column_name].astype(str).str.strip()
        df[column_name] = df[column_name].map(LIKERT_MAP).fillna(df[column_name])
# ... existing code ...
df.replace(LIKERT_MAP, inplace=True)
# Keep a copy of raw (post-mapping) for reference
df_raw = df.copy()

# Define sections (use normalized question texts)

def N(text: str) -> str:
    return normalize_text(text)
sections: Dict[str, List[str]] = {
    "knowledge": [
        N("Is a blood pressure reading above  140/90 mmHg considered high blood pressure??"),
        N("Do you think people with high blood pressure experience any symptoms?"),
        N("Does high salt intake affect blood pressure levels?"),
        N("Does being overweight or obese increase the risk of high blood pressure?"),
        N("Can high blood pressure be inherited from family members?"),
        N("Does frequent stress contribute to high blood pressure?"),
        N("Can long-term exposure to environmental pollution increase the risk of high blood pressure?"),
        N("Can uncontrolled high blood pressure lead to serious health complications such as Kidney Disease?"),
    ],
    "attitude": [
        N("I believe blood pressure control is necessary for overall health?"),
        N("High blood pressure is a serious health condition that should be taken seriously, even in young people like me."),
        N("I believe High blood pressure can be prevented with lifestyle changes"),
        N("I believe I could effectively manage my blood pressure if needed?"),
        N("Regular blood pressure check-ups are important for everyone, including students."),
        N("Taking medication is important even when there are no symptoms."),
        N("I would be willing to reduce salt in my diet to lower my risk."),
    ],
    "riskperception": [
        N("I believe I could develop high blood pressure in the future."),
        N("Do you believe your current lifestyle puts you at risk of high blood pressure?"),
        N("I don’t consider myself at risk because I’m young and feel healthy."),
        N("Ignoring early signs of high blood pressure can lead to serious long-term health problems."),
        N("University students are at risk of developing high blood pressure due to stress, poor diet, and lack of exercise."),
        N("My family history increases my chances of having high blood pressure."),
    ],
    "practices": [
        N("Do you monitor your blood pressure  regularly ?"),
        N("Do you engage in physical activity most days of the week?"),
        N("Do you  limit your intake of salt or high-sodium foods such as noodles?"),
        N("Do you consume fruits and vegetables regularly ?"),
        N("Do you regularly engage in activities to manage or reduce stress (e.g., hobbies, meditation, sports)?"),
        N("Do you smoke?"),
        N("Do you drink energy drinks?"),
        N("Do you usually get 7-8 of sleep hours per night most nights?"),
    ],
}
# Normalize column names further (collapse double spaces inside questions to match spelling)
df.columns = [re.sub(r"\s+", " ", c) for c in df.columns]
for key in list(sections.keys()):
    sections[key] = [re.sub(r"\s+", " ", q) for q in sections[key]]
# Validate section columns exist
all_section_questions: List[str] = []
for q_list in sections.values():
    all_section_questions.extend(q_list)
missing_columns = [q for q in all_section_questions if q not in df.columns]
if missing_columns:
    raise ValueError(
        "Missing expected columns after cleaning: " + ", ".join(missing_columns)
    )

# Reverse-score negatively keyed items
reverse_practices = [
    N("Do you smoke?"),
    N("Do you drink energy drinks?"),
]
reverse_risk = [
    N("I don’t consider myself at risk because I’m young and feel healthy."),
]
reverse_items = [re.sub(r"\s+", " ", c) for c in (reverse_practices + reverse_risk)]
for col in reverse_items:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = 3 - df[col]

# Section composite scores and reliability

df_section_scores = pd.DataFrame(index=df.index)
for section_name, questions in sections.items():
    df_section_scores[section_name] = df[questions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
# Cronbach's alpha per section
section_reliability = {
    name: compute_cronbach_alpha(df[qs]) for name, qs in sections.items()
}
print("Section Cronbach's alpha (reliability):")
for name, alpha in section_reliability.items():
    print(f"  - {name}: {alpha:.3f}")


# Descriptive stats for numeric composites
# Descriptive statistics
percentiles = df_section_scores.quantile([0.25, 0.5, 0.75]).rename(
    index={0.25: 'p25', 0.5: 'p50', 0.75: 'p75'}
    )
descriptive_stats = pd.concat([
    df_section_scores.mean().rename('mean'),
    df_section_scores.std(ddof=1).rename('std'),
    percentiles.T
], axis=1)
descriptive_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptives_sections.csv"))
print("\nDescriptive statistics saved: descriptives_sections.csv")

# === ADDED: UNIVARIATE VISUALIZATIONS ===
# Histograms for numeric variables
numeric_vars = ['knowledge', 'attitude', 'riskperception', 'practices']
for var in numeric_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_section_scores[var].dropna(), kde=True, bins=15)
    plt.title(f'Distribution of {var.capitalize()} Score')
    plt.xlabel(f'{var.capitalize()} Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{var}_distribution_histogram.png'), dpi=200)
    plt.close()

# Bar charts for categorical variables
categorical_vars = ['Gender', 'Faculty/School', 'Year of Study', 'Age']
for var in categorical_vars:
    if var in df_raw.columns:
        plt.figure(figsize=(8, 5))
        # Get value counts and sort by descending order
        counts = df_raw[var].value_counts().reset_index()
        counts.columns = [var, 'Count']
        sns.barplot(x=var, y='Count', data=counts, order=counts[var])
        plt.title(f'Distribution of {var}')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'barchart_{var.replace(" ", "_").replace("/", "-")}.png'), dpi=200)
        plt.close()

# Pearson Correlations matrices
pearson_corr = df_section_scores.corr(method="pearson")
print(pearson_corr)

# Generate correlation heatmap visualization
plt.figure(figsize=(6, 5))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Pearson Correlation - Section Composites")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_pearson.png"), dpi=200)
plt.close()


# Strongest Pearson correlation between distinct sections
corr = pearson_corr.copy()
abs_corr = corr.abs()
np.fill_diagonal(abs_corr.values, np.nan)  # ignore self-corr = 1
(i, j) = abs_corr.stack().idxmax()
val = pearson_corr.loc[i, j]
print(f"\nKey Finding: strongest Pearson correlation is between {i} and {j}: r={val:.3f}")

# PCA on Likert items only (exclude demographics)
likert_candidate_columns = [c for c in df.columns if c not in DEMOGRAPHIC_COLUMNS]
# Ensure numeric
df_likert = pd.DataFrame(index=df.index)
for c in likert_candidate_columns:
    df_likert[c] = pd.to_numeric(df[c], errors="coerce")
# If any columns are entirely NaN after numeric coercion, drop them
df_likert = df_likert.dropna(axis=1, how="all")
scaler = StandardScaler()
scaled = scaler.fit_transform(df_likert.fillna(df_likert.mean()))
pca = PCA(n_components=None, random_state=RANDOM_SEED)
pca.fit(scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance ratio (first 10 PCs):")
print(np.round(explained_variance_ratio[:10], 4))
print("Cumulative explained variance (first 10 PCs):")
print(np.round(np.cumsum(explained_variance_ratio)[:10], 4))


# Scree and cumulative plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Component')
plt.ylabel('Cumulative Variance Ratio')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance_plots.png"), dpi=200)
plt.close()

# === FIXED: DEFINED pca_10 BEFORE K-MEANS SECTION ===
# Keep first 10 PCs
principal_components = pca.transform(scaled)
pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(principal_components.shape[1])])
pca_10 = pca_df.iloc[:, :10].copy()  # <-- MOVED HERE

# K-Means: choose k via elbow and silhouette; then cluster with k=4

wcss = []
K_RANGE = range(2, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    km.fit(pca_10)
    wcss.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(K_RANGE), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (K-Means)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(list(K_RANGE))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow.png"), dpi=200)
plt.close()
# Compute silhouette for each k
sil_scores = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(pca_10)
    sil = silhouette_score(pca_10, labels)
    sil_scores[k] = sil
best_k_sil = max(sil_scores, key=sil_scores.get)
print("\nSilhouette scores by k:")
for k, s in sil_scores.items():
    print(f"  k={k}: {s:.3f}")
print(f"Best k by silhouette: {best_k_sil}")
# Default to k=4 as planned, but report silhouette-chosen as reference

#
FINAL_K = 3
kmeans = KMeans(n_clusters=FINAL_K, random_state=RANDOM_SEED, n_init=10)
cluster_labels = kmeans.fit_predict(pca_10)
pca_10["cluster_label"] = cluster_labels



# Cluster profiling (sections)
section_profiles = df_section_scores.copy()
section_profiles["cluster_label"] = cluster_labels
cols = ["knowledge", "attitude", "riskperception", "practices"]
section_cluster_means = section_profiles.groupby("cluster_label")[cols].mean()

# Optional: sizes per cluster
cluster_sizes = section_profiles["cluster_label"].value_counts().sort_index()
print("\nCluster sizes:")
for i, n in cluster_sizes.items():
    print(f"  Cluster {i}: n={n}")
# Interpretation: highest section per cluster (+ full profile)
print("\nCluster Interpretation (by highest section mean):")
for i, row in section_cluster_means.iterrows():
    top = row.idxmax()
    print(f"  Cluster {i}: highest in {top} ({row[top]:.2f}); profile: " +
          ", ".join(f"{k}={v:.2f}" for k, v in row.sort_values(ascending=False).items()))
# Original question means per cluster (Likert items only)
orig_profiles = df_likert.copy()
orig_profiles["cluster_label"] = cluster_labels
orig_cluster_means = orig_profiles.groupby("cluster_label").mean()
# Save cluster means to CSV for inspection
section_cluster_means.to_csv(os.path.join(OUTPUT_DIR, "cluster_means_sections.csv"))
orig_cluster_means.to_csv(os.path.join(OUTPUT_DIR, "cluster_means_original_items.csv"))

# --- ADDED: Generate heatmap for original item means per cluster ---
# Plot heatmap for detailed cluster profiles (original items)
plt.figure(figsize=(12, len(orig_cluster_means.columns) * 0.4)) # Adjust height based on number of items
sns.heatmap(orig_cluster_means.T, # Transpose so items are on y-axis, clusters on x-axis
            annot=True,           # Show the mean score values on the heatmap
            fmt=".2f",            # Format the annotations to 2 decimal places
            cmap="viridis",       # Choose a colormap (viridis, coolwarm, YlGnBu are good options)
            linewidths=.5,        # Add thin lines between cells for clarity
            cbar_kws={"shrink": .8} # Shrink the color bar slightly
           )
plt.title("Cluster Profiles: Mean Scores for Original Survey Items")
plt.xlabel("Cluster Label")
plt.ylabel("Survey Question (Original Items)")
plt.yticks(rotation=0) # Keep item labels horizontal for better readability
plt.xticks(rotation=45) # Rotate cluster labels if needed
plt.tight_layout() # Adjust layout to prevent clipping
# Save the figure
heatmap_filename = os.path.join(OUTPUT_DIR, "cluster_profiles_original_items_heatmap.png")
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight') # High DPI for better quality
plt.close()
print(f"- {os.path.basename(heatmap_filename)}") # Print filename as confirmation
# --- END ADDED ---

# 2D visualization using PC1 vs PC2

viz_df = pca_df.iloc[:, :2].copy()
viz_df["cluster_label"] = cluster_labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="cluster_label", data=viz_df, palette="viridis")
plt.title("K-Means Clusters (PC1 vs PC2)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_scatter_pc1_pc2.png"), dpi=200)
plt.close()
print("- pca_variance_plots.png")
print("- kmeans_scatter_pc1_pc2.png")

# Demographics analysis: Age and Level vs K/A/R/P

# Safely extract demographics from original df_raw (post mapping of Likert but before numeric coercion)
demo_df = pd.DataFrame(index=df.index)
demo_df["Age"] = df_raw.get("Age")
demo_df["Year of Study"] = df_raw.get("Year of Study")
# Normalize common variants
if "Age" in demo_df:
    demo_df["Age"] = demo_df["Age"].astype(str).str.strip()
if "Year of Study" in demo_df:
    demo_df["Year of Study"] = demo_df["Year of Study"].astype(str).str.strip()
# Create analysis frame combining composites and demographics
composites_demo = df_section_scores.copy()
composites_demo = pd.concat([composites_demo, demo_df], axis=1)
# Define ordered categories for nicer plots
age_order = [
    "18-20",
    "21-24",
    "25-30",
]
level_order = [
    "100 Level",
    "200 Level",
    "300 Level",
    "400 Level",
    "500 Level",
]
# Group means by Age and Level
age_group_means = composites_demo.groupby("Age")[["knowledge", "attitude", "riskperception", "practices"]].mean().reindex(age_order)
level_group_means = composites_demo.groupby("Year of Study")[["knowledge", "attitude", "riskperception", "practices"]].mean().reindex(level_order)
# Save CSVs
age_group_means.to_csv(os.path.join(OUTPUT_DIR, "age_group_means_sections.csv"))
level_group_means.to_csv(os.path.join(OUTPUT_DIR, "level_group_means_sections.csv"))
# Plot heatmaps
plt.figure(figsize=(7, 4))
sns.heatmap(age_group_means, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Section Means by Age Group")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_group_means_heatmap.png"), dpi=200)
plt.close()
plt.figure(figsize=(7, 5))
sns.heatmap(level_group_means, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Section Means by Year of Study")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "level_group_means_heatmap.png"), dpi=200)
plt.close()
# Non-parametric tests (Kruskal-Wallis) across groups due to ordinal composites
print("\nDemographic effects (Kruskal-Wallis H-tests):")
for var in ["knowledge", "attitude", "riskperception", "practices"]:
    # Age
    age_groups = [
        composites_demo.loc[composites_demo["Age"] == a, var].dropna() for a in age_order if a in set(composites_demo["Age"])
    ]
    if len(age_groups) >= 2:
        stat_a, p_a = kruskal(*age_groups)
        print(f"  Age vs {var}: H={stat_a:.3f}, p={p_a:.4f}")
    # Level
    level_groups = [
        composites_demo.loc[composites_demo["Year of Study"] == l, var].dropna() for l in level_order if l in set(composites_demo["Year of Study"])
    ]
    if len(level_groups) >= 2:
        stat_l, p_l = kruskal(*level_groups)
        print(f"  Level vs {var}: H={stat_l:.3f}, p={p_l:.4f}")
print("\nArtifacts saved (demographics):")
print("- age_group_means_sections.csv")
print("- level_group_means_sections.csv")
print("- age_group_means_heatmap.png")
print("- level_group_means_heatmap.png")

# Add this after the cluster interpretation section
total_students = len(cluster_labels)
cluster_percentages = (cluster_sizes / total_students * 100).round(1)
print("\nCluster Distribution (% of total students):")
for i, pct in cluster_percentages.items():
    print(f"  Cluster {i}: {pct}% ({cluster_sizes[i]} students)")

# --- EFFICIENT METHOD: Generate descriptive statistics for all questions ---
print("\nGenerating descriptive statistics for individual questions...")

# Create a copy to safely work with, excluding the non-question 'cluster_label'
if 'cluster_label' in df_likert.columns:
    df_for_stats = df_likert.drop(columns=['cluster_label'])
else:
    df_for_stats = df_likert.copy()

# Use the .describe() method and transpose (.T) the result
desc_stats = df_for_stats.describe().T

# The question names are now the index. Let's make them a regular column.
desc_stats = desc_stats.reset_index()

# Rename columns to match the desired output format
desc_stats = desc_stats.rename(columns={
    'index': 'Question',
    'count': 'Count',
    'mean': 'Mean',
    'std': 'Std',
    'min': 'Min',
    'max': 'Max'
})

# Add the interpretation-friendly labels as before
desc_stats['Interpretation'] = desc_stats.apply(
    lambda row: f"Mean={row['Mean']:.2f}, Range={row['Min']}-{row['Max']}",
    axis=1
)

# Export the final table to a CSV file
output_path = os.path.join(OUTPUT_DIR, "question_descriptive_statistics.csv")
desc_stats.to_csv(output_path, index=False)

#print(f"\nSuccessfully generated and saved descriptive statistics to:\n{output_path}")
# print(desc_stats.head())

#print("\n--- Descriptive Statistics Table (for Presentations) ---")
#print(desc_stats.to_markdown(index=False))
# 3. Export the table as a high-quality PNG image for presentations
img_output_path = os.path.join(OUTPUT_DIR, "descriptive_statistics_table.png")
dfi.export(desc_stats, img_output_path)
