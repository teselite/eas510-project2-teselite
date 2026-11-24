# Why Aren't You Famous Yet?

**Teslim Carew**  
EAS 510: The Basics of AI — Fall 2026 (Project 2)  
Final Report:

## About This Project
Small content creators deal with a tough cold start problem on social media because recommendation algorithms tend to favor creators that already have strong engagement. For this project, I am looking at YouTube trending video data to see which metadata and engagement-rate features actually separate breakout ("amplified") videos from standard ones.

I framed this as a binary classification problem where the top 20% of videos by peak views are labeled as **Amplified** and the rest are **Standard**. To avoid leakage, I excluded raw view, like, and comment counts from the features and only used engagement rates and video metadata such as like title length, tag count, publish time, and category.

## Logistic Regression
| Run | Dataset | Unique Videos | Top 20% View Cutoff | Precision | Recall | F1 Score | ROC AUC | Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Sample | `data/sample/youtube_sample.csv` | 1,974 | 2,176,154 | 0.2919 | 0.5949 | 0.3917 | 0.6390 | [results/sample/baseline_metrics.json](results/sample/baseline_metrics.json) |
| Full | `data/raw/USvideos.csv` | 6,351 | 1,888,184 | 0.2998 | 0.6693 | 0.4141 | 0.7045 | [results/full/baseline_metrics.json](results/full/baseline_metrics.json) |

## Environment
- **System:** macOS 
- **Python:** Python 
- **Packages:** (`numpy`, `pandas`, `scikit-learn`, `matplotlib`).
  
## How to Run



### 1. Install packages and run on the sample
```bash
python -m pip install -r requirements.txt
python main.py
```

### 3. Run on the full dataset 
1. Download `USvideos.csv` from Kaggle: [Trending YouTube Video Statistics (`datasnaek/youtube-new`)](https://www.kaggle.com/datasets/datasnaek/youtube-new).
2. Put the file in `data/raw/USvideos.csv`.
3. Run:
```bash
python main.py --full
```

## Expected Outputs
- Running `python main.py` prints the classification report and saves the metrics to [results/sample/baseline_metrics.json](results/sample/baseline_metrics.json).
- Expected sample numbers: Precision ~0.2919, Recall ~0.5949, F1 ~0.3917, ROC AUC ~0.6390. Randomness is controlled in `config.py`.
- Running `python main.py --full` saves the full-dataset metrics to [results/full/baseline_metrics.json](results/full/baseline_metrics.json).





















## Data & Compute Info
- **Source:** Kaggle *Trending YouTube Video Statistics* (`datasnaek/youtube-new`, CC0 Public Domain), downloaded November 23, 2026.
- **Sample creation:** `data/sample/youtube_sample.csv` is a 2,500-row random sample (`random_state=42`) taken from the 40,949 rows in `USvideos.csv` so anyone can run the code without downloading the full 60 MB file.
