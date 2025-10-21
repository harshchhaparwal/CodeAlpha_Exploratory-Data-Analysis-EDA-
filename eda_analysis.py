#Source Code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def make_plots(df, outdir):
    # Price distribution
    plt.figure(figsize=(8,5))
    df['price'].hist(bins=30)
    plt.title("Price distribution")
    plt.xlabel("Price (GBP)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "price_histogram.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    sns.boxplot(x="rating", y="price", data=df)
    plt.title("Price by Rating")
    plt.xlabel("Rating (1-5)")
    plt.ylabel("Price (GBP)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "price_by_rating_boxplot.png"))
    plt.close()
    top_categories = df['category'].value_counts().nlargest(15)
    plt.figure(figsize=(10,6))
    top_categories.plot(kind='bar')
    plt.title("Top 15 Categories (by number of books)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_categories.png"))
    plt.close()
    plt.figure(figsize=(6,4))
    df['rating'].value_counts().sort_index().plot(kind='bar')
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rating_distribution.png"))
    plt.close()
def generate_report(df, outdir):
    report_lines = []
    report_lines.append("EDA Report - Books dataset\n")
    report_lines.append(f"Total rows: {len(df)}\n")
    report_lines.append("Columns and types:\n")
    report_lines.append(str(df.dtypes) + "\n")
    report_lines.append("\nMissing values per column:\n")
    report_lines.append(str(df.isna().sum()) + "\n")
    report_lines.append("\nPrice statistics:\n")
    report_lines.append(str(df['price'].describe()) + "\n")
    report_lines.append("\nTop 10 categories:\n")
    report_lines.append(str(df['category'].value_counts().head(10)) + "\n")
    report_lines.append("\nAverage price by rating:\n")
    report_lines.append(str(df.groupby('rating')['price'].mean().sort_index()) + "\n")
    with open(os.path.join(outdir, "eda_report.txt"), "w", encoding="utf-8") as f:
        f.writelines("\n".join(report_lines))
    print("Report written to", os.path.join(outdir, "eda_report.txt"))
def main(input_csv, outdir):
    ensure_outdir(outdir)
    df = pd.read_csv(input_csv)
    # Basic cleaning/checks
    # price is assumed numeric (float)
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    print("Dataset head:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    df.sample(5).to_csv(os.path.join(outdir, "sample_rows.csv"), index=False)
    make_plots(df, outdir)
    agg = df.groupby('category').agg(
        count=('title', 'count'),
        mean_price=('price', 'mean'),
        median_price=('price', 'median')
    ).sort_values('count', ascending=False)
    agg.to_csv(os.path.join(outdir, "category_stats.csv"))
    generate_report(df, outdir)
    print("EDA complete. Outputs saved to:", outdir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA on books.csv")
    parser.add_argument("--input", "-i", default="books.csv", help="Input CSV file (from scraper)")
    parser.add_argument("--outdir", "-o", default="outputs", help="Output directory for plots and reports")
    args = parser.parse_args()
    main(args.input, args.outdir)
