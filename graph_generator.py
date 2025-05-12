import pandas as pd
import matplotlib.pyplot as plt


similarity_report_path = 'output/csv_files/similarity_report.csv'
df = pd.read_csv(similarity_report_path)

df = df[~df['assigned_gid'].astype(str).str.startswith('unk')]


grouped = df.groupby('assigned_gid')[['cosine_sim', 'hist_sim']].mean().reset_index()

# Cosine Similarity Histogram
plt.figure(figsize=(10, 5))
plt.bar(grouped['assigned_gid'].astype(str), grouped['cosine_sim'], color='skyblue')
plt.title('Average Cosine Similarity vs ID')
plt.xlabel('GID')
plt.ylabel('Average Cosine Similarity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Histogram Similarity Histogram
plt.figure(figsize=(10, 5))
plt.bar(grouped['assigned_gid'].astype(str), grouped['hist_sim'], color='lightgreen')
plt.title('Average Histogram Similarity vs ID')
plt.xlabel('GID')
plt.ylabel('Average Histogram Similarity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y')
plt.show()

