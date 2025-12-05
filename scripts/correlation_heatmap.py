import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# load data
df = pd.read_csv("data/qtm350_final.csv")

# define the 5 variables
vars = ['life_expectancy_birth', 'basic_water_access_pct', 'basic_sanitation_access_pct', 'primary_school_enrollment', 'gdp_per_capita_constant_usd']

# average by country
country_means = (
    df.groupby(["Region", "Country"])[vars]
      .mean()
      .reset_index()
)

# subset countries in each region
ssa = country_means[country_means["Region"] == "Sub-Saharan Africa"]
sa  = country_means[country_means["Region"] == "South Asia"]

# compute region-specific correlations
corr_ssa = ssa[vars].corr()
corr_sa  = sa[vars].corr()


# plot heatmaps
mask = np.triu(np.ones_like(corr_ssa, dtype=bool), k=1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(corr_ssa, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask)
plt.title("Correlation Matrix - Sub-Saharan Africa")

plt.subplot(1, 2, 2)
sns.heatmap(corr_sa, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask)
plt.title("Correlation Matrix - South Asia")

plt.tight_layout()

# save figure
plt.savefig("figures/correlation_heatmap.png")