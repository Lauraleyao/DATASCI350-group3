# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: fig-heatmap
#| fig-cap: "Correlation Heatmaps for Sub-Saharan Africa and South Asia"

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
plt.title("Correlation Matrix – Sub-Saharan Africa")

plt.subplot(1, 2, 2)
sns.heatmap(corr_sa, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask)
plt.title("Correlation Matrix – South Asia")

plt.tight_layout()
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| include: false
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# connect to csv file
df = pd.read_csv("data/qtm350_final.csv")

#Check for NAs and drop them
df[[
    "life_expectancy_birth",
    "basic_water_access_pct",
    "basic_sanitation_access_pct",
    "primary_school_enrollment",
    "gdp_per_capita_constant_usd"
]].isna().sum()

df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
    "life_expectancy_birth",
    "basic_water_access_pct",
    "basic_sanitation_access_pct",
    "primary_school_enrollment",
    "gdp_per_capita_constant_usd"
])
#
#
#
#| echo: false 
#| label: tbl-lmsimplest
#| title: "Linear Regression Model Output"
#| tbl-cap: "Summary of linear regression where region is not a variable"

#One multiple regression ran on all variables, not including region

Y = df_clean["life_expectancy_birth"]
X = df_clean[[
    "basic_water_access_pct",
    "basic_sanitation_access_pct",
    "primary_school_enrollment",
    "gdp_per_capita_constant_usd"
]]

X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false 
#| label: tbl-lmparallel
#| title: "Parallel Lines Model Output"
#| tbl-cap: "Summary of parallel lines model"

#Create column for Sub-Saharan Africa binary variable (1=Africa, 0=Asia)
df_clean["Binary_subsaharan_africa"] = (df_clean["Region"] == "Sub-Saharan Africa").astype(int)

#Run regression with region, parallel lines model (where two regions have parallel lines: same slope, different intercepts)

Y = df_clean["life_expectancy_birth"]
X = df_clean[[
    "basic_water_access_pct",
    "basic_sanitation_access_pct",
    "primary_school_enrollment",
    "gdp_per_capita_constant_usd",
    "Binary_subsaharan_africa"
]]

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| label: fig-parallelwatervslifeexp
#| fig-cap: "Water Access: Predicted vs Actual (Parallel Lines Model)"

x_var = "basic_water_access_pct"
label = "Water Access (%)"
x_range = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 150)

# Mean values for other predictors
means = {
    "basic_sanitation_access_pct": df_clean["basic_sanitation_access_pct"].mean(),
    "primary_school_enrollment": df_clean["primary_school_enrollment"].mean(),
    "gdp_per_capita_constant_usd": df_clean["gdp_per_capita_constant_usd"].mean()
}

# Predicted life expectancy for Asia
pred_asia = (
    model.params["const"]
    + model.params["basic_water_access_pct"] * x_range
    + model.params["basic_sanitation_access_pct"] * means["basic_sanitation_access_pct"]
    + model.params["primary_school_enrollment"] * means["primary_school_enrollment"]
    + model.params["gdp_per_capita_constant_usd"] * means["gdp_per_capita_constant_usd"]
)

# Africa prediction = Asia + region intercept
pred_africa = pred_asia + model.params["Binary_subsaharan_africa"]

plt.figure(figsize=(7,5))

# Asia scatter (blue)
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, "life_expectancy_birth"],
    c="blue",
    alpha=0.6,
    label="Asia (Actual)"
)

# Africa scatter (red)
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, "life_expectancy_birth"],
    c="red",
    alpha=0.4,
    label="Africa (Actual)"
)

# --- Predicted lines ---
plt.plot(x_range, pred_asia, "k--", label="Asia (Predicted)")
plt.plot(x_range, pred_africa, "k-", label="Africa (Predicted)")

plt.xlabel(label)
plt.ylabel("Life Expectancy at Birth")
plt.title("Life Expectancy vs Water Access — Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```
#
#| echo: false
#| label: fig-parallelsanitationvslifeexp
#| fig-cap: "Sanitation Access: Predicted vs Actual (Parallel Lines Model)"

x_var = "basic_sanitation_access_pct"
x_range = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 150)

# Mean values for other predictors
means = {
    "basic_water_access_pct": df_clean["basic_water_access_pct"].mean(),
    "primary_school_enrollment": df_clean["primary_school_enrollment"].mean(),
    "gdp_per_capita_constant_usd": df_clean["gdp_per_capita_constant_usd"].mean()
}

# Predictions for Asia
pred_asia = (
    model.params["const"]
    + model.params["basic_water_access_pct"] * means["basic_water_access_pct"]
    + model.params["primary_school_enrollment"] * means["primary_school_enrollment"]
    + model.params["gdp_per_capita_constant_usd"] * means["gdp_per_capita_constant_usd"]
    + model.params["basic_sanitation_access_pct"] * x_range
)
# Predictions for Africa
pred_africa = pred_asia + model.params["Binary_subsaharan_africa"]

# Plot
plt.figure(figsize=(7,5))
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, "life_expectancy_birth"],
    c="blue", alpha=0.6, label="Asia (Actual)"
)
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, "life_expectancy_birth"],
    c="red", alpha=0.4, label="Africa (Actual)"
)
plt.plot(x_range, pred_asia, "k--", label="Asia (Predicted)")
plt.plot(x_range, pred_africa, "k-", label="Africa (Predicted)")

plt.xlabel("Basic Sanitation Access (%)")
plt.ylabel("Life Expectancy at Birth")
plt.title("Life Expectancy vs Sanitation Access — Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
#
#
#
#| echo: false
#| label: fig-parallelschoolvslifeexp
#| fig-cap: "Primary School Enrollment: Predicted vs Actual (Parallel Lines Model)"

x_var = "primary_school_enrollment"
x_range = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 150)

# Mean values for other predictors
means = {
    "basic_water_access_pct": df_clean["basic_water_access_pct"].mean(),
    "basic_sanitation_access_pct": df_clean["basic_sanitation_access_pct"].mean(),
    "gdp_per_capita_constant_usd": df_clean["gdp_per_capita_constant_usd"].mean()
}

# Predictions for Asia
pred_asia = (
    model.params["const"]
    + model.params["basic_water_access_pct"] * means["basic_water_access_pct"]
    + model.params["basic_sanitation_access_pct"] * means["basic_sanitation_access_pct"]
    + model.params["gdp_per_capita_constant_usd"] * means["gdp_per_capita_constant_usd"]
    + model.params["primary_school_enrollment"] * x_range
)
# Predictions for Africa
pred_africa = pred_asia + model.params["Binary_subsaharan_africa"]

# Plot
plt.figure(figsize=(7,5))
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 0, "life_expectancy_birth"],
    c="blue", alpha=0.6, label="Asia (Actual)"
)
plt.scatter(
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, x_var],
    df_clean.loc[df_clean["Binary_subsaharan_africa"] == 1, "life_expectancy_birth"],
    c="red", alpha=0.4, label="Africa (Actual)"
)
plt.plot(x_range, pred_asia, "k--", label="Asia (Predicted)")
plt.plot(x_range, pred_africa, "k-", label="Africa (Predicted)")

plt.xlabel("Primary School Enrollment (%)")
plt.ylabel("Life Expectancy at Birth")
plt.title("Life Expectancy vs Primary School Enrollment — Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#
#
#
#
#
#
#| echo: false 
#| label: tbl-lminteraction
#| title: "Interaction Model Output"
#| tbl-cap: "Summary of interaction model"

#Interaction model (where two regions have different slopes and intercepts)

# Create interaction terms
df_clean["water_x_region"] = df_clean["basic_water_access_pct"] * df_clean["Binary_subsaharan_africa"]
df_clean["sanitation_x_region"] = df_clean["basic_sanitation_access_pct"] * df_clean["Binary_subsaharan_africa"]
df_clean["school_x_region"] = df_clean["primary_school_enrollment"] * df_clean["Binary_subsaharan_africa"]
df_clean["gdp_x_region"] = df_clean["gdp_per_capita_constant_usd"] * df_clean["Binary_subsaharan_africa"]

# Y variable
Y = df_clean["life_expectancy_birth"]

# X variables with interactions
X = df_clean[[
    "basic_water_access_pct",
    "basic_sanitation_access_pct",
    "primary_school_enrollment",
    "gdp_per_capita_constant_usd",
    "Binary_subsaharan_africa",
    "water_x_region",
    "sanitation_x_region",
    "school_x_region",
    "gdp_x_region"
]]

X = sm.add_constant(X)
model_interaction = sm.OLS(Y, X).fit()
print(model_interaction.summary())
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
