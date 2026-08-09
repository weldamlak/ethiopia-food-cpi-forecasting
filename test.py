import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10

# ==========================================
# 1. CREATE SYNTHETIC DATASET (Ethiopia Regional Secondary Data)
# ==========================================
np.random.seed(42)

regions = [
    "Addis Ababa",
    "Oromia",
    "Amhara",
    "Sidama",
    "SNNP",
    "Tigray",
    "Dire Dawa",
    "Harari",
    "Afar",
    "Benishangul-Gumuz",
]

data = []
for region in regions:
    # Base enrollment scale based on regional size
    base_male = (
        np.random.randint(15000, 85000)
        if region in ["Oromia", "Amhara"]
        else np.random.randint(3000, 20000)
    )
    # Gender enrollment ratio (~45-52% female enrollment)
    female_ratio = np.random.uniform(0.45, 0.52)
    total_enrolled = int(base_male / (1 - female_ratio))
    female_enrolled = int(total_enrolled * female_ratio)
    male_enrolled = total_enrolled - female_enrolled

    # Regional pass rates (mirroring national exam ranges: ~3% to 15%)
    base_pass = (
        np.random.uniform(0.08, 0.16)
        if region in ["Addis Ababa", "Harari", "Dire Dawa"]
        else np.random.uniform(0.03, 0.08)
    )

    male_passed = int(male_enrolled * base_pass)
    # Introducing a slight gender gap in pass rates
    female_passed = int(female_enrolled * (base_pass * np.random.uniform(0.85, 0.98)))

    data.append(
        {
            "Region": region,
            "Male_Enrolled": male_enrolled,
            "Female_Enrolled": female_enrolled,
            "Male_Passed": male_passed,
            "Female_Passed": female_passed,
        }
    )

df = pd.DataFrame(data)

# ==========================================
# 2. FEATURE ENGINEERING & CALCULATIONS
# ==========================================
df["Total_Enrolled"] = df["Male_Enrolled"] + df["Female_Enrolled"]
df["Total_Passed"] = df["Male_Passed"] + df["Female_Passed"]

# Key Performance Indicators
df["Overall_Pass_Rate_%"] = round((df["Total_Passed"] / df["Total_Enrolled"]) * 100, 2)
df["Male_Pass_Rate_%"] = round((df["Male_Passed"] / df["Male_Enrolled"]) * 100, 2)
df["Female_Pass_Rate_%"] = round(
    (df["Female_Passed"] / df["Female_Enrolled"]) * 100, 2
)

# Gender Parity Index (GPI) for Pass Rates (< 1.0 indicates advantage toward males)
df["Pass_Rate_GPI"] = round(
    df["Female_Pass_Rate_%"] / df["Male_Pass_Rate_%"], 2
)

print("--- REGIONAL SUMMARY TABLE ---")
print(
    df[
        [
            "Region",
            "Total_Enrolled",
            "Overall_Pass_Rate_%",
            "Male_Pass_Rate_%",
            "Female_Pass_Rate_%",
            "Pass_Rate_GPI",
        ]
    ]
)

# ==========================================
# 3. VISUALIZATIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Total Enrollment by Gender per Region
df_melted_enroll = df.melt(
    id_vars=["Region"],
    value_vars=["Male_Enrolled", "Female_Enrolled"],
    var_name="Gender",
    value_name="Count",
)
df_melted_enroll["Gender"] = df_melted_enroll["Gender"].str.replace("_Enrolled", "")

sns.barplot(
    data=df_melted_enroll,
    x="Count",
    y="Region",
    hue="Gender",
    ax=axes[0],
    palette="Blues_r",
)
axes[0].set_title("Student Enrollment by Region & Gender", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Students Enrolled")

# Plot 2: Pass Rate Comparison (Male vs Female)
df_melted_pass = df.melt(
    id_vars=["Region"],
    value_vars=["Male_Pass_Rate_%", "Female_Pass_Rate_%"],
    var_name="Gender",
    value_name="Pass_Rate",
)
df_melted_pass["Gender"] = df_melted_pass["Gender"].str.replace("_Pass_Rate_%", "")

sns.barplot(
    data=df_melted_pass,
    x="Pass_Rate",
    y="Region",
    hue="Gender",
    ax=axes[1],
    palette="magma",
)
axes[1].set_title("National Exam Pass Rates (%) by Gender", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Pass Rate (%)")

plt.tight_layout()
plt.show()