import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by converting cholesterol and gluc to 0 if normal, 1 if above normal
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Categorical plot function
def draw_cat_plot():
    # 5. Create DataFrame for `cat plot` using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=[
            'active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'
        ]
    )

    # 6. Group and reformat the data to split by 'cardio', then count 'value's. Reset index.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the catplot with 'sns.catplot()'
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        data=df_cat
    )

    # 8. Get the figure for the output
    fig = g.fig

    # 9. Save
    fig.savefig('catplot.png')
    return fig

# 10. Heat map function
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15. Draw the heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        center=0,
        vmax=0.32,
        vmin=-0.16
    )

    # 16. Save
    fig.savefig('heatmap.png')
    return fig
