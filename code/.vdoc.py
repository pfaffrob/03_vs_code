# type: ignore
# flake8: noqa
#
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator
#
#
#

grey = "#595857"
dark_grey = '#383838'
red = '#FF0000'
dark_red = '#8B0000'

colors = [grey, dark_red]
#
#
#
#
#
#
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/forest_cover_year_2000.csv')
df2 = pd.read_csv('../code/results/tables/primary_forest_2001.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'primary forest'
df2.rename(columns={'Area (ha)': 'primary forest'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['Non primary forest (2000)'] = merged_df['total'] - merged_df['primary forest']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)

# Define the total value
total_island = 72879900

# Calculate the sums of 'primary forest' and 'non primary forest'
primary_forest_sum = merged_df['primary forest'].sum()
non_primary_forest_sum = merged_df['Non primary forest (2000)'].sum()

# Define the total value
total_island = 72879900
island_no_forest = total_island - primary_forest_sum - non_primary_forest_sum
# Create a list of the sums
forest_sums = [primary_forest_sum, non_primary_forest_sum, island_no_forest]

# Create a list of labels
labels = ['primary forest (2001)', 'secondary forest', 'non-forest']
colors = [dark_red, red, grey]
# Create a figure


# Increase the size of all text by 20%
plt.rcParams.update({'font.size': 12 * 1})


fig, ax = plt.subplots()

plt.pie(forest_sums, labels = labels, colors = colors, autopct=lambda p: '{:.1f}%\n{:.0f} km²'.format(p,(p/100)*total_island/100))

# Create a new axes for the text
ax_text = fig.add_axes([0.5, 0.01, 0.1, 0.1])
ax_text.axis('off')  # Hide the axes

# Add text to the new axes
ax_text.text(0.5, 0.5, 'Total area of Borneo: {:.0f} km²'.format(total_island/100), 
             horizontalalignment='center', verticalalignment='center', 
             fontweight = 'bold', transform=ax_text.transAxes)

# Display the chart
plt.tight_layout()

# Save the figure in high resolution
fig.savefig('results/final_plots/fcover_2000.png', dpi=120)

# Then show the plot
plt.show()

plt.rcParams.update({'font.size': 12})

#
#
#
#
#


def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/gfc_deforestation_total_yearly.csv')
df2 = pd.read_csv('../code/results/tables/forest_fires_loss_total_yearly.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'forest fires'
df2.rename(columns={'Area (ha)': 'forest fires'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')

# Create a new column 'loss_no_forest_fires'
merged_df['logging'] = merged_df['total'] - merged_df['forest fires']

for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)

# Convert 'Year' to integer
merged_df['Year'] = merged_df['Year'].astype(int)
df_complete_deforestation = merged_df

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/primary_forest_loss_total_yearly.csv')
df2 = pd.read_csv('../code/results/tables/primary_forest_loss_forest_fires_total_yearly.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'forest fires'
df2.rename(columns={'Area (ha)': 'forest fires'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['logging'] = merged_df['total'] - merged_df['forest fires']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)

merged_df['Year'] = merged_df['Year'] + 2000

df_primary = merged_df  


# Subtract values from the second merged DataFrame from the first merged DataFrame
for col in df_complete_deforestation.columns:
    if col != 'Year':
        df_complete_deforestation[col] = df_complete_deforestation[col] - df_primary[col]
merged_df = df_complete_deforestation

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='logging', data=merged_df, label='logging', alpha=1, ax=ax, color=colors[0], zorder=2)
sns.barplot(x='Year', y='forest fires', data=merged_df, label='forest fires', alpha=1, ax=ax, color=colors[1], zorder=3, bottom=merged_df['logging'])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))
# Set title and labels
ax.set_xlabel('year')
ax.set_ylabel('forest loss Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = (merged_df['logging'] + merged_df['forest fires']).max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(merged_df['Year']))))
ax.set_xticklabels(merged_df['Year'], rotation=90)

# Show the plot
legend = plt.legend(loc='upper left')
legend.get_texts()[0].set_text('logging')
legend.get_texts()[1].set_text('forest fires')

# Calculate the sum of the 'logging' column
total_sum = merged_df['total'].sum()
logging_sum = merged_df['logging'].sum()
fire_sum = merged_df['forest fires'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'total forest loss: {round(total_sum/1000000, 2)} Mha \nlogging: {round(logging_sum/1000000, 2)} Mha \nfires: {round(fire_sum/1000000, 2)} Mha'
ax.set_ylim(0,1000000)
# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 0.755, s=custom_legend_entry, fontsize=12)
plt.tight_layout()

fig.savefig('results/final_plots/total_deforestation.png', dpi=120)
plt.show()




#
#
#
#
#
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/deforestation_protected_areas_yearly.csv')
df2 = pd.read_csv('../code/results/tables/deforestation_no_forest_fires_protected_areas_yearly.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'primary forest'
df2.rename(columns={'Area (ha)': 'logging'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['forest fires'] = merged_df['total'] - merged_df['logging']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)
merged_df['Year'] = merged_df['Year'] + 2000
# Define your colors

df_complete_deforestation = merged_df


df1 = pd.read_csv('../code/results/tables/deforestation_primary_forest_protected_areas.csv')
df2 = pd.read_csv('../code/results/tables/deforestation_primary_forest_no_forest_fires_protected_areas.csv')


# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'forest fires'
df2.rename(columns={'Area (ha)': 'forest fires'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['logging'] = merged_df['total'] - merged_df['forest fires']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)

merged_df['Year'] = merged_df['Year'] + 2000

df_primary = merged_df  


# Subtract values from the second merged DataFrame from the first merged DataFrame
for col in df_complete_deforestation.columns:
    if col != 'Year':
        df_complete_deforestation[col] = df_complete_deforestation[col] - df_primary[col]
merged_df = df_complete_deforestation





# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='forest fires', data=merged_df, label='forest fires', alpha=1, ax=ax, color=colors[1], zorder=3, bottom=merged_df['logging'])
sns.barplot(x='Year', y='logging', data=merged_df, label='logging', alpha=1, ax=ax, color=colors[0], zorder=2)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))
# Set title and labels
ax.set_xlabel('year')
ax.set_ylabel('forest loss Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = (merged_df['logging'] + merged_df['forest fires']).max()

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(merged_df['Year']))))
ax.set_xticklabels(merged_df['Year'], rotation=90)

# Show the plot
legend = plt.legend(loc='upper left')
legend.get_texts()[0].set_text('logging')
legend.get_texts()[1].set_text('forest fires')

# Calculate the sum of the 'logging' column
total_sum = merged_df['total'].sum()
logging_sum = merged_df['logging'].sum()
fire_sum = merged_df['forest fires'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'secondary forest loss (PA): {round(total_sum/1000000, 2)} Mha \nlogging: {round(logging_sum/1000000, 2)} Mha \nfires: {round(fire_sum/1000000, 2)} Mha'
ax.set_ylim(0,70000)
# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 0.85, s=custom_legend_entry, fontsize=12)
plt.tight_layout()


fig.savefig('results/final_plots/deforestation_protected_areas.png', dpi=120)
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
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/primary_forest_loss_total_yearly.csv')
df2 = pd.read_csv('../code/results/tables/primary_forest_loss_forest_fires_total_yearly.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'forest fires'
df2.rename(columns={'Area (ha)': 'forest fires'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['logging'] = merged_df['total'] - merged_df['forest fires']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)

merged_df['Year'] = merged_df['Year'] + 2000

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='logging', data=merged_df, label='logging', alpha=1, ax=ax, color=colors[0], zorder=2)
sns.barplot(x='Year', y='forest fires', data=merged_df, label='forest fires', alpha=1, ax=ax, color=colors[1], zorder=3, bottom=merged_df['logging'])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))# Set title and labels
ax.set_xlabel('year')
ax.set_ylabel('forest loss Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = (merged_df['logging'] + merged_df['forest fires']).max()
ax.set_ylim(0,1000000)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(merged_df['Year']))))
ax.set_xticklabels(merged_df['Year'], rotation=90)

# Show the plot
legend = plt.legend(loc='upper left')
legend.get_texts()[0].set_text('logging')
legend.get_texts()[1].set_text('forest fires')

# Calculate the sum of the 'logging' column
total_sum = merged_df['total'].sum()
logging_sum = merged_df['logging'].sum()
fire_sum = merged_df['forest fires'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'total forest loss: {round(total_sum/1000000, 2)} Mha \nlogging: {round(logging_sum/1000000, 2)} Mha \nfires: {round(fire_sum/1000000, 2)} Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.25, s=custom_legend_entry, fontsize=12)
plt.tight_layout()

fig.savefig('results/final_plots/total_primary_forest_deforestation.png', dpi=120)
plt.show()



#
#
#
#
#
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df1 = pd.read_csv('../code/results/tables/deforestation_primary_forest_protected_areas.csv')
df2 = pd.read_csv('../code/results/tables/deforestation_primary_forest_no_forest_fires_protected_areas.csv')

# Rename 'Area (ha)' column in df1 to 'total'
df1.rename(columns={'Area (ha)': 'total'}, inplace=True)

# Rename 'Area (ha)' column in df2 to 'primary forest'
df2.rename(columns={'Area (ha)': 'logging'}, inplace=True)

# Perform a left join on 'Year'
merged_df = pd.merge(df1, df2, on='Year', how='left')
# Create a new column 'loss_no_forest_fires'
merged_df['forest fires'] = merged_df['total'] - merged_df['logging']
for col in merged_df.columns:
    if col != 'Year':
        merged_df[col] = merged_df[col].apply(round_to_nearest_100)
merged_df['Year'] = merged_df['Year'] + 2000



# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='logging', data=merged_df, label='logging', alpha=1, ax=ax, color=colors[0], zorder=2)
sns.barplot(x='Year', y='forest fires', data=merged_df, label='forest fires', alpha=1, ax=ax, color=colors[1], zorder=3, bottom=merged_df['logging'])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))# Set title and labels
ax.set_xlabel('year')
ax.set_ylabel('forest loss Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = (merged_df['logging'] + merged_df['forest fires']).max()
ax.set_ylim(0,70000)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(merged_df['Year']))))
ax.set_xticklabels(merged_df['Year'], rotation=90)

# Show the plot
legend = plt.legend(loc='upper left')
legend.get_texts()[1].set_text('forest fires')
legend.get_texts()[0].set_text('logging')


# Calculate the sum of the 'logging' column
total_sum = merged_df['total'].sum()
logging_sum = merged_df['logging'].sum()
fire_sum = merged_df['forest fires'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'primary forest loss (PA): {round(total_sum/1000000, 2)} Mha \nlogging: {round(logging_sum/1000000, 2)} Mha \nfires: {round(fire_sum/1000000, 2)} Mha'
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 0.82, s=custom_legend_entry, fontsize=12)
plt.tight_layout()

fig.savefig('results/final_plots/deforestation_primary_forest_protected_areas.png', dpi=300)
plt.show()

#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_detection_yearly.csv')

# Remove the first row
df = df.drop(df.index[0])

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('new oil palm (Mha)')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'Total new oil palm: {round(total_sum/1000000, 2)} mio Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.tight_layout()

# Remove the legend
plt.legend().remove()

fig.savefig('results/final_plots/new_oil_palm_plantations.png', dpi=300)
plt.show()


#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_after_deforestation.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
df['Year'] = df['Year'] + 2000

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil after deforestation', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0,500000)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'new oil palm after deforestation: {round(total_sum/1000000, 2)} Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.35, s=custom_legend_entry, fontsize=12)

plt.legend().remove()
plt.tight_layout()

fig.savefig('results/final_plots/new_oil_palm_after_deforestation.png', dpi=300)
plt.show()


#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_after_forest_fires.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].astype(int)

df['Year'] = df['Year'] + 2000

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations after forest fires', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('ha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'Total new oil palm after forest fires: {total_sum} ha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.legend().remove()
plt.tight_layout()

fig.savefig('results/final_plots/new_oil_palm_after_forest_fires.png', dpi=300)
plt.show()

#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_protected_areas.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].astype(int)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations in protected areas', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('ha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'total new oil palm in PAs: {total_sum} ha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.legend().remove()
plt.tight_layout()

fig.savefig('results/final_plots/new_oil_palm_protected_areas.png', dpi=300)
plt.show()


#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_nonforest.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


df['Year'] = df['Year'] + 2000
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on nonforest area', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'new oil palm on nonforest area: {round(total_sum/1000000, 2)} mio Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.legend().remove()
plt.tight_layout()

fig.savefig('results/final_plots/new_oil_palm_nonforest.png', dpi=300)
plt.show()


#
#
#
#
#
#


from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_primary.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


df['Year'] = df['Year'] + 2000
# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on primary forest', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'new oil palm on primary forest: {round(total_sum/1000000, 2)} Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.legend().remove()
plt.tight_layout()

fig.savefig('results/final_plots/new_oil_palm_primary.png', dpi=300)
plt.show()

#
#
#
#
colors = [grey, dark_red, red][::-1]

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_detection_yearly.csv')

# Remove the first row
df = df.drop(df.index[0])

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
df['Year'] = df['Year']
df_totalop = df


# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_nonforest.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
df['Year'] = df['Year']
# Increase all indexes by 1
df.index = df.index + 1

df_op_nonforest = df

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/new_oil_palm_primary.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
df['Year'] = df['Year']
df.index = df.index + 1
df_op_primary = df

df_totalop['primary'] = df_op_primary['total']
df_totalop['secondary'] = df_totalop['total'] - df_op_primary['total'] - df_op_nonforest['total']
df_totalop['nonforest'] = df_op_nonforest['total']





fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Calculate the sums
primary_sum = df_totalop['primary'].sum()
secondary_sum = df_totalop['secondary'].sum()
nonforest_sum = df_totalop['nonforest'].sum()

# Plot the data
sns.barplot(x='Year', y='primary', data=df_totalop, label=f'Primary ({round(primary_sum/1000000, 2)} Mha)', alpha=1, ax=ax, color=colors[0], zorder=2, bottom=df_totalop['secondary'] + df_totalop['nonforest'])
sns.barplot(x='Year', y='nonforest', data=df_totalop, label=f'Nonforest ({round(nonforest_sum/1000000, 2)} Mha)', alpha=1, ax=ax, color=colors[2], zorder=2, bottom=df_totalop['secondary'])
sns.barplot(x='Year', y='secondary', data=df_totalop, label=f'Secondary ({round(secondary_sum/1000000, 2)} Mha)', alpha=1, ax=ax, color=colors[1], zorder=2)


ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df_totalop[['primary', 'secondary', 'nonforest']].sum(axis=1).max()
ax.set_ylim(0,500000)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df_totalop['Year']))))
ax.set_xticklabels(df_totalop['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df_totalop[['primary', 'secondary', 'nonforest']].sum().sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'total new oil palm: {round(total_sum/1000000, 2)} Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 0.82, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up

plt.legend()
plt.tight_layout()

fig.savefig('results/final_plots/oilpalm_overview.png', dpi=300)
plt.show()


#
#
#
#
#
df = pd.read_csv('../code/results/tables/forest_loss_new_build_up_areas_yearly.csv')

sum_forest_loss_new_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/forest_loss_build_up_areas_2000_yearly.csv')

sum_forest_loss_existing_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/primary_forest_loss_new_build_up_areas_yearly.csv')

sum_primary_forest_loss_new_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/primary_forest_loss_build_up_areas_2000_yearly.csv')

sum_primary_forest_loss_existing_build_up = df['Area (ha)'].sum()

#
#
#

def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/LAEA_build_up_area.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
# Assuming 'df' is your DataFrame
df.loc[0, 'Year'] = 'build up 2001 - 2020'
df.loc[1, 'Year'] = 'build up 2000'


# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

new_build_up_total = df['total'][0]
existing_build_up_total = df['total'][1]

df = pd.read_csv('../code/results/tables/forest_loss_new_build_up_areas_yearly.csv')

sum_forest_loss_new_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/forest_loss_build_up_areas_2000_yearly.csv')

sum_forest_loss_existing_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/primary_forest_loss_new_build_up_areas_yearly.csv')

sum_primary_forest_loss_new_build_up = df['Area (ha)'].sum()

df = pd.read_csv('../code/results/tables/primary_forest_loss_build_up_areas_2000_yearly.csv')

sum_primary_forest_loss_existing_build_up = df['Area (ha)'].sum()


non_forest_new = new_build_up_total - sum_forest_loss_new_build_up

non_forest_existing = existing_build_up_total - sum_forest_loss_existing_build_up

forest_non_primary_new = sum_forest_loss_new_build_up - sum_primary_forest_loss_new_build_up

forest_non_primary_existing = sum_forest_loss_existing_build_up - sum_primary_forest_loss_existing_build_up


import matplotlib.pyplot as plt
import numpy as np

# Define the figure size for the pie charts
figsize = (6, 5)

# Define the data and labels for the first pie chart
build_up_sums = [non_forest_new, forest_non_primary_new, sum_primary_forest_loss_new_build_up]
labels = ['no forest \n loss', 'other \n forest loss', 'primary \n forest loss']
colors = [dark_red, '#a30202', red]

# Create the first pie chart
fig1, ax1 = plt.subplots(figsize=figsize)
patches, texts, autotexts = plt.pie(build_up_sums, colors=colors, autopct=lambda p: '{:.1f}%\n{:.0f} km²'.format(p, (p / 100) * new_build_up_total / 100))

# Manually add labels to the first pie chart
for i, (patch, label) in enumerate(zip(patches, labels)):
    ang = (patch.theta2 - patch.theta1) / 2. + patch.theta1
    x = 1.3 * patch.r * np.cos(np.deg2rad(ang))
    y = 1.3 * patch.r * np.sin(np.deg2rad(ang))
    plt.text(x, y, label, ha='center', va='center')

# Add a text box to the first pie chart
ax_text1 = fig1.add_axes([0.5, 0.1, 0.1, 0.1])
ax_text1.axis('off')
ax_text1.text(0.2, 0.2, 'New built-up area (2001 - 2020): {:.0f} km²'.format(new_build_up_total / 100), horizontalalignment='center', verticalalignment='center', fontweight='bold', transform=ax_text1.transAxes)

# Save the first figure
fig1.savefig('results/final_plots/pie_new_built_up.png', dpi=120, bbox_inches='tight')

# Create the second pie chart with the same figure size
build_up_sums = [non_forest_existing, forest_non_primary_existing, sum_primary_forest_loss_existing_build_up]
colors = ['#9e9e9e', '#707070', '#454545']

fig2, ax2 = plt.subplots(figsize=figsize)
patches, texts, autotexts = plt.pie(build_up_sums, colors=colors, autopct=lambda p: '{:.1f}%\n{:.0f} km²'.format(p, (p / 100) * existing_build_up_total / 100))

# Manually add labels to the second pie chart
for i, (patch, label) in enumerate(zip(patches, labels)):
    ang = (patch.theta2 - patch.theta1) / 2. + patch.theta1
    x = 1.3 * patch.r * np.cos(np.deg2rad(ang))
    y = 1.3 * patch.r * np.sin(np.deg2rad(ang))
    plt.text(x, y, label, ha='center', va='center')

# Add a text box to the second pie chart
ax_text2 = fig2.add_axes([0.5, 0.1, 0.1, 0.1])
ax_text2.axis('off')
ax_text2.text(0.5, 0.2, 'Built-up area in 2000: {:.0f} km²'.format(existing_build_up_total / 100), horizontalalignment='center', verticalalignment='center', fontweight='bold', transform=ax_text2.transAxes)

# Save the second figure
fig2.savefig('results/final_plots/pie_existing_built_up.png', dpi=120, bbox_inches='tight')

# Show both plots
plt.show()



#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/forest_loss_new_build_up_areas_yearly.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)


df['total'] = df['total'].apply(round_to_nearest_100)



# Create a figure and axis
fig, ax = plt.subplots()

# Bar cMhart for 'total' with a label
bars = ax.bar(df['Year'], df['total'], label='Forest loss on new build up areas', alpha=0.7)

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('hectares')

# Adjust spines and grid
ax.spines['bottom'].set_visible(True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Add total labels for each year
for year, total in zip(df['Year'], df['total']):
    ax.text(year, total + y_max*0.02, f'{round(total)}', ha='center', va='bottom', fontsize=10, rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add the sum of the 'total' column to the plot
ax.text(df['Year'].min(), y_max * 1.2, f'Total forest loss on new build up areas 2001 - 2020: {total_sum} Mha', ha='left', va='center')

# Turn off scientific notation
ax.ticklabel_format(style='plain')

# Set x-axis labels to integers
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Show the plot
plt.legend()
plt.show()

# Save the figure in high resolution
fig.savefig('results/final_plots/forest_loss_new_build_up_areas_yearly.png', dpi=300)

#
#
#
#
#

def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/build_up_area_non_forest.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)
# Assuming 'df' is your DataFrame
df.loc[0, 'Year'] = 'build up 2001 - 2020'
df.loc[1, 'Year'] = 'build up 2000'

# Create a figure and axis
fig, ax = plt.subplots()

# Bar cMhart for 'total' with a label
bars = ax.bar(df['Year'], df['total'], label='build up area on nonforest area', alpha=0.7)

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('hectares')

# Adjust spines and grid
ax.spines['bottom'].set_visible(True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Add total labels for each year
for year, total in zip(df['Year'], df['total']):
    ax.text(year, total + y_max*0.02, f'{round(total)}', ha='center', va='bottom', fontsize=10)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add the sum of the 'total' column to the plot
ax.text(df['Year'].min(), y_max * 1.2, f'Total build up area on nonforest area 2020: {round(total_sum/1000000, 2)} mio Mha', ha='left', va='center')

# Show the plot
plt.legend()
plt.show()

# Save the figure in high resolution
fig.savefig('results/final_plots/build_up_area_non_forest.png', dpi=300)

#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/forest_fires_new_build_up_area_yearly.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


# Create a figure and axis
fig, ax = plt.subplots()

# Bar cMhart for 'total' with a label
bars = ax.bar(df['Year'], df['total'], label='forest fire on new build up area', alpha=0.7)

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('hectares')

# Adjust spines and grid
ax.spines['bottom'].set_visible(True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Add total labels for each year
for year, total in zip(df['Year'], df['total']):
    ax.text(year, total + y_max*0.02, f'{round(total)}', ha='center', va='bottom', fontsize=10, rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add the sum of the 'total' column to the plot
ax.text(df['Year'].min(), y_max * 1.2, f'Total forest fire on new build up area (2001 - 2020): {round(total_sum/1000000, 2)} mio Mha', ha='left', va='center')

# Turn off scientific notation
ax.ticklabel_format(style='plain')

# Set x-axis labels to integers
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Show the plot
plt.legend()
plt.show()

# Save the figure in high resolution
fig.savefig('results/final_plots/forest_fires_new_build_up_area_yearly.png', dpi=300)

#
#
#
#
#
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_buffer(files, title1, title2, output1, output2):
    new_column_names = [100, 200, 500, 1000, 2000]

    df = pd.read_csv(files[0])
    df.columns.values[1] = new_column_names[0]

    for file, column_name in zip(files[1:], new_column_names[1:]):
        other_df = pd.read_csv(file)
        other_df.columns.values[1] = column_name
        df = pd.merge(df, other_df, on='Year', how='left')

    for column in df.columns:
        if column != 'Year':
            df[column] = np.round(df[column], -2).astype(int)

    df.set_index('Year', inplace=True)

    ax = df.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='cividis_r', width=0.75)

    plt.title(title1)
    plt.xlabel('Year')
    plt.ylabel('Area (ha)')

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    legend = ax.legend()
    legend.set_title("Buffer in m")

    # Save the first plot before displaying it
    plt.savefig(output1, dpi=300, bbox_inches='tight')

    df_sum = df.sum().to_frame().T

    ax_sum = df_sum.plot(kind='bar', stacked=False, figsize=(6, 6), colormap='cividis_r', width=0.75)

    plt.title(title2)
    plt.ylabel('Area (ha)')

    formatter = mticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax_sum.yaxis.set_major_formatter(formatter)

    legend = ax_sum.legend()
    legend.set_title("Buffer in m")

    ax_sum.set_xticklabels([])

    # Save the second plot before displaying it
    plt.savefig(output2, dpi=300, bbox_inches='tight')

    # Display both plots after saving
    plt.show()


#
#
#
#
#
#
files = files = ['results/tables/oil_palm_buffer_100_build_up_area.csv',
         'results/tables/oil_palm_buffer_200_build_up_area.csv',
         'results/tables/oil_palm_buffer_500_build_up_area.csv',
         'results/tables/oil_palm_buffer_1000_build_up_area.csv',
         'results/tables/oil_palm_buffer_2000_build_up_area.csv']
title1 = 'Oil palm plantations within different buffer distances from total build up area'
title2 = 'Sum of oil palm plantations within different buffer distances from total build up area'
output1 = 'results/final_plots/buffer_oil_palm_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_oil_palm_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/oil_palm_buffer_100_new_build_up_area.csv',
         'results/tables/oil_palm_buffer_200_new_build_up_area.csv',
         'results/tables/oil_palm_buffer_500_new_build_up_area.csv',
         'results/tables/oil_palm_buffer_1000_new_build_up_area.csv',
         'results/tables/oil_palm_buffer_2000_new_build_up_area.csv']
title1 = 'Oil palm within different buffer distances from new build up area'
title2 = 'Sum of oil palm within different buffer distances new total build up area'
output1 = 'results/final_plots/buffer_oil_palm_new_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_oil_palm_new_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/oil_palm_buffer_100_build_up_area_2000.csv',
         'results/tables/oil_palm_buffer_200_build_up_area_2000.csv',
         'results/tables/oil_palm_buffer_500_build_up_area_2000.csv',
         'results/tables/oil_palm_buffer_1000_build_up_area_2000.csv',
         'results/tables/oil_palm_buffer_2000_build_up_area_2000.csv']
title1 = 'Oil palm within different buffer distances from build up area (2000)'
title2 = 'Sum of oil palm within different buffer distances total build up area (2000)'
output1 = 'results/final_plots/buffer_oil_palm_build_up_area_2000_yearly.png'
output2 = 'results/final_plots/buffer_oil_palm_build_up_area_2000_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
#
#
files = files = ['results/tables/forest_fires_buffer_100_build_up_area.csv',
         'results/tables/forest_fires_buffer_200_build_up_area.csv',
         'results/tables/forest_fires_buffer_500_build_up_area.csv',
         'results/tables/forest_fires_buffer_1000_build_up_area.csv',
         'results/tables/forest_fires_buffer_2000_build_up_area.csv']
title1 = 'forest fires within different buffer distances from total build up area'
title2 = 'Sum of forest fire within different buffer distances from total build up area'
output1 = 'results/final_plots/buffer_forest_fires_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_forest_fires_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/forest_fires_buffer_100_new_build_up_area.csv',
         'results/tables/forest_fires_buffer_200_new_build_up_area.csv',
         'results/tables/forest_fires_buffer_500_new_build_up_area.csv',
         'results/tables/forest_fires_buffer_1000_new_build_up_area.csv',
         'results/tables/forest_fires_buffer_2000_new_build_up_area.csv']
title1 = 'forest fire within different buffer distances from new build up area'
title2 = 'Sum of forest fire within different buffer distances new total build up area'
output1 = 'results/final_plots/buffer_forest_fires_new_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_forest_fires_new_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/forest_fires_buffer_100_build_up_area_2000.csv',
         'results/tables/forest_fires_buffer_200_build_up_area_2000.csv',
         'results/tables/forest_fires_buffer_500_build_up_area_2000.csv',
         'results/tables/forest_fires_buffer_1000_build_up_area_2000.csv',
         'results/tables/forest_fires_buffer_2000_build_up_area_2000.csv']
title1 = 'forest fire within different buffer distances from build up area (2000)'
title2 = 'Sum of forest fire within different buffer distances total build up area (2000)'
output1 = 'results/final_plots/buffer_forest_fires_build_up_area_2000_yearly.png'
output2 = 'results/final_plots/buffer_forest_fires_build_up_area_2000_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
#
#
files = files = ['results/tables/deforestation_no_forest_fires_buffer100_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer200_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer500_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer1000_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer2000_build_up_area.csv']
title1 = 'Deforestation (excluding forest fires) within different buffer distances from total build up area'
title2 = 'Sum of deforestation (excluding forest fires) within different buffer distances from total build up area'
output1 = 'results/final_plots/buffer_no_forest_fires_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_no_forest_fires_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/deforestation_no_forest_fires_buffer100_new_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer200_new_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer500_new_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer1000_new_build_up_area.csv',
         'results/tables/deforestation_no_forest_fires_buffer2000_new_build_up_area.csv']
title1 = 'Deforestation (excluding forest fires) within different buffer distances from new build up area'
title2 = 'Sum of deforestation (excluding forest fires) within different buffer distances new total build up area'
output1 = 'results/final_plots/buffer_no_forest_fires_new_build_up_area_yearly.png'
output2 = 'results/final_plots/buffer_no_forest_fires_new_build_up_area_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
files = files = ['results/tables/deforestation_no_forest_fires_buffer100_build_up_area_2000.csv',
         'results/tables/deforestation_no_forest_fires_buffer200_build_up_area_2000.csv',
         'results/tables/deforestation_no_forest_fires_buffer500_build_up_area_2000.csv',
         'results/tables/deforestation_no_forest_fires_buffer1000_build_up_area_2000.csv',
         'results/tables/deforestation_no_forest_fires_buffer2000_build_up_area_2000.csv']
title1 = 'Deforestation (excluding forest fires) within different buffer distances from build up area (2000)'
title2 = 'Sum of deforestation (excluding forest fires) within different buffer distances total build up area (2000)'
output1 = 'results/final_plots/buffer_no_forest_fires_build_up_area_2000_yearly.png'
output2 = 'results/final_plots/buffer_no_forest_fires_build_up_area_2000_total.png'


plot_buffer(files, title1, title2, output1, output2)
#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/oil_palm_in_RSPO_certified_regions.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on nonforest area', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'new oil palm on RSPO certified: {round(total_sum/1000000, 2)} mio Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=1, y=y_max * 0.9, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up
ax.set_ylim(0,150000)
plt.legend().remove()
plt.tight_layout()


# Save the figure in high resolution
fig.savefig('results/final_plots/oil_palm_in_RSPO_certified_regions.png', dpi=300)
plt.show()
#
#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 100) * 100

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/oil_palm_in_RSPO_uncertified_regions.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on nonforest area', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('Mha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'new oil palm on RSPO uncertified: {round(total_sum/1000000, 2)} mio Mha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 2.2, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up
ax.set_ylim(0,150000)
plt.legend().remove()
plt.tight_layout()


# Save the figure in high resolution
fig.savefig('results/final_plots/oil_palm_in_RSPO_uncertified_regions.png', dpi=300)
plt.show()


#
#
#
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 1) * 1

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/RSPO_primary_forest_loss.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int) + 2000


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on nonforest area', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('ha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'primary forest loss in RSPO certified: {round(total_sum)} ha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.05, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up
ax.set_ylim(0,1250)
plt.legend().remove()
plt.tight_layout()


# Save the figure in high resolution
fig.savefig('results/final_plots/RSPO_certified_primary_loss.png', dpi=300)
plt.show()
#
#
#
#
from matplotlib.ticker import MaxNLocator
def round_to_nearest_100(n):
    return round(n / 1) * 1

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/RSPO_primary_forest_loss_uncertified.csv')

# Rename 'Area (ha)' column to 'total'
df.rename(columns={'Area (ha)': 'total'}, inplace=True)

for col in df.columns:
    if col != 'Year':
        df[col] = df[col].apply(round_to_nearest_100)
df['Year'] = df['Year'].astype(int) + 2000


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 5))  # Adjust the width and height as needed

# Use seaborn's barplot function with the specified colors
sns.barplot(x='Year', y='total', data=df, label='New palm oil plantations on nonforest area', alpha=1, ax=ax, color=colors[0], zorder=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# Set title and labels
ax.set_xlabel('Year')
ax.set_ylabel('ha')

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# Adjust y-axis limit
y_max = df['total'].max()
ax.set_ylim(0, y_max * 1.15)

# Rotate x-axis labels by 90 degrees
ax.xaxis.set_major_locator(FixedLocator(range(len(df['Year']))))
ax.set_xticklabels(df['Year'], rotation=90)

# Calculate the sum of the 'total' column
total_sum = df['total'].sum()

# Add a custom legend entry for the total sum
custom_legend_entry = f'primary forest loss in RSPO uncertified: {round(total_sum)} ha'

# Add text at an arbitrary location of the Axes
ax.text(x=0, y=y_max * 1.04, s=custom_legend_entry, fontsize=12)  # Increase y-value to move text up
ax.set_ylim(0,4500)
plt.legend().remove()
plt.tight_layout()


# Save the figure in high resolution
fig.savefig('results/final_plots/RSPO_uncertified_primary_loss.png', dpi=300)
plt.show()
#
#
#
#
#
#
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# Load the data from the CSV file
df = pd.read_csv('../code/results/tables/croplands.csv')
df['Year'] = [2003, 2007, 2011, 2015, 2019]

df_croplands_deforestation = pd.read_csv('../code/results/tables/croplands_deforestation.csv')
df['deforestation'] = df_croplands_deforestation['Area (ha)']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the width and height as needed

# Use seaborn's lineplot function for 'Area (ha)'
sns.lineplot(x='Year', y='Area (ha)', data=df, ax=ax, color=dark_red, label = 'cropland area', linewidth=2.5, marker='o', markersize = 10)

# Add lineplot for 'deforestation'
sns.lineplot(x='Year', y='deforestation', data=df, ax=ax, color=grey, label = 'previous deforestation', linewidth=2.5, marker='o', markersize = 10)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x / 1e6)))

# Set title and labels for x and y axes
ax.set_xlabel('Year', labelpad= 10, fontweight = 'bold', fontsize = 12)
ax.set_ylabel('Area Mha', fontweight = 'bold')

# Set x-ticks and rotate x-axis labels by 90 degrees
ax.set_xticks(df['Year'])
ax.set_xticklabels(df['Year'], rotation=0)

# Adjust spines and grid
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)
ax.set_ylim(0,340000)

# Show the plot
plt.tight_layout()
plt.show()

fig.savefig('../text/04_literature_review_files/croplands.png', dpi=200)



#
#
#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Daten erstellen
data = {
    'paths': ['Rice', 'Maize', 'Coffee', 'Cassava', 'Banana', 'Groundnut', 'Soybean', 'Vegetabes', 'Other Oil'],
    '2000': [845096, 71826, 69038, 60075, 28827, 22347, 16552, None, None],
    '2005': [693625, 69876, 64004, 40199, 19330, 19258, 6764, 66130, 70825],
    '2010': [487506, 53694, 29116, 11468, None, 2745, 452, 22458, 71045]
}

df = pd.DataFrame(data)

# Daten für das Diagramm vorbereiten
df_melt = df.melt('paths', var_name='Year', value_name='total')

# Diagramm erstellen
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='Year', y='total', hue='paths', data=df_melt)

# Achsenbeschriftungen setzen
ax.set_xlabel('Year')
ax.set_ylabel('ha')

# Gitterlinien anpassen
sns.despine(left=True)
ax.xaxis.grid(False)
ax.yaxis.grid(True)

# y-Achsen-Ticks formatieren
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))

# y-Achsen-Limit anpassen
y_max = df_melt['total'].max()
ax.set_ylim(0,y_max*1.15)

# x-Achsen-Beschriftungen um 90 Grad drehen
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()



#
#
#
#
#
#
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
# Define the colors

grey = '#383838'
red = '#FF0000'
dark_red = '#8B0000'
# Load the data from the CSV file

df = pd.read_csv('../../06_data/vegetation/oil_palm/FAOSTAT_oilpalm.csv')
# Filter the data for 'Area Mharvested' and 'Production'

df_area = df[df['Element'] == 'Area Mharvested']
df_prod = df[df['Element'] == 'Production']
# Merge the two dataframes on 'Year'

df_merged = pd.merge(df_area, df_prod, on='Year', suffixes=('_area', '_prod'))
# Calculate the ratio of 'Production' to 'Area Mharvested'

df_merged['Ratio'] = df_merged['Value_prod'] / df_merged['Value_area']
# Create a figure and axis

fig, ax3 = plt.subplots(figsize=(10, 5))  # Adjust the width and height as needed
# Use seaborn's lineplot function for ratio on the third y-axis

# Use seaborn's lineplot function for ratio on the third y-axis
sns.lineplot(x='Year', y='Ratio', data=df_merged, ax=ax3, color=dark_grey, label = 'yield', legend = False, linewidth=2.5, marker='o', markersize = 10)

# Create a second y-axis tMhat sMhares the same x-axis

ax2 = ax3.twinx()
# Use seaborn's lineplot function with the specified colors for the second y-axis

sns.lineplot(x='Year', y='Value_area', data=df_merged, ax=ax2, color=red, label = 'Area Mharvested', legend = False, linewidth=1 ,marker='o', markersize = 5)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1e6)))
# Create a third subplot for production tMhat sMhares the same x-axis

ax1 = ax3.twinx()
# Use seaborn's lineplot function with the specified colors for the first y-axis

sns.lineplot(x='Year', y='Value_prod', data=df_merged, ax=ax1, color=dark_red, label = 'Production', legend = False, linewidth=0.5, marker='o', markersize = 5)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1e6)))
# Move the spine of the second axis to the right side of the plot

ax2.spines['right'].set_position(('outward', 60))
# Set title and labels for all y-axes

ax3.set_xlabel('Year', labelpad= 10, fontweight = 'bold', fontsize = 12)
ax3.set_ylabel('yield (t/Mha)', color=dark_grey, fontweight = 'bold')
ax2.set_ylabel('Area Harvested (MMha)', color=red)
ax1.set_ylabel('Production (Mt)', color=dark_red)
# Set x-ticks and rotate x-axis labels by 90 degrees

ax3.set_xticks(df_merged['Year'])
ax3.set_xticklabels(df_merged['Year'], rotation=90)
# Set y-axis limits

ax3.set_ylim(0, 30)  # Adjust these values as needed for Ratio
ax2.set_ylim(0, 30000000)  # Adjust these values as needed for Area Harvested
ax1.set_ylim(0, 600000000)  # Adjust these values as needed for Production
# Adjust spines and grid

sns.despine(left=True)
ax3.xaxis.grid(False)
ax3.yaxis.grid(True)
ax2.grid(False)
# Show the plot

fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax3.transAxes)
plt.tight_layout()
plt.show()
fig.savefig('../text/04_literature_review_files/op_yield.png', dpi=200)

#
#
#
#
#
def calculate_sum(files):
    new_column_names = [100, 200, 500, 1000, 2000]

    df = pd.read_csv(files[0])
    df.columns.values[1] = new_column_names[0]

    for file, column_name in zip(files[1:], new_column_names[1:]):
        other_df = pd.read_csv(file)
        other_df.columns.values[1] = column_name
        df = pd.merge(df, other_df, on='Year', how='left')

    for column in df.columns:
        if column != 'Year':
            df[column] = np.round(df[column], -2).astype(int)

    df.set_index('Year', inplace=True)

    df_sum = df.sum().to_frame().T

    # Convert the sum DataFrame to a list
    sum_list = df_sum.values.tolist()[0]

    return sum_list
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
# Store the original font size
original_font_size = plt.rcParams['font.size']

# Increase the size of all texts by 15%
plt.rcParams.update({'font.size': 1.15 * original_font_size})

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

labels = ['100m', '200m', '500m', '1000m', '2000m']

figsize = (6,3)

xlim = 0, 55000000
ylim = 0, 4200000

lighter_colors = ['#f5bd4e', '#f2e079', '#ffbea3', '#ff8471', '#e04053']
darker_colors = ['#a88236', '#a69a53', '#b38672', '#b35d4f', '#942b37']

files1 = [
    'results/tables/oil_palm_buffer_100_build_up_area.csv',
    'results/tables/oil_palm_buffer_200_build_up_area.csv',
    'results/tables/oil_palm_buffer_500_build_up_area.csv',
    'results/tables/oil_palm_buffer_1000_build_up_area.csv',
    'results/tables/oil_palm_buffer_2000_build_up_area.csv'
]

files2 = [
    'results/tables/oil_palm_buffer_100_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_200_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_500_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_1000_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_2000_build_up_area_2000.csv'
]

files3 = [
    'results/tables/oil_palm_buffer_100_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_200_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_500_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_1000_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_2000_new_build_up_area.csv'
]



# Define the corresponding CSV files for widths for each subplot
width_files = [
    'results/tables/buffer_areas_total_built_up.csv',
    'results/tables/buffer_areas_built_up_2000.csv',
    'results/tables/buffer_areas_new_built_up.csv'
    
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of subplots
titles = ['a) existing and new built-up', 'b) year 2000 built-up', 'c) new built-up 2001 - 2020']
# Define labels for the legend
legend_labels = ['100m', '200m', '500m', '1000m', '2000m']  # Replace with your own labels

# Create an empty list to store legend artists for the combined legend
legend_artists = []

# Loop through each subplot and create the plots
for i, (files, title, width_file) in enumerate(zip([files1, files2, files3], titles, width_files)):
    ax = axes[i]

    # Read the CSV file to get the widths for this subplot
    df = pd.read_csv(width_file)
    widths = df.iloc[:, 0].values[::-1].tolist()

    # Your existing code for creating the plot goes here
    values = calculate_sum(files)[::-1]

    bars = []  # List to store the bars
    for i in range(len(files1)):
        bar = ax.bar(0, values[i], width=widths[i], align='edge', bottom=0, color=lighter_colors[i])
        bars.append(bar)

        # Calculate the slope and intercept of the line
        slope = values[i] / widths[i]
        intercept = 0

        # Draw a red line from the bottom left corner to the top right corner of the bar
        # and extend it to the height of the largest bar
        x = [0, max(values) / slope]
        y = [intercept, max(values)]
        ax.plot(x, y, color='grey', linewidth=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('area covered by buffer (Mha)')
    ax.set_ylabel('new oil palm plantations (Mha)')

    # Add horizontal grid lines in light grey
    ax.yaxis.grid(True, color='lightgrey', alpha=0.5)

    # Remove tick lines on the y-axis
    ax.tick_params(axis='y', which='both', length=0)

    # Format y-axis and x-axis to show numbers in millions
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1e6)))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Change color of bottom spine and remove left spine
    ax.spines['bottom'].set_edgecolor('grey')
    ax.spines['left'].set_visible(False)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    if i == 2:  # For the third subplot
        # Create a custom legend in the desired order
        custom_legend = [bars[3], bars[1], bars[0], bars[2], bars[4]]  # Reordered bars

        # Add a legend with custom labels in the desired order
        legend = ax.legend(custom_legend, legend_labels, bbox_to_anchor=(0.9, 0.5), loc='center left')
        legend.set_title("Buffer distances", prop={'weight': 'bold'})

    # Extend the legend artists list for the combined legend
    legend_artists.extend(bars)

legend_artists = [bars[4], bars[3], bars[2], bars[1], bars[0]]

# Create a combined legend for the entire figure with the rearranged legend_labels
legend = fig.legend(legend_artists, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=5)
legend.set_title("Buffer distances", prop={'weight': 'bold'})

# Save the entire figure as a single image
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # adjust the bottom space
fig.savefig('results/final_plots/op_buffer.png', dpi=300)

# Display the combined figure with subplots
plt.show()
plt.rcParams.update({'font.size': original_font_size})

#
#
#
#
#

# Store the original font size
original_font_size = plt.rcParams['font.size']

# Increase the size of all texts by 15%
plt.rcParams.update({'font.size': 1.15 * original_font_size})

labels = ['100m', '200m', '500m', '1000m', '2000m']

figsize = (6, 3)

xlim = 0, 25000000
ylim = 0, 6200000

lighter_colors = ['#f5bd4e', '#f2e079', '#ffbea3', '#ff8471', '#e04053']
darker_colors = ['#a88236', '#a69a53', '#b38672', '#b35d4f', '#942b37']




filesa = ['results/tables/primary_forest_fires_buffer_100_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_200_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_500_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_1000_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_2000_build_up_area.csv'
        ]
filesb = ['results/tables/primary_no_forest_fires_buffer_100_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_200_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_500_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_1000_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_2000_build_up_area.csv'
        ]

values1 = []
for filea, fileb in zip(filesa, filesb):
    dfa = pd.read_csv(filea)
    dfb = pd.read_csv(fileb)
    suma = dfa['Area (ha)'].sum()
    sumb = dfb['Area (ha)'].sum()
    sumab = suma + sumb
    values1.append(sumab)



filesa = ['results/tables/primary_forest_fires_buffer_100_build_up_area_2000.csv',
        'results/tables/primary_forest_fires_buffer_200_build_up_area_2000.csv',
        'results/tables/primary_forest_fires_buffer_500_build_up_area_2000.csv',
        'results/tables/primary_forest_fires_buffer_1000_build_up_area_2000.csv',
        'results/tables/primary_forest_fires_buffer_2000_build_up_area_2000.csv'
        ]
filesb = ['results/tables/primary_no_forest_fires_buffer_100_build_up_area_2000.csv',
        'results/tables/primary_no_forest_fires_buffer_200_build_up_area_2000.csv',
        'results/tables/primary_no_forest_fires_buffer_500_build_up_area_2000.csv',
        'results/tables/primary_no_forest_fires_buffer_1000_build_up_area_2000.csv',
        'results/tables/primary_no_forest_fires_buffer_2000_build_up_area_2000.csv'
        ]

values2 = []
for filea, fileb in zip(filesa, filesb):
    dfa = pd.read_csv(filea)
    dfb = pd.read_csv(fileb)
    suma = dfa['Area (ha)'].sum()
    sumb = dfb['Area (ha)'].sum()
    sumab = suma + sumb
    values2.append(sumab)

filesa = ['results/tables/primary_forest_fires_buffer_100_new_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_200_new_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_500_new_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_1000_new_build_up_area.csv',
        'results/tables/primary_forest_fires_buffer_2000_new_build_up_area.csv'
        ]

filesb = ['results/tables/primary_no_forest_fires_buffer_100_new_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_200_new_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_500_new_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_1000_new_build_up_area.csv',
        'results/tables/primary_no_forest_fires_buffer_2000_new_build_up_area.csv'
        ]

values3 = []
for filea, fileb in zip(filesa, filesb):
    dfa = pd.read_csv(filea)
    dfb = pd.read_csv(fileb)
    suma = dfa['Area (ha)'].sum()
    sumb = dfb['Area (ha)'].sum()
    sumab = suma + sumb
    values3.append(sumab)

values1 = values1[::-1]
values2 = values2[::-1]
values3 = values3[::-1]

# Define the corresponding CSV files for widths for each subplot
width_files = [
    'results/tables/buffer_areas_primary_built_up.csv',
    'results/tables/buffer_areas_primary_built_up_2000.csv',
    'results/tables/buffer_areas_primary_new_built_up.csv'
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of subplots
titles = ['a) existing and new built-up', 'b) year 2000 built-up', 'c) new built-up 2001 - 2020']
# Define labels for the legend
legend_labels = ['100m', '200m', '500m', '1000m', '2000m']  # Replace with your own labels

# Create an empty list to store legend artists for the combined legend
legend_artists = []

# Loop through each subplot and create the plots
for i, (title, width_file) in enumerate(zip(titles, width_files)):
    ax = axes[i]

    # Read the CSV file to get the widths for this subplot
    df = pd.read_csv(width_file)
    widths = df.iloc[:, 0].values[::-1].tolist()

    # Your existing code for creating the plot goes here
    if i == 0:
        values = values1
    elif i == 1:
        values = values2
    else:
        values = values3

    bars = []  # List to store the bars
    for j in range(len(values)):
        bar = ax.bar(0, values[j], width=widths[j], align='edge', bottom=0, color=lighter_colors[j])
        bars.append(bar)

        # Calculate the slope and intercept of the line
        slope = values[j] / widths[j]
        intercept = 0

        # Draw a red line from the bottom left corner to the top right corner of the bar
        # and extend it to the height of the largest bar
        x = [0, max(values) / slope]
        y = [intercept, max(values)]
        ax.plot(x, y, color='grey', linewidth=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('area of primary covered by buffer (Mha)')
    ax.set_ylabel('primary forest loss (Mha)')

    # Add horizontal grid lines in light grey
    ax.yaxis.grid(True, color='lightgrey', alpha=0.5)

    # Remove tick lines on the y-axis
    ax.tick_params(axis='y', which='both', length=0)

    # Format y-axis and x-axis to show numbers in millions
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e6)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1e6)))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Change color of bottom spine and remove left spine
    ax.spines['bottom'].set_edgecolor('grey')
    ax.spines['left'].set_visible(False)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    # Extend the legend artists list for the combined legend
    legend_artists.extend(bars)

# Adjust the layout of subplots to prevent overlapping
plt.tight_layout()

legend_artists = [bars[4], bars[3], bars[2], bars[1], bars[0]]
# Create a combined legend for the entire figure
legend = fig.legend(legend_artists, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=5)
legend.set_title("Buffer distances", prop={'weight': 'bold'})
# Remove the legend from the third plot

plt.subplots_adjust(bottom=0.3)  # adjust the bottom space
# Save the entire figure as a single image
fig.savefig('results/final_plots/deforestation_buffer_primary_forest.png', dpi=300)

# Display the combined figure with subplots
plt.show()

plt.rcParams.update({'font.size': original_font_size})

#
#
#
#
#

import pandas as pd

files1 = [
    'results/tables/oil_palm_buffer_100_build_up_area.csv',
    'results/tables/oil_palm_buffer_200_build_up_area.csv',
    'results/tables/oil_palm_buffer_500_build_up_area.csv',
    'results/tables/oil_palm_buffer_1000_build_up_area.csv',
    'results/tables/oil_palm_buffer_2000_build_up_area.csv'
]

files2 = [
    'results/tables/oil_palm_buffer_100_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_200_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_500_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_1000_build_up_area_2000.csv',
    'results/tables/oil_palm_buffer_2000_build_up_area_2000.csv'
]

files3 = [
    'results/tables/oil_palm_buffer_100_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_200_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_500_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_1000_new_build_up_area.csv',
    'results/tables/oil_palm_buffer_2000_new_build_up_area.csv'
]

width_files = [
    'results/tables/new_oil_palm_detection_yearly.csv'
    
]

def get_first_column_values(files):
    values = []
    for file in files:
        df = pd.read_csv(file)
        values.append(df.iloc[:, 1].sum().tolist())
    return values

values1 = get_first_column_values(files1)
values2 = get_first_column_values(files2)
values3 = get_first_column_values(files3)
width_values = get_first_column_values(width_files)

#
#
#
