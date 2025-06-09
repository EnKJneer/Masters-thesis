import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter(df, x_col, y_col, x_label, y_label, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_2d_with_color(x_values, y_values, color_values,
                       label='|v_x + v_y|', title = '2D Plot von pos_x und pos_y mit Farbe', dpi=300, xlabel = 'pos_x', ylabel = 'pos_y'):
    """
    Erstellt einen 2D-Plot mit Linien, deren Farbe basierend auf den color_values bestimmt wird.

    :param x_values: Liste oder Array der x-Werte
    :param y_values: Liste oder Array der y-Werte
    :param color_values: Liste oder Array der Werte, die die Farbe bestimmen
    :param label: Name der Farbskala (Standard: '|v_x + v_y|')
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # Erstellen des Plots mit höherer Auflösung
    plt.figure(figsize=(10, 6), dpi=dpi)

    # Normalisieren der color_values für den Farbverlauf
    normalized_color_values = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))

    # Erstellen eines Farbverlaufs basierend auf den color_values
    #for i in range(len(x_values) - 1):
    #    plt.plot(x_values[i:i+2], y_values[i:i+2], c=plt.cm.viridis(normalized_color_values[i]))

    # Erstellen eines Streudiagramms, um die Farbskala anzuzeigen
    sc = plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=1)

    # Hinzufügen einer Farbskala
    plt.colorbar(sc, label=label)

    # Beschriftungen und Titel hinzufügen
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}')
    #plt.savefig(filename + '.png')
    # Anzeigen des Plots
    plt.show()

file_rf = 'Load_Data_DataSets/Results/2025_06_09_09_51_29/Predictions/AL_2007_T4_Plate_Normal_3.csv'
df_rf = pd.read_csv(file_rf)

file_moirai = 'MoE_Moirai/Results/2025_06_04_16_22_51/Predictions/AL_2007_T4_Plate_Normal_3.csv'
df_moirai = pd.read_csv(file_moirai)

file_rnn = 'Recursive_Nets/Results/2025_06_09_10_41_12/Predictions/AL_2007_T4_Plate_Normal_3.csv'
df_rnn = pd.read_csv(file_rnn)

file_naive = 'NaiveModel/Results/2025_06_06_16_59_01/Predictions/AL_2007_T4_Plate_Normal_3.csv'
df_naive = pd.read_csv(file_naive)

print(df_naive.columns)
print(df_moirai.columns)
print(df_rf.columns)
print(df_rnn.columns)


common_columns = list(set(df_rf.columns) & set(df_moirai.columns) & set(df_rnn.columns) & set(df_naive.columns))

# Führen Sie die DataFrames zusammen
df_merged = df_naive.copy()
df_merged = df_merged.merge(df_moirai, on=common_columns, how='left')
df_merged = df_merged.merge(df_rf, on=common_columns, how='left')
df_merged = df_merged.merge(df_rnn, on=common_columns, how='left')

# Remove specified columns
columns_to_remove = ['PK_TrainVal_Neural_Net_y', 'PK_TrainVal_Random_Forest', 'PK_TrainVal_Physical_Model_Single_Axis']
df_merged = df_merged.drop(columns=[col for col in columns_to_remove if col in df_merged.columns])

# Remove rows with NaN values
df_cleaned = df_merged.dropna().reset_index(drop=True)

# Identify columns starting with 'PK_TrainVal' or 'PKL_TrainVal'
pk_columns = [col for col in df_cleaned.columns if col.startswith('PK_TrainVal') or col.startswith('PKL_TrainVal')]

# Dictionary to store deviations
dev_dict = {}

# Plot for each identified column
for col in pk_columns:
    # Calculate deviation
    dev = df_cleaned['curr_x'] - df_cleaned[col]
    dev_dict[col] = dev

"""    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df_cleaned['curr_x'], label='curr_x', color='blue')
    plt.plot(df_cleaned[col], label=col, color='orange')
    plt.plot(dev, label='Deviation', linestyle='--', color='green')
    plt.title(f'Plot of {col} and curr_x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()"""

# Plot all deviations together
plt.figure(figsize=(14, 6))  # Increased figure width
for col, dev in dev_dict.items():
    plt.plot(dev, label=f'Dev {col}')

plt.title('Deviations of all PK columns from curr_x')
plt.xlabel('Index')
plt.ylabel('Deviation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
plt.show()

# Calculate the mean of deviations over time
mean_dev_over_time = pd.DataFrame(dev_dict).mean(axis=1)

# Plot the temporal progression of the mean deviation
plt.figure(figsize=(14, 6))
plt.plot(mean_dev_over_time, label='Mean Deviation Over Time', color='red')

plt.title('Temporal Progression of Mean Deviation')
plt.xlabel('Index')
plt.ylabel('Mean Deviation')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

df_cleaned['mean_dev'] = mean_dev_over_time

plot_2d_with_color(df_cleaned['pos_x'], df_cleaned['pos_y'], df_cleaned['mean_dev'], label='mean loss', title='Mean Deviation')

# Plot temporal progression of 'v_x' and 'mean_dev' with separate axes
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot 'v_x' on the first axis
color = 'tab:blue'
ax1.set_xlabel('Index')
ax1.set_ylabel('v_x', color=color)
ax1.plot(df_cleaned['v_x'], label='v_x', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

# Create a second y-axis with the same x-axis
ax2 = ax1.twinx()

# Plot 'mean_dev' on the second axis
color = 'tab:red'
ax2.set_ylabel('Mean Deviation', color=color)
ax2.plot(df_cleaned['mean_dev'], label='Mean Deviation', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Set the title
plt.title('Temporal Progression of v_x and Mean Deviation')

# Show the plot
fig.tight_layout()
plt.show()

# List of columns to plot against 'mean_dev'
columns_to_plot = ['v_x', 'v_y', 'v_z', 'a_x', 'curr_x', 'curr_y', 'curr_sp', 'f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'materialremoved_sim']

# Filter data where 'v_x' is approximately 0
threshold = 0.01  # Define a threshold for 'v_x' being approximately 0
df_filtered = df_cleaned[abs(df_cleaned['v_x']) < threshold]

# Calculate the correlation matrix for the filtered data
correlation_matrix = df_filtered[columns_to_plot + ['mean_dev']].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for v_x ≈ 0')
plt.show()

# Function to create scatter plots
def plot_scatter(df, x_col, y_col, x_label, y_label, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Create scatter plots for the filtered data
for col in columns_to_plot:
    plot_scatter(df_filtered, col, 'mean_dev', col, 'Mean Deviation', f'{col} vs Mean Deviation (v_x ≈ 0)')
