# Data Loading Method Comparison

## Project Overview

This project aims to compare two different data loading methods to determine which one yields better performance in terms of model accuracy and efficiency. The two methods under consideration are:

1. **Individual Data Loading**: Loading data individually for each dataset.
2. **Batched Data Loading**: Loading data in batches and splitting them into training, validation, and testing sets.

The goal is to evaluate the impact of these data loading strategies on the performance of machine learning models, specifically Neural Networks and Random Forests.

## Code Structure

The code is structured into several key components:

### Classes

- **`DataClass_new`**: A class to encapsulate the parameters for loading datasets in the new method (batched data loading).
- **`DataClass`**: A class to encapsulate the parameters for loading datasets in the old method (individual data loading).

### Functions

- **`load_data_new`**: Loads and preprocesses data for training, validation, and testing using the new batched data loading method.
- **`load_data`**: Loads and preprocesses data for training, validation, and testing using the old individual data loading method.
- **`hyperparameter_optimization_ml`**: Performs hyperparameter optimization for the Neural Network model.
- **`hyperparameter_optimization_rf`**: Performs hyperparameter optimization for the Random Forest model.
- **`calculate_mse_and_std`**: Calculates the Mean Squared Error (MSE) and standard deviation for model predictions.

### Data Sets

- **`dataSet_same_material_diff_workpiece_new`**: Dataset for the new method with the same material but different workpieces.
- **`dataSet_diff_material_same_workpiece_new`**: Dataset for the new method with different materials but the same workpiece.
- **`dataSet_diff_material_diff_workpiece_new`**: Dataset for the new method with different materials and different workpieces.
- **`dataSet_same_material_diff_workpiece`**: Dataset for the old method with the same material but different workpieces.
- **`dataSet_diff_material_same_workpiece`**: Dataset for the old method with different materials but the same workpiece.
- **`dataSet_diff_material_diff_workpiece`**: Dataset for the old method with different materials and different workpieces.

### Main Execution

- **Hyperparameter Optimization**: Optimizes hyperparameters for both Neural Network and Random Forest models.
- **Model Training and Evaluation**: Trains and evaluates models using both data loading methods.
- **Results Calculation and Visualization**: Calculates MSE and standard deviation for model predictions and visualizes the results.

## Results

The results of the model comparisons are documented in model_comparsion_results.txt

## Conclusion

The results indicate that the new batched data loading method generally leads to better model performance, as evidenced by lower MSE values and higher percentage improvements across different datasets and models. This suggests that loading data in batches and splitting them into training, validation, and testing sets can be more effective than loading data individually.
