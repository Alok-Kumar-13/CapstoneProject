
# Auto MPG Prediction and Drift Monitoring

This repository contains a machine learning project to predict a car's Miles Per Gallon (MPG), including model training, a prediction, and data drift analysis.

## Project Structure

* **`Capstone_project_auto_mpg.ipynb`**: The main notebook for data exploration, model training, and generating `autompg_model.bin`.

* **`Capstone_project_data_monitoring.ipynb`**: A notebook for analyzing and visualizing data drift, generating reports and plots.

* **`predict.py`**: A script that loads `autompg_model.bin` to serve MPG predictions.

* **`requirements.txt`**: Lists all required Python libraries.

* **`auto_mpg.csv`**: The raw dataset used for the project.

### Generated Artifacts

These files are outputs from the project's notebooks and scripts:

* `autompg_model.bin`: The pickled binary file of the final model and its `StandardScaler`.

* `ml_flow_model_comparison.png` & `model_version.png`: Visualizations of model performance and versioning.

* `model_performance_report_*.html` & `data_drift_report_*.html`: HTML reports detailing model performance and data drift.

* `Data_drift.png`: A plot visualizing the data drift.

### CI/CD Pipeline

* `.github/workflows/ci-cd.yml`: Defines the GitHub Actions workflow for continuous integration and deployment.

