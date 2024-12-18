Lung Cancer Analysis and Visualization
This project analyzes and visualizes lung cancer probability in relation to various factors such as age, gender, smoking habits, and yellow fingers, based on data provided in a CSV file.

Features
Data Overview:

Reads and displays an initial snapshot of the dataset.
Focuses on columns of interest: AGE, GENDER, SMOKING, and LUNG_CANCER.
Age Distribution Analysis:

Visualizes the distribution of age using a bar graph.
Smoking and Age Relationship:

Explores the relationship between age and smoking habits using grouped bar charts.
Smoking and Lung Cancer Correlation:

Examines the correlation between smoking status and lung cancer probabilities.
Heatmap Visualizations:

Smoking vs. Age Group: Displays lung cancer probabilities by smoking status and age group.
Yellow Fingers vs. Age Group: Shows lung cancer probabilities by yellow fingers and age group using a heatmap.
Machine Learning Model:

Implements a basic Random Forest Classifier to predict lung cancer based on smoking habits.
Splits data into training and testing sets, trains the model, and evaluates its accuracy.
Technologies Used
Python Libraries:
pandas and numpy for data manipulation and analysis.
matplotlib and seaborn for visualization.
sklearn for machine learning.
