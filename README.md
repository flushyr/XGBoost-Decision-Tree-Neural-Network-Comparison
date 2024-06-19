## Comparative Analysis of XGBoost, Decision Tree, and Neural Network Models for Binary Classification
My report for COMP4702: Machine Learning, A comparison between XGBoost, Decision Trees, and Neural Networks for a binary classification task.

### Background
I found machine learning to be such an interesting course, so I've decided I'd like to focus my career around it. I am extremely proud of this assignment since I put a lot of thought into coding, tuning, training, then repeating again. I even mixed in some statistical analysis in the form of p-scores in the final comparison.

We were given a tabular dataset from a scientific research article discussing the evolutionary biology of the fruit fly, Drosophila melanogaster. This dataset includes measurements of thorax length (millimetres), wing to thorax ratio (wing loading), and various wing dimensions (l2, l3p, lpd, l3, w1, w2, w3) from five laboratory populations of Drosophila buzzatii and Drosophila aldrichi. The data was collected in 1994 after these populations were maintained in the laboratory for five generations at a constant temperature of 25°C. Measurements were taken of the fifth generation’s offspring, raised under three different temperature treatments (20°C, 25°C, and 30°C).

I decided to set the aim of the model to predict whether each fly is classified as Drosophila aldrichi or Drosophila buzzatii.

In the following sections, I will summarise my literature review, data preparation, model training, results, comparison, and evaluation.

### Literature Review
Despite the lack of direct comparisons between Decision Trees, XGBoost, and Neural Networks in published research, there is a general consensus on their qualitative aspects like versatility, scalability, and performance. XGBoost is an improvement over decision trees in terms of performance but sacrifices comprehensibility, making decision trees preferable for interpretability (Wohlwend, 2023). However, XGBoost struggles with imbalanced data and misclassification costs (Zhang et al., 2022). Neural networks, while complex and prone to overfitting, are not well-suited for tabular data but excel in visual analysis (Donges, 2020). Overall, XGBoost is expected to outperform the other models for structured data, with tree-based models generally outperforming neural network models (Grinsztajn et al., 2022).

### Data Preparation and Exploration
Initially, a decision tree model was run to assess feature importance, focusing on physical features of the fly. After dropping one row with a NaN value, 1730 rows of data remained. Also, the data was relatively balanced where the two classes were split 48.53% to 51.47%. A correlation matrix revealed weak positive linear relationships between features and species, with a maximum coefficient of 0.22. Interestingly, there was little connection between the features' correlation with species and their importance ranking in the decision tree, since the latter uses information entropy to rank attributes.

<figure style="text-align: center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Correlation%20Matrix%20-%20Dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Correlation%20Matrix.png">
    <img alt="Correlation Matrix" src="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Correlation%20Matrix.png" width="600">
  </picture>
  <figcaption style="text-align:center;">Correlation Matrix</figcaption>
</figure>

<figure style="text-align: center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Decision%20Tree%20Importance%20-%20Dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Decision%20Tree%20Importance.png">
    <img alt="Decision Tree Importance" src="https://github.com/flushyr/XGBoost-Decision-Tree-Neural-Network-Comparison/blob/main/Figures/Features'%20Decision%20Tree%20Importance.png" width="600">
  </picture>
  <figcaption style="text-align:center;">Decision Tree Importance</figcaption>
</figure>

## Model Training
All models in the comparative analysis were trained on the same dataset, ensuring a fair evaluation of their performance. To mitigate overfitting risks, five-fold cross-validation was employed for all models. Hyperparameter tuning was conducted using Grid Search for XGBoost and Decision Trees due to their limited hyperparameters, while a random search was employed for the Neural Network's extensive parameter space. To address variability in optimal hyperparameters, training, tuning, and evaluation were repeated ten times, with the final hyperparameter chosen by majority vote.

#### XGBoost Model
The XGBoost model was optimized by tuning essential hyperparameters like learning rate, maximum delta step, subsample ratio, and number of parallel trees, which significantly improved its performance and mitigated overfitting risks for binary classification tasks.

#### Decision Tree Model
The Decision Tree model's performance was enhanced by adjusting critical parameters such as split criterion, maximum tree depth, and minimum sample requirements, leading to improved accuracy, reduced overfitting, and better generalization capabilities.

#### Neural Network Model
The Neural Network model's effectiveness was boosted through tuning hidden layer sizes, activation function, optimiser, learning rate schedule, and early stopping settings, resulting in improved capacity to capture complex patterns, generalize effectively, and adapt to unseen data.  


## Comparison and Discussion
I compared the performance of XGBoost, Decision Tree, and Neural Network models using accuracy, precision, recall, F1 score, and area under the ROC curve (AUC). XGBoost generally outperforms the other models across these metrics, with statistically significant differences in most cases, although it performs similarly to the Neural Network in terms of recall. All in all, while XGBoost was better than Neural Networks overall, the similar scores suggests that model performance may be limited by the dataset's predictive power and the similarity of characteristics between the two fly species. Below is a table showing 

