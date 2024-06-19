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
Initially, a decision tree model was run to assess feature importance, focusing on physical features of the fly while excluding irrelevant attributes such as tempurature and location. After removing one row with a NaN value, 1730 rows of data remained. Also, the data was relatively balanced where the two classes were split 48.53% to 51.47. A correlation matrix revealed weak positive linear relationships between features and species, with a maximum correlation coefficient of 0.22. Interestingly, there was little connection between the features' correlation with species and their importance ranking in the decision tree, which uses information entropy to rank attributes.
