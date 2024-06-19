from models import (
    xgboost, decision_tree, deep_neural_net, xgboost_cv, decision_tree_cv, deep_neural_net_cv,
    deep_torch_neural_net, train_h2o_model, generate_tuples, plot_combined_roc_curves, clean_data, ATTRIBUTES,
    plot_correlation_matrix
)

if __name__ == "__main__":
    # Path to the data file
    DATA_FILE_PATH = "data/83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv"

    # Attributes to plot for the correlation matrix
    attributes_to_plot = [
        "Species", "Replicate", "Sex", "Thorax_length", "l2", "l3p", "l3d", "lpd", "l3", "w1", "w2", "w3", "wing_loading"
    ]

    # Uncomment the following line to plot the correlation matrix
    # plot_correlation_matrix(DATA_FILE_PATH, attributes_to_plot)

    # Run the decision tree model
    decision_tree(DATA_FILE_PATH)

    # Uncomment the following lines to run other models or cross-validation
    # xgboost(DATA_FILE_PATH)
    # deep_neural_net(DATA_FILE_PATH)
    # xgboost_cv(DATA_FILE_PATH)
    # decision_tree_cv(DATA_FILE_PATH)
    # deep_neural_net_cv(DATA_FILE_PATH)
    # deep_torch_neural_net(DATA_FILE_PATH)
    # train_h2o_model(DATA_FILE_PATH)

    # Example of plotting combined ROC curves for multiple models
    # for _ in range(5):
    #     roc_data = [
    #         xgboost_cv(DATA_FILE_PATH),
    #         decision_tree_cv(DATA_FILE_PATH),
    #         deep_neural_net_cv(DATA_FILE_PATH)
    #     ]
    #     plot_combined_roc_curves(roc_data)

    # Example of calculating and printing cross-validation metrics for multiple models
    # xgboost_cv_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    # decision_tree_cv_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    # deep_neural_net_cv_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    #
    # for t in range(100):
    #     print(t)
    #     xgboost_result = xgboost_cv(DATA_FILE_PATH)
    #     xgboost_cv_metrics['auc'].append(xgboost_result['auc'])
    #     xgboost_cv_metrics['accuracy'].append(xgboost_result['accuracy'])
    #     xgboost_cv_metrics['precision'].append(xgboost_result['precision'])
    #     xgboost_cv_metrics['recall'].append(xgboost_result['recall'])
    #     xgboost_cv_metrics['f1'].append(xgboost_result['f1'])
    #
    #     decision_tree_result = decision_tree_cv(DATA_FILE_PATH)
    #     decision_tree_cv_metrics['auc'].append(decision_tree_result['auc'])
    #     decision_tree_cv_metrics['accuracy'].append(decision_tree_result['accuracy'])
    #     decision_tree_cv_metrics['precision'].append(decision_tree_result['precision'])
    #     decision_tree_cv_metrics['recall'].append(decision_tree_result['recall'])
    #     decision_tree_cv_metrics['f1'].append(decision_tree_result['f1'])
    #
    #     deep_neural_net_result = deep_neural_net_cv(DATA_FILE_PATH)
    #     deep_neural_net_cv_metrics['auc'].append(deep_neural_net_result['auc'])
    #     deep_neural_net_cv_metrics['accuracy'].append(deep_neural_net_result['accuracy'])
    #     deep_neural_net_cv_metrics['precision'].append(deep_neural_net_result['precision'])
    #     deep_neural_net_cv_metrics['recall'].append(deep_neural_net_result['recall'])
    #     deep_neural_net_cv_metrics['f1'].append(deep_neural_net_result['f1'])
    #
    # # Calculate means
    # xgboost_cv_means = {key: sum(val) / len(val) for key, val in xgboost_cv_metrics.items()}
    # decision_tree_cv_means = {key: sum(val) / len(val) for key, val in decision_tree_cv_metrics.items()}
    # deep_neural_net_cv_means = {key: sum(val) / len(val) for key, val in deep_neural_net_cv_metrics.items()}
    #
    # print(xgboost_cv_means)
    # print(decision_tree_cv_means)
    # print(deep_neural_net_cv_means)
