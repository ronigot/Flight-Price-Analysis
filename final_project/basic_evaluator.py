class BasicEvaluator:
    """
    Basic evaluator for comparing feature distributions before and after transformation.
    Computes various metrics to evaluate the effect of transformations on feature distributions
    and model performance.

    Attributes:
        stats (dict): Stores the computed metrics (skewness, kurtosis, and normality p-values)
                     for each feature.
        model_performance (dict): Stores metrics comparing model performance before and after
                                transformations.
    """
    def __init__(self):
        """Initialize the BasicEvaluator instance."""
        self.stats = {}  # Distribution metrics
        self.model_performance = {}  # Model performance metrics
        self.transformations = {}  # Transformations applied per feature
        self.fitted_models = {}  # Trained models
        self.predictions = {}  # Model predictions
        self.train_test_data = None  # Train/test split data


    def set_transformations(self, transformations: dict):
        """
        Store the transformations dictionary that the recommender produced.
        """
        self.transformations = transformations


    def compute_metrics(self, original: pd.DataFrame, transformed: pd.DataFrame) -> dict:
        """
        Compute evaluation metrics for each feature, comparing original and transformed data.

        Args:
            original (pd.DataFrame): The original dataset with untransformed features.
            transformed (pd.DataFrame): The dataset with transformed features.

        Returns:
            dict: A dictionary containing the computed metrics for each feature. Each feature has a dictionary
                  with 'original_skew', 'transformed_skew', 'original_kurtosis', 'transformed_kurtosis',
                  'original_normality_p', and 'transformed_normality_p'.
        """
        metrics = {}
        for col in original.columns:
            if col in transformed.columns:
                orig_data = original[col].dropna().values
                trans_data = transformed[col].dropna().values
                try:
                    orig_stat, orig_p = stats.normaltest(orig_data)
                except Exception as e:
                    orig_stat, orig_p = np.nan, np.nan
                try:
                    trans_stat, trans_p = stats.normaltest(trans_data)
                except Exception as e:
                    trans_stat, trans_p = np.nan, np.nan
                metrics[col] = {
                    'original_skew': stats.skew(orig_data),
                    'transformed_skew': stats.skew(trans_data),
                    'original_kurtosis': stats.kurtosis(orig_data),
                    'transformed_kurtosis': stats.kurtosis(trans_data),
                    'original_normality_p': orig_p,
                    'transformed_normality_p': trans_p
                }
        self.stats = metrics
        return metrics


    def evaluate_model_performance(self, X_orig: pd.DataFrame, X_trans: pd.DataFrame,
                             y: pd.Series, model_type: str = 'linear'):
        """
        Evaluate model performance before and after transformations.
        Stores fitted models and predictions for reuse by other visualization functions.

        Args:
            X_orig (pd.DataFrame): Original features DataFrame.
            X_trans (pd.DataFrame): Transformed features DataFrame.
            y (pd.Series): Target variable.
            model_type (str): 'linear' for regression or 'logistic' for classification.
        """
        # Define colors for messages
        BLUE = '\033[94m'
        END = '\033[0m'

        # Split data
        X_orig_train, X_orig_test, X_trans_train, X_trans_test, y_train, y_test = \
            train_test_split(X_orig, X_trans, y, test_size=0.2, random_state=42)

        # Store split data for reuse
        self.train_test_data = {
            'X_orig_train': X_orig_train, 'X_orig_test': X_orig_test,
            'X_trans_train': X_trans_train, 'X_trans_test': X_trans_test,
            'y_train': y_train, 'y_test': y_test,
            'model_type': model_type
        }

        # Select model and metric based on model type
        if model_type == 'linear':
            model_orig = LinearRegression()
            model_trans = LinearRegression()
            metric = mean_squared_error
            metric_name = 'MSE'

            # Message about model type
            print(f"{BLUE}Using Linear Regression model for evaluation.{END}")
        else:
            model_orig = LogisticRegression(max_iter=1000)
            model_trans = LogisticRegression(max_iter=1000)
            metric = accuracy_score
            metric_name = 'Accuracy'

            # Message about model type
            print(f"{BLUE}Using Logistic Regression model for evaluation.{END}")

        # Evaluate original data
        print(f"{BLUE}Training and evaluating model on original data...{END}")
        model_orig.fit(X_orig_train, y_train)
        orig_pred = model_orig.predict(X_orig_test)
        orig_score = metric(y_test, orig_pred)

        # Evaluate transformed data
        print(f"{BLUE}Training and evaluating model on transformed data...{END}")
        model_trans.fit(X_trans_train, y_train)
        trans_pred = model_trans.predict(X_trans_test)
        trans_score = metric(y_test, trans_pred)

        # Store models and predictions for reuse
        self.fitted_models = {
            'original': model_orig,
            'transformed': model_trans
        }

        self.predictions = {
            'original': orig_pred,
            'transformed': trans_pred
        }

        # Calculate improvement percentage
        self.model_performance = {
            'metric_name': metric_name,
            'original_score': orig_score,
            'transformed_score': trans_score,
            'improvement': ((orig_score - trans_score) / orig_score * 100
                          if metric_name == 'MSE'
                          else (trans_score - orig_score) / orig_score * 100)
        }

        print(f"{BLUE}Model evaluation complete.{END}")


    def print_summary(self):
        """
        Print a formatted summary of the evaluation metrics and model performance.
        Values after transformation are colored to indicate change:
        Green: Improvement
        Red: Degradation
        Default: No change
        """
        # Define colors
        GREEN = '\033[92m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        END = '\033[0m'

        print("=== Transformation Evaluation Summary ===\n")

        # Print selected transformations from self.transformations
        if self.transformations:
            # Create transformations table
            trans_data = []
            for feature, trans in self.transformations.items():
                trans_data.append([feature, trans['method']])

            print("Selected Transformations:")
            print(tabulate(trans_data,
                          headers=['Feature', 'Transformation'],
                          tablefmt='pretty',
                          colalign=("left", "left")))
        else:
            print(f"\n{BLUE}{'='*6}\n"
              f"NOTE: No transformation data available.\n"
              f"(Ensure that the recommender has been executed and set_transformations(transformations) has been called.)\n"
              f"{'='*6}{END}")


        # Print statistical metrics if computed
        if self.stats:
            # Create statistical metrics table
            print("\nStatistical Metrics:")
            stats_data = []
            for col, met in self.stats.items():
                # Determine improvement for each metric
                skew_change = abs(met['transformed_skew']) < abs(met['original_skew'])
                kurt_change = (abs(met['transformed_kurtosis']) <
                  abs(met['original_kurtosis']))  # Closer to 0 (normal)
                norm_change = met['transformed_normality_p'] > met['original_normality_p']

                # Color the transformed values based on change
                skew_color = GREEN if skew_change else RED if met['transformed_skew'] != met['original_skew'] else ''
                kurt_color = GREEN if kurt_change else RED if met['transformed_kurtosis'] != met['original_kurtosis'] else ''
                norm_color = GREEN if norm_change else RED if met['transformed_normality_p'] != met['original_normality_p'] else ''

                stats_data.append([
                    col,
                    f"{met['original_skew']:.3f} → {skew_color}{met['transformed_skew']:.3f}{END}",
                    f"{met['original_kurtosis']:.3f} → {kurt_color}{met['transformed_kurtosis']:.3f}{END}",
                    f"{met['original_normality_p']:.3f} → {norm_color}{met['transformed_normality_p']:.3f}{END}"
                ])

            print(tabulate(stats_data,
                          headers=['Feature',
                                  'Skewness (before → after)',
                                  'Kurtosis (before → after)',
                                  'Normality p-value (before → after)'],
                          tablefmt='pretty',
                          colalign=("left", "right", "right", "right")))
            print("\nNote: In large datasets, even minor deviations from normality can lead to very low p-values.\n"
                  "This is a known characteristic of the normaltest, and it doesn't necessarily indicate a severe lack of normality in practical terms.")

            print("\nColor Guide:")
            print(f"{GREEN}Green values{END}:")
            print("- Skewness: Closer to 0 (better symmetry)")
            print("- Kurtosis: Closer to 0 (better tail behavior, using excess kurtosis)")
            print("- Normality p-value: Higher value (stronger evidence of normality)")

            print(f"\n{RED}Red values{END}:")
            print("- Skewness: Further from 0 (worse symmetry)")
            print("- Kurtosis: Further from 0 (worse tail behavior, using excess kurtosis)")
            print("- Normality p-value: Lower value (weaker evidence of normality)")

            print("\nRegular values: No change in the metric")

        else:
            print(f"\n{BLUE}{'='*6}\n"
              f"NOTE: Statistical metrics not computed.\n"
              f"(If desired, call compute_metrics(original, transformed)\n"
              f" before printing the summary.)\n"
              f"{'='*6}{END}")

        # Print model performance if available
        if self.model_performance:
            print("\n=== Model Performance ===")
            improvement = self.model_performance['improvement']
            color = GREEN if improvement > 0 else RED if improvement < 0 else ''
            perf_data = [
                ['Metric', self.model_performance['metric_name']],
                ['Original score', f"{self.model_performance['original_score']:.3f}"],
                ['Transformed score', f"{color}{self.model_performance['transformed_score']:.3f}{END}"],
                ['Improvement', f"{color}{abs(improvement):.1f}%{END}"]
            ]
            print(tabulate(perf_data,
                          tablefmt='pretty',
                          colalign=("left", "right")))

            # Add explanation based on metric type
            if self.model_performance['metric_name']:
                print(f"\nNote: For MSE (Mean Squared Error), {GREEN}lower{END} values indicate better performance")
            else:  # for metrics like accuracy
                print(f"\nNote: For {self.model_performance['metric_name']}, {GREEN}higher{END} values indicate better performance")
        else:
            print(f"\n{BLUE}{'='*6}\n"
              f"NOTE: Model performance not evaluated.\n"
              f"(If desired, call evaluate_model_performance(X_orig, X_trans, y, model_type)\n"
              f" before printing the summary.)\n"
              f"{'='*6}{END}")

        print("\n" + "="*50)


    def plot_feature_distributions(self, original: pd.DataFrame, transformed: pd.DataFrame, feature: str = None):
        """
        Plot the distribution of features before and after transformation.

        Args:
            original (pd.DataFrame): Original dataset
            transformed (pd.DataFrame): Transformed dataset
            feature (str, optional): Feature name to plot. Options:
                - None: plots only features that underwent transformation
                - 'all': plots all features
                - feature name: plots the specific feature
        """

        if not self.transformations:
            print("Transformations not defined. Please call set_transformations() before plotting.")
            return

        if feature == 'all':
            # Plot all features
            features_to_plot = list(original.columns)
        elif feature is not None:
            # Plot specific feature
            if feature not in original.columns:
                raise ValueError(f"Feature '{feature}' not found in the dataset")
            features_to_plot = [feature]
        else:
            # Plot only transformed features
            features_to_plot = [
                feat for feat, trans in self.transformations.items()
                if trans['method'] != 'none'
            ]

        if not features_to_plot:
            print("No features to display based on current settings.")
            return

        # Calculate grid dimensions
        n_features = len(features_to_plot)
        n_cols = 2
        n_rows = n_features

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        fig.suptitle('Feature Distributions Before and After Transformation', y=1.02)

        # Make axes 2D if it's a single feature
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        for idx, feat in enumerate(features_to_plot):
            # Plot original
            sns.histplot(original[feat], kde=True, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'Original - {feat}')

            # Plot transformed
            sns.histplot(transformed[feat], kde=True, ax=axes[idx, 1])
            transformation = self.transformations[feat]['method']
            title = (f'After {transformation} - {feat}\n(No transformation applied)'
                    if transformation == 'none' else f'After {transformation} - {feat}')
            axes[idx, 1].set_title(title)

        plt.tight_layout()
        plt.show()



    def plot_model_evaluation(self):
        """
        Plot comprehensive model evaluation visualizations based on model type.
        Reuses fitted models and predictions from evaluate_model_performance if available.
        """
        # Define colors for messages
        BLUE = '\033[94m'
        END = '\033[0m'

        # Check if we have stored models and data
        if not hasattr(self, 'fitted_models') or not self.fitted_models or not hasattr(self, 'train_test_data') or not self.train_test_data:
            print(f"{BLUE}{'='*6}\n"
                  f"NOTE: No trained models or evaluation data available.\n"
                  f"Please call evaluate_model_performance(X_orig, X_trans, y, model_type) first\n"
                  f"before creating model evaluation plots.\n"
                  f"{'='*6}{END}")
            return

        # Use stored data
        y_test = self.train_test_data['y_test']
        model_type = self.train_test_data['model_type']

        if model_type == 'linear':
            # Get stored predictions and models
            pred_orig = self.predictions['original']
            pred_trans = self.predictions['transformed']

            # Calculate residuals
            residuals_orig = y_test - pred_orig
            residuals_trans = y_test - pred_trans

            # Calculate MSE
            mse_orig = mean_squared_error(y_test, pred_orig)
            mse_trans = mean_squared_error(y_test, pred_trans)

            # Create a 2x2 grid of plots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Row 1: Predicted vs Actual plots
            # Original data
            axs[0, 0].scatter(y_test, pred_orig, alpha=0.5, color='purple')
            axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axs[0, 0].set_xlabel("Actual Values")
            axs[0, 0].set_ylabel("Predicted Values")
            axs[0, 0].set_title(f"Original Data: Predicted vs Actual\nMSE={mse_orig:.3f}")

            # Transformed data
            axs[0, 1].scatter(y_test, pred_trans, alpha=0.5, color='green')
            axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axs[0, 1].set_xlabel("Actual Values")
            axs[0, 1].set_ylabel("Predicted Values")
            axs[0, 1].set_title(f"Transformed Data: Predicted vs Actual\nMSE={mse_trans:.3f}")

            # Row 2: Residual plots
            # Original data
            axs[1, 0].scatter(pred_orig, residuals_orig, alpha=0.5, color='purple')
            axs[1, 0].axhline(0, color='red', linestyle='--')
            axs[1, 0].set_xlabel('Predicted Values')
            axs[1, 0].set_ylabel('Residuals')
            axs[1, 0].set_title(f'Original Data: Residual Plot')

            # Transformed data
            axs[1, 1].scatter(pred_trans, residuals_trans, alpha=0.5, color='green')
            axs[1, 1].axhline(0, color='red', linestyle='--')
            axs[1, 1].set_xlabel('Predicted Values')
            axs[1, 1].set_ylabel('Residuals')
            axs[1, 1].set_title(f'Transformed Data: Residual Plot')

            plt.suptitle('Linear Regression Model Evaluation', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

            # Print improvement summary
            improvement = ((mse_orig - mse_trans) / mse_orig * 100)
            print(f"MSE Improvement: {improvement:.1f}%")

        else:  # Logistic regression
            # Get stored test data
            X_orig_test = self.train_test_data['X_orig_test']
            X_trans_test = self.train_test_data['X_trans_test']

            # Get stored models
            model_orig = self.fitted_models['original']
            model_trans = self.fitted_models['transformed']

            # Get probabilities (not stored earlier since we only stored predictions)
            prob_orig = model_orig.predict_proba(X_orig_test)[:, 1]
            prob_trans = model_trans.predict_proba(X_trans_test)[:, 1]

            # Get stored predictions or make them if not available
            pred_orig = self.predictions['original']
            pred_trans = self.predictions['transformed']

            # Calculate metrics
            acc_orig = accuracy_score(y_test, pred_orig)
            acc_trans = accuracy_score(y_test, pred_trans)
            f1_orig = f1_score(y_test, pred_orig, average='weighted')
            f1_trans = f1_score(y_test, pred_trans, average='weighted')

            # ROC curves
            fpr_orig, tpr_orig, _ = roc_curve(y_test, prob_orig)
            fpr_trans, tpr_trans, _ = roc_curve(y_test, prob_trans)
            auc_orig = auc(fpr_orig, tpr_orig)
            auc_trans = auc(fpr_trans, tpr_trans)

            # Calculate calibration curves
            fraction_of_positives_orig, mean_predicted_value_orig = calibration_curve(y_test, prob_orig, n_bins=10)
            fraction_of_positives_trans, mean_predicted_value_trans = calibration_curve(y_test, prob_trans, n_bins=10)

            # Create a 2x2 grid of plots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Top row: Calibration plots
            # Original data
            axs[0, 0].plot(mean_predicted_value_orig, fraction_of_positives_orig, "s-", label="Calibration curve")
            axs[0, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axs[0, 0].set_xlabel("Mean Predicted Probability")
            axs[0, 0].set_ylabel("Fraction of Positives")
            axs[0, 0].set_title(f"Original Data: Calibration Curve\nAccuracy={acc_orig:.3f}, F1={f1_orig:.3f}")
            axs[0, 0].legend(loc="lower right")

            # Transformed data
            axs[0, 1].plot(mean_predicted_value_trans, fraction_of_positives_trans, "s-", label="Calibration curve")
            axs[0, 1].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axs[0, 1].set_xlabel("Mean Predicted Probability")
            axs[0, 1].set_ylabel("Fraction of Positives")
            axs[0, 1].set_title(f"Transformed Data: Calibration Curve\nAccuracy={acc_trans:.3f}, F1={f1_trans:.3f}")
            axs[0, 1].legend(loc="lower right")

            # Bottom row: ROC curves
            # Original data
            axs[1, 0].plot(fpr_orig, tpr_orig, label=f'Original (AUC = {auc_orig:.3f})')
            axs[1, 0].plot([0, 1], [0, 1], 'k--')
            axs[1, 0].set_xlabel('False Positive Rate')
            axs[1, 0].set_ylabel('True Positive Rate')
            axs[1, 0].set_title('Original Data: ROC Curve')
            axs[1, 0].legend(loc="lower right")

            # Transformed data
            axs[1, 1].plot(fpr_trans, tpr_trans, label=f'Transformed (AUC = {auc_trans:.3f})')
            axs[1, 1].plot([0, 1], [0, 1], 'k--')
            axs[1, 1].set_xlabel('False Positive Rate')
            axs[1, 1].set_ylabel('True Positive Rate')
            axs[1, 1].set_title('Transformed Data: ROC Curve')
            axs[1, 1].legend(loc="lower right")

            plt.suptitle('Logistic Regression Model Evaluation', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

            # Print improvement summary
            acc_improvement = ((acc_trans - acc_orig) / acc_orig * 100)
            f1_improvement = ((f1_trans - f1_orig) / f1_orig * 100)
            auc_improvement = ((auc_trans - auc_orig) / auc_orig * 100)
            print(f"Accuracy Improvement: {acc_improvement:.1f}%")
            print(f"F1 Score Improvement: {f1_improvement:.1f}%")
            print(f"AUC Improvement: {auc_improvement:.1f}%")



    def plot_correlation_heatmap(self, X_orig, X_trans, annot=True, highlight_changes=True, change_threshold=0.1):
        """
        Plot side-by-side correlation heatmaps with numerical annotations.
        Highlights significant changes in correlations between original and transformed data.

        Args:
            X_orig (pd.DataFrame): Original features DataFrame
            X_trans (pd.DataFrame): Transformed features DataFrame
            annot (bool): Whether to display correlation values on heatmap
            highlight_changes (bool): Whether to highlight significant correlation changes
            change_threshold (float): Threshold for considering a correlation change significant
        """
        # Select only numeric columns
        numeric_cols_orig = X_orig.select_dtypes(include=[np.number]).columns
        numeric_cols_trans = X_trans.select_dtypes(include=[np.number]).columns

        # Ensure the same columns are used for both matrices
        common_cols = list(set(numeric_cols_orig).intersection(set(numeric_cols_trans)))

        if not common_cols:
            print("No common numeric columns found between original and transformed data.")
            return

        # Calculate correlation matrices
        corr_orig = X_orig[common_cols].corr().round(2)
        corr_trans = X_trans[common_cols].corr().round(2)

        # Create masks for upper triangles
        mask = np.triu(np.ones_like(corr_orig, dtype=bool))

        # Create figure and axes
        fig, axs = plt.subplots(1, 3 if highlight_changes else 2,
                                figsize=(18 if highlight_changes else 14, 7))

        # Plot original correlations
        sns.heatmap(corr_orig, ax=axs[0], cmap='coolwarm',
                    annot=annot, fmt='.2f', square=True, mask=mask,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        axs[0].set_title("Original Correlations")

        # Plot transformed correlations
        sns.heatmap(corr_trans, ax=axs[1], cmap='coolwarm',
                    annot=annot, fmt='.2f', square=True, mask=mask,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        axs[1].set_title("Transformed Correlations")

        # Optionally plot correlation changes
        if highlight_changes:
            # Calculate absolute differences
            corr_diff = (corr_trans - corr_orig).abs()

            # Create a colormap that highlights significant changes
            cmap = sns.diverging_palette(10, 133, as_cmap=True)

            # Highlight significant changes
            sns.heatmap(corr_diff, ax=axs[2], cmap=cmap,
                      annot=annot, fmt='.2f', square=True, mask=mask,
                      linewidths=0.5, cbar_kws={"shrink": 0.8},
                      vmin=0, vmax=max(change_threshold*2, corr_diff.max().max()))
            axs[2].set_title(f"Absolute Correlation Changes\n(threshold={change_threshold})")

            # Add red squares around significant changes
            if annot:
                for i in range(len(common_cols)):
                    for j in range(len(common_cols)):
                        if i > j and corr_diff.iloc[i, j] >= change_threshold:
                            axs[2].add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                        edgecolor='black', lw=2))

        plt.suptitle("Correlation Analysis: Before vs. After Transformation", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # Print summary of the most significant correlation changes
        if highlight_changes:
            significant_changes = []
            for i in range(len(common_cols)):
                for j in range(i+1, len(common_cols)):
                    change = abs(corr_trans.iloc[i, j] - corr_orig.iloc[i, j])
                    if change >= change_threshold:
                        significant_changes.append((
                            common_cols[i],
                            common_cols[j],
                            corr_orig.iloc[i, j],
                            corr_trans.iloc[i, j],
                            change
                        ))

            if significant_changes:
                # Sort by change magnitude
                significant_changes.sort(key=lambda x: x[4], reverse=True)

                print("\nSignificant correlation changes (|Δr| ≥ {:.2f}):".format(change_threshold))
                print("{:<15} {:<15} {:<15} {:<15} {:<15}".format(
                    "Feature 1", "Feature 2", "Original r", "Transformed r", "Change |Δr|"))
                print("-" * 75)

                for feat1, feat2, orig, trans, change in significant_changes:
                    print("{:<15} {:<15} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                        feat1, feat2, orig, trans, change))
            else:
                print("\nNo significant correlation changes detected (threshold = {:.2f})".format(
                    change_threshold))