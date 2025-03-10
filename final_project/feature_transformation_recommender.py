class FeatureTransformationRecommender:
    """
    An automated feature transformation recommender for tabular data.

    When a target variable is provided, each numeric feature is evaluated individually using
    an inner cross-validation loop. For each feature, candidate transformations are generated:
      - 'none' (no change),
      - 'log' (log1p, if all values > 0),
      - 'sqrt' (square root, if all values >= 0),
      - 'yeo-johnson' (power transformation),
      - 'quantile' (quantile transformation with a normal output distribution),
      - 'boxcox' (Box-Cox transformation, if all values > 0),
      - 'standardization' (Z-score normalization),
      - 'minmax' (min-max scaling to [0,1]),
      - 'polynomial' (polynomial features, degree 2),
      - 'binning' (quantile-based discretization).

    For each candidate, a simple model (LinearRegression for regression or LogisticRegression
    for classification) is trained on the dataset where only that feature is replaced by the candidate.
    The candidate with the best CV performance (if improved by at least a relative threshold) is selected.

    If no target is provided, the system falls back to a combined metric based on distribution normality.
    """
    def __init__(self, model_type='linear', min_improvement=0.02, cv_folds=3):
        """
        Initialize the recommender with model type, improvement threshold, and CV folds.

        Args:
            model_type (str): 'linear' for regression or 'logistic' for classification.
            min_improvement (float): Minimum relative improvement in CV score required to adopt a candidate.
            cv_folds (int): Number of folds for the inner cross-validation evaluation.
        """
        self.model_type = model_type
        self.min_improvement = min_improvement
        self.cv_folds = cv_folds
        self.transformations = {}  # Dictionary to store the best transformation info for each feature

    def evaluate_distribution(self, data):
        """
        Evaluate the distribution of the input data using a normality test.

        If the normality test is successful, the p-value is returned.
        If an error occurs, a fallback metric based on the skewness of the data is returned.

        Args:
            data (array-like): The input data to evaluate, which should be a 1D array or Series.

        Returns:
            float: The p-value from the normality test, or a fallback metric based on skewness if the test fails.
        """
        try:
            stat, p_value = stats.normaltest(data)
            return p_value
        except Exception as e:
            # In case of an error, return a fallback metric based on the skewness of the data
            return 1.0 / (1 + abs(stats.skew(data)))

    def _model_and_scoring(self):
        """
        Determine the model and scoring metric based on the type of problem.

        Returns:
            tuple: A tuple containing:
                - model: the regression or classification model.
                - scoring: the scoring metric for cross-validation.
        """
        if self.model_type == 'linear':
            return LinearRegression(), 'neg_mean_squared_error'  # For regression: lower MSE is better (negated)
        else:
            return LogisticRegression(max_iter=1000), 'accuracy'  # For classification: higher accuracy is better

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the transformation recommender to the input dataset.

        If a target variable (y) is provided, each numeric feature is evaluated by comparing the performance
        of models trained on various candidate transformations against the baseline (raw feature).
        The transformation yielding a significant improvement in model performance is selected.

        If no target variable (y) is provided, a fallback method based on distribution normality is used
        to select the best transformation for each feature.

        Args:
            X (pd.DataFrame): The input dataset, where each column represents a feature to be transformed.
            y (pd.Series, optional): The target variable, provided for supervised learning. Defaults to None.

        Returns:
            self: The fitted recommender object with the selected transformations.
        """
        if y is not None:
            # Using performance-based evaluation (cross-validation)
            model, scoring = self._model_and_scoring()
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number):
                    # Process numeric features:
                    # Fill missing values with the column mean and convert to numpy array
                    raw_data = X[col].fillna(X[col].mean()).values

                    # Compute the baseline performance using the raw feature
                    baseline_score = np.mean(cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring))
                    best_method = 'none'
                    best_score = baseline_score
                    best_transformer = None
                    candidates = []

                    # Candidate: log transformation (if all values are > 0)
                    if np.all(raw_data > 0):
                        try:
                            candidate_log = np.log1p(raw_data)
                            candidates.append(('log', candidate_log, None))
                        except Exception as e:
                            pass  # If an error occurs, skip this candidate

                    # Candidate: sqrt transformation (if all values are >= 0)
                    if np.all(raw_data >= 0):
                        try:
                            candidate_sqrt = np.sqrt(raw_data)
                            candidates.append(('sqrt', candidate_sqrt, None))
                        except Exception as e:
                            pass

                    # Candidate: yeo-johnson transformation using PowerTransformer
                    try:
                        pt = PowerTransformer(method='yeo-johnson')
                        candidate_yj = pt.fit_transform(raw_data.reshape(-1, 1)).ravel()
                        candidates.append(('yeo-johnson', candidate_yj, pt))
                    except Exception as e:
                        pass

                    # Candidate: quantile transformation (to achieve a normal output distribution)
                    try:
                        qt = QuantileTransformer(output_distribution='normal')
                        candidate_qt = qt.fit_transform(raw_data.reshape(-1, 1)).ravel()
                        candidates.append(('quantile', candidate_qt, qt))
                    except Exception as e:
                        pass

                    # Candidate: box-cox transformation (only applicable if all values > 0)
                    if np.all(raw_data > 0):
                        try:
                            boxcox_data, lambda_param = stats.boxcox(raw_data)
                            candidates.append(('boxcox', boxcox_data, lambda_param))
                        except Exception as e:
                            pass

                    # Candidate: standardization (Z-score normalization)
                    scaler = StandardScaler()
                    candidates.append(('standardization', scaler.fit_transform(raw_data.reshape(-1, 1)).ravel(), scaler))

                    # Candidate: min-max scaling (scales data to the range [0, 1])
                    minmax = MinMaxScaler()
                    candidates.append(('minmax', minmax.fit_transform(raw_data.reshape(-1, 1)).ravel(), minmax))

                    # Candidate: polynomial features (degree 2, capturing the quadratic term)
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    poly_features = poly.fit_transform(raw_data.reshape(-1, 1))[:, 1]  # Only quadratic term is kept
                    candidates.append(('polynomial', poly_features, poly))

                    # Candidate: binning (discretizes the continuous variable into quantile-based bins)
                    try:
                        binned_data = pd.qcut(raw_data, q=5, labels=False)
                        candidates.append(('binning', binned_data, None))
                    except Exception as e:
                        pass

                    # Evaluate each candidate transformation using cross-validation
                    for method, candidate_data, transformer in candidates:
                        X_temp = X.copy()  # Work on a copy to avoid modifying the original dataset
                        # Replace the current column with the candidate transformed data
                        X_temp[col] = candidate_data
                        candidate_score = np.mean(cross_val_score(model, X_temp, y, cv=self.cv_folds, scoring=scoring))
                        improvement = candidate_score - baseline_score
                        # Select candidate if it improves performance sufficiently and is better than previous candidates
                        if improvement >= self.min_improvement * abs(baseline_score) and candidate_score > best_score:
                            best_score = candidate_score
                            best_method = method
                            best_transformer = transformer
                    # Store the best transformation info for the current feature
                    self.transformations[col] = {'method': best_method, 'transformer': best_transformer}
                else:
                    # For non-numeric features, choose encoding:
                    # Use one-hot encoding if there are more than 2 unique values; otherwise, use label encoding
                    if X[col].nunique() > 2:
                        self.transformations[col] = {'method': 'one_hot', 'transformer': None}
                    else:
                        self.transformations[col] = {'method': 'label', 'transformer': None}
        else:
            # Fallback method: use distribution-based metrics when no target variable is provided
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number):
                    original = X[col].fillna(X[col].mean()).values
                    original_normality = self.evaluate_distribution(original)
                    baseline_metric = original_normality
                    candidates = []
                    candidates.append(('none', original, None))
                    # Candidate: log transformation
                    if np.all(original > 0):
                        try:
                            log_transformed = np.log1p(original)
                            candidates.append(('log', log_transformed, None))
                        except Exception as e:
                            pass

                    # Candidate: sqrt transformation
                    if np.all(original >= 0):
                        try:
                            sqrt_transformed = np.sqrt(original)
                            candidates.append(('sqrt', sqrt_transformed, None))
                        except Exception as e:
                            pass

                    # Candidate: yeo-johnson transformation
                    try:
                        pt = PowerTransformer(method='yeo-johnson')
                        yj_transformed = pt.fit_transform(original.reshape(-1,1)).ravel()
                        candidates.append(('yeo-johnson', yj_transformed, pt))
                    except Exception as e:
                        pass

                    # Candidate: quantile transformation
                    try:
                        qt = QuantileTransformer(output_distribution='normal')
                        candidate_qt = qt.fit_transform(original.reshape(-1, 1)).ravel()
                        candidates.append(('quantile', candidate_qt, qt))
                    except Exception as e:
                        pass

                    # Candidate: box-cox transformation
                    if np.all(original > 0):
                        try:
                            boxcox_data, lambda_param = stats.boxcox(original)
                            candidates.append(('boxcox', boxcox_data, lambda_param))
                        except Exception as e:
                            pass

                    # Candidate: standardization (Z-score normalization)
                    scaler = StandardScaler()
                    candidates.append(('standardization', scaler.fit_transform(original.reshape(-1, 1)).ravel(), scaler))

                    # Candidate: min-max scaling
                    minmax = MinMaxScaler()
                    candidates.append(('minmax', minmax.fit_transform(original.reshape(-1, 1)).ravel(), minmax))

                    # Candidate: polynomial transformation (degree 2)
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    poly_features = poly.fit_transform(original.reshape(-1, 1))[:, 1]  # Keep quadratic term only
                    candidates.append(('polynomial', poly_features, poly))

                    # Candidate: binning (discretization into quantiles)
                    try:
                        binned_data = pd.qcut(original, q=5, labels=False)
                        candidates.append(('binning', binned_data, None))
                    except Exception as e:
                        pass

                    best_method = 'none'
                    best_transformer = None
                    best_metric = baseline_metric
                    # Evaluate each candidate based on the normality metric
                    for method, transformed, transformer in candidates:
                        candidate_metric = self.evaluate_distribution(transformed)
                        if candidate_metric > best_metric + self.min_improvement:
                            best_metric = candidate_metric
                            best_method = method
                            best_transformer = transformer
                    # Store the selected transformation for the current feature
                    self.transformations[col] = {'method': best_method, 'transformer': best_transformer}
                else:
                    # For categorical features, select the encoding method as before
                    if X[col].nunique() > 2:
                        self.transformations[col] = {'method': 'one_hot', 'transformer': None}
                    else:
                        self.transformations[col] = {'method': 'label', 'transformer': None}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the selected transformations to the dataset X.
        For each feature, the corresponding transformation is applied as determined during fitting.

        This method handles both numeric and categorical features:
            - For numeric features, the transformation specified during fitting is applied (e.g., log, sqrt, etc.).
            - For categorical features, either one-hot encoding or label encoding is applied, based on the chosen method.

        Args:
            X (pd.DataFrame): The input dataset to transform. It should have the same columns as during the fitting process.

        Returns:
            pd.DataFrame: The transformed dataset with the same index as the original input, but with transformed features.
        """
        X_transformed = pd.DataFrame(index=X.index)  # Initialize an empty DataFrame for transformed data
        for col in X.columns:
            # If no transformation was selected for the feature, copy it directly
            if col not in self.transformations:
                X_transformed[col] = X[col]
                continue
            trans_info = self.transformations[col]  # Retrieve transformation info for this feature
            method = trans_info['method']
            transformer = trans_info['transformer']

            if np.issubdtype(X[col].dtype, np.number):  # Check if the column is numeric
                # Apply numeric transformations based on the chosen method
                if method == 'none':
                    X_transformed[col] = X[col]
                elif method == 'log':
                    X_transformed[col] = np.log1p(X[col])  # Apply log transformation (log1p to handle zero values)
                elif method == 'sqrt':
                    X_transformed[col] = np.sqrt(X[col])  # Apply square root transformation
                elif method == 'yeo-johnson' and transformer:
                    col_values = X[col].fillna(X[col].mean()).values.reshape(-1, 1)  # Handle missing values
                    X_transformed[col] = transformer.transform(col_values).ravel()  # Apply Yeo-Johnson transformation
                elif method == 'quantile' and transformer:
                    col_values = X[col].fillna(X[col].mean()).values.reshape(-1, 1)  # Handle missing values
                    X_transformed[col] = transformer.transform(col_values).ravel()  # Apply quantile transformation
                elif method == 'boxcox':
                    X_transformed[col] = stats.boxcox(X[col], transformer)  # Apply Box-Cox transformation (requires all values > 0)
                elif method == 'standardization' and transformer:
                    col_values = X[col].fillna(X[col].mean()).values.reshape(-1, 1)  # Handle missing values
                    X_transformed[col] = transformer.transform(col_values).ravel()  # Apply standardization (Z-score normalization)
                elif method == 'minmax' and transformer:
                    col_values = X[col].fillna(X[col].mean()).values.reshape(-1, 1)  # Handle missing values
                    X_transformed[col] = transformer.transform(col_values).ravel()  # Apply Min-Max scaling
                elif method == 'polynomial' and transformer:
                    col_values = X[col].fillna(X[col].mean()).values.reshape(-1, 1)  # Handle missing values
                    X_transformed[col] = transformer.transform(col_values)[:, 1]  # Apply polynomial features (quadratic term only)
                elif method == 'binning':
                    X_transformed[col] = pd.qcut(X[col], q=5, labels=False)  # Apply binning (quantile-based discretization)
                else:
                    X_transformed[col] = X[col]  # If no known method, keep the column unchanged
            else:
                # Apply transformations for categorical features (encoding)
                if method == 'one_hot':
                    # Generate one-hot encoded dummy variables for categorical data
                    dummies = pd.get_dummies(X[col], prefix=col)
                    X_transformed = pd.concat([X_transformed, dummies], axis=1)
                elif method == 'label':
                    # Label encode the categorical variable
                    X_transformed[col] = LabelEncoder().fit_transform(X[col])
                else:
                    X_transformed[col] = X[col]  # If no encoding, keep the column unchanged

        return X_transformed  # Return the transformed dataset


    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit the transformation recommender to the input dataset and apply the selected transformations.

        This method is a convenience function that combines both fitting and transforming in one call.
        It first fits the recommender (selecting the best transformations), then transforms the dataset
        according to the selected transformations.

        Args:
            X (pd.DataFrame): The input dataset, where each column represents a feature to be transformed.
            y (pd.Series, optional): The target variable, provided for supervised learning. Defaults to None.

        Returns:
            pd.DataFrame: The transformed dataset with the same index as the original input,
                          but with features transformed according to the selected transformations.
        """
        self.fit(X, y)
        return self.transform(X)