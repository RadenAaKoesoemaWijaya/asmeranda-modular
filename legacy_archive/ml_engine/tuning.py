from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def create_optuna_study(problem_type, model_type, X_train, y_train, cv_params, custom_param_ranges=None):
    
    def objective(trial):
        if problem_type == "Classification":
            if model_type == "Random Forest":
                if custom_param_ranges and 'n_estimators' in custom_param_ranges:
                    n_estimators_range = custom_param_ranges['n_estimators']
                    if isinstance(n_estimators_range, list) and len(n_estimators_range) == 3:
                        n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1], step=n_estimators_range[2])
                    else:
                        n_estimators = trial.suggest_int('n_estimators', 50, 300)
                else:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                
                if custom_param_ranges and 'max_depth' in custom_param_ranges:
                    max_depth_range = custom_param_ranges['max_depth']
                    if isinstance(max_depth_range, list) and len(max_depth_range) == 3:
                        max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1], step=max_depth_range[2])
                    else:
                        max_depth = trial.suggest_int('max_depth', 3, 20)
                else:
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                
                if custom_param_ranges and 'min_samples_split' in custom_param_ranges:
                    min_samples_split_range = custom_param_ranges['min_samples_split']
                    if isinstance(min_samples_split_range, list) and len(min_samples_split_range) == 3:
                        min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1], step=min_samples_split_range[2])
                    else:
                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                else:
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                
                if custom_param_ranges and 'min_samples_leaf' in custom_param_ranges:
                    min_samples_leaf_range = custom_param_ranges['min_samples_leaf']
                    if isinstance(min_samples_leaf_range, list) and len(min_samples_leaf_range) == 3:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1], step=min_samples_leaf_range[2])
                    else:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                else:
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                
                if custom_param_ranges and 'max_features' in custom_param_ranges:
                    max_features_values = custom_param_ranges['max_features']
                    if isinstance(max_features_values, list):
                        max_features = trial.suggest_categorical('max_features', max_features_values)
                    else:
                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                else:
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                if custom_param_ranges and 'bootstrap' in custom_param_ranges:
                    bootstrap_values = custom_param_ranges['bootstrap']
                    if isinstance(bootstrap_values, list):
                        bootstrap = trial.suggest_categorical('bootstrap', bootstrap_values)
                    else:
                        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                else:
                    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'bootstrap': bootstrap,
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif model_type == "Logistic Regression":
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
                    'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
                    'random_state': 42
                }
                if params['penalty'] == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                model = LogisticRegression(**params)
            elif model_type == "SVM":
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'coef0': trial.suggest_float('coef0', 0.0, 1.0),
                    'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                    'probability': True,
                    'random_state': 42
                }
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 10)
                model = SVC(**params)
            elif model_type == "KNN":
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'p': trial.suggest_int('p', 1, 5),
                    'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                    'metric': trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
                }
                model = KNeighborsClassifier(**params)
            elif model_type == "Decision Tree":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 'auto']),
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1, step=0.001),
                    'random_state': 42
                }
                model = DecisionTreeClassifier(**params)
            elif model_type == "Gradient Boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 'auto']),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
            elif model_type == "Naive Bayes":
                params = {
                    'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
                }
                model = GaussianNB(**params)
            else:
                return 0
                
            score = cross_val_score(model, X_train, y_train, cv=cv_params['cv'], scoring='accuracy').mean()
            return score
            
        else:  # Regression
            if model_type == "Random Forest":
                if custom_param_ranges and 'n_estimators' in custom_param_ranges:
                    n_estimators_range = custom_param_ranges['n_estimators']
                    if isinstance(n_estimators_range, list) and len(n_estimators_range) == 3:
                        n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1], step=n_estimators_range[2])
                    else:
                        n_estimators = trial.suggest_int('n_estimators', 50, 300)
                else:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                
                if custom_param_ranges and 'max_depth' in custom_param_ranges:
                    max_depth_range = custom_param_ranges['max_depth']
                    if isinstance(max_depth_range, list) and len(max_depth_range) == 3:
                        max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1], step=max_depth_range[2])
                    else:
                        max_depth = trial.suggest_int('max_depth', 3, 20)
                else:
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                
                if custom_param_ranges and 'min_samples_split' in custom_param_ranges:
                    min_samples_split_range = custom_param_ranges['min_samples_split']
                    if isinstance(min_samples_split_range, list) and len(min_samples_split_range) == 3:
                        min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1], step=min_samples_split_range[2])
                    else:
                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                else:
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                
                if custom_param_ranges and 'min_samples_leaf' in custom_param_ranges:
                    min_samples_leaf_range = custom_param_ranges['min_samples_leaf']
                    if isinstance(min_samples_leaf_range, list) and len(min_samples_leaf_range) == 3:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1], step=min_samples_leaf_range[2])
                    else:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                else:
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                
                if custom_param_ranges and 'max_features' in custom_param_ranges:
                    max_features_values = custom_param_ranges['max_features']
                    if isinstance(max_features_values, list):
                        max_features = trial.suggest_categorical('max_features', max_features_values)
                    else:
                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                else:
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                if custom_param_ranges and 'bootstrap' in custom_param_ranges:
                    bootstrap_values = custom_param_ranges['bootstrap']
                    if isinstance(bootstrap_values, list):
                        bootstrap = trial.suggest_categorical('bootstrap', bootstrap_values)
                    else:
                        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                else:
                    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'bootstrap': bootstrap,
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            elif model_type == "Linear Regression":
                params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    'positive': trial.suggest_categorical('positive', [True, False]),
                    'copy_X': trial.suggest_categorical('copy_X', [True, False])
                }
                model = LinearRegression(**params)
            elif model_type == "Gradient Boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
            elif model_type == "SVR":
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
                }
                model = SVR(**params)
            else:
                return 0
                
            score = cross_val_score(model, X_train, y_train, cv=cv_params['cv'], scoring='r2').mean()
            return score
    
    return objective

def parse_custom_range(range_str, param_type='int'):
    """
    Parse custom parameter range string into appropriate format
    """
    if not range_str or range_str.strip() == "":
        return None
    
    range_str = range_str.strip()
    
    # Handle categorical parameters (comma-separated values)
    if ',' in range_str:
        values = [val.strip() for val in range_str.split(',')]
        # Convert numeric strings to numbers, handle None/True/False
        converted_values = []
        for val in values:
            if val.lower() == 'none':
                converted_values.append(None)
            elif val.lower() == 'true':
                converted_values.append(True)
            elif val.lower() == 'false':
                converted_values.append(False)
            else:
                try:
                    if param_type == 'int':
                        converted_values.append(int(val))
                    elif param_type == 'float':
                        converted_values.append(float(val))
                    else:
                        converted_values.append(val)
                except ValueError:
                    converted_values.append(val)
        return converted_values
    
    # Handle numeric ranges (min:max:step format)
    elif ':' in range_str:
        parts = range_str.split(':')
        if len(parts) == 3:
            try:
                start = int(parts[0]) if param_type == 'int' else float(parts[0])
                stop = int(parts[1]) if param_type == 'int' else float(parts[1])
                step = int(parts[2]) if param_type == 'int' else float(parts[2])
                
                if param_type == 'int':
                    return list(range(start, stop + 1, step))
                else:
                    values = []
                    current = start
                    while current <= stop:
                        values.append(current)
                        current += step
                    return values
            except ValueError:
                return None
    
    # Single value
    try:
        if param_type == 'int':
            return [int(range_str)]
        elif param_type == 'float':
            return [float(range_str)]
        else:
            return [range_str]
    except ValueError:
        return None
