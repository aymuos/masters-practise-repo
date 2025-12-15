from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, roc_auc_score
import pandas as pd
import numpy as np
import pickle
import os
import time

class LoopClassifier:
    def __init__(self):
        # Define candidate models
        # Added class_weight='balanced' to handle skewed datasets
        self.classifiers_pool = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        self.regressors_pool = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'SVR': SVR(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        self.best_models = {} # Store the winning model for each target
        self.feature_cols = []

    def train_and_evaluate_cv(self, X, y_dict, k=5):
        """
        Train multiple models using k-Fold CV, select the best one for each target.
        Calculates Accuracy, AUC-ROC, Latency, and MSE/R2.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y_dict (dict of pd.Series): Dictionary of target vectors.
            k (int): Number of folds.
            
        Returns:
            dict: Best metrics per target.
            pd.DataFrame: CV predictions of the BEST model for the dataset.
            dict: Full comparison log of all models.
        """
        self.feature_cols = X.columns.tolist()
        
        best_metrics_results = {}
        full_comparison_log = {}
        
        # DataFrame to store the unbiased CV predictions of the WINNING model
        cv_preds_best_all = pd.DataFrame(index=X.index)
        
        for target_name, y in y_dict.items():
            print(f"\n=== Evaluating Models for Target: {target_name} ===")
            
            # Align Data
            valid_mask = ~y.isna()
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(y_clean) < k:
                print(f"Skipping {target_name}: Not enough data.")
                continue

            # Determine task type
            is_regression = 'Score' in target_name
            
            # Calculate negative/positive ratio for XGBoost if Classification
            scale_pos_weight = 1.0
            if not is_regression:
                pos_count = np.sum(y_clean == 1)
                neg_count = np.sum(y_clean == 0)
                if pos_count > 0:
                     scale_pos_weight = neg_count / pos_count

            candidate_pool = self.regressors_pool if is_regression else self.classifiers_pool
            cv_strategy = KFold(n_splits=k, shuffle=True, random_state=42) if is_regression else StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            
            best_score = -float('inf')
            best_model_name = None
            best_y_pred_cv = None
            target_comparison = {}
            
            for model_name, model in candidate_pool.items():
                print(f" Testing {model_name}...", end=" ")
                
                # Dynamic XGBoost weighting
                if model_name == 'XGBoost' and not is_regression:
                    model.set_params(scale_pos_weight=scale_pos_weight)
                    
                start_time = time.time()
                try:
                    # 1. Predictions (Classes/Values)
                    y_pred_cv = cross_val_predict(model, X_clean, y_clean, cv=cv_strategy)
                    
                    # 2. Probability (for AUC, only classifiers)
                    y_prob_cv = None
                    if not is_regression and hasattr(model, "predict_proba"):
                        try:
                            # predict_proba returns (n_samples, n_classes), we want prob of class 1
                            y_prob_cv = cross_val_predict(model, X_clean, y_clean, cv=cv_strategy, method='predict_proba')[:, 1]
                        except:
                            pass # Some models might fail or not support it via cross_val_predict easily
                    
                    latency = (time.time() - start_time) / k # Average time per fold
                    
                    metrics = {'latency': latency}
                    
                    if is_regression:
                        score = r2_score(y_clean, y_pred_cv)
                        metric_val = score # Maximize R2
                        mse = mean_squared_error(y_clean, y_pred_cv)
                        metrics.update({'r2': score, 'mse': mse})
                        print(f"R2: {score:.4f} (Time: {latency:.3f}s)")
                    else:
                        score = accuracy_score(y_clean, y_pred_cv)
                        metrics.update({'accuracy': score})
                        
                        auc = 0.5
                        if y_prob_cv is not None:
                            try:
                                auc = roc_auc_score(y_clean, y_prob_cv)
                                metrics['roc_auc'] = auc
                            except:
                                metrics['roc_auc'] = 0.5 # Fail-safe
                        
                        # Preferred Metric: AUC (unless not computed/applicable)
                        metric_val = auc if 'roc_auc' in metrics else score
                        
                        print(f"Acc: {score:.4f} (AUC: {metrics.get('roc_auc', 0):.4f}, Time: {latency:.3f}s)")
                        
                    target_comparison[model_name] = metrics
                        
                    # Check if best
                    if metric_val > best_score:
                        best_score = metric_val
                        best_model_name = model_name
                        best_y_pred_cv = y_pred_cv
                        
                except Exception as e:
                    print(f"Failed: {e}")
            
            print(f" >> WINNER for {target_name}: {best_model_name}")
            
            # Store Winner info
            full_comparison_log[target_name] = target_comparison
            best_metrics_results[target_name] = target_comparison[best_model_name]
            best_metrics_results[target_name]['model_name'] = best_model_name
            
            # Save Predictions of Winner
            cv_preds_series = pd.Series(best_y_pred_cv, index=X_clean.index, name=target_name)
            cv_preds_best_all = cv_preds_best_all.join(cv_preds_series, how='outer')
            
            # Retrain Winner on Full Data
            winner_model = candidate_pool[best_model_name]
            winner_model.fit(X_clean, y_clean)
            self.best_models[target_name] = winner_model
            
            # Save Feature Importance if available
            os.makedirs('outputs', exist_ok=True)
            if hasattr(winner_model, 'feature_importances_'):
                # 1. Store in results dict
                importances = winner_model.feature_importances_
                best_metrics_results[target_name]['feature_importance'] = dict(zip(self.feature_cols, importances))
                
                # 2. Save CSV to disk for transparency
                fi_df = pd.DataFrame({'feature': self.feature_cols, 'importance': importances})
                fi_df = fi_df.sort_values(by='importance', ascending=False)
                fi_df.to_csv(f"outputs/feature_importance_{target_name}.csv", index=False)

        # Save CV Predictions to disk
        cv_preds_best_all.to_csv("outputs/cv_predictions_best_models.csv")
        return best_metrics_results, cv_preds_best_all, full_comparison_log

    def predict(self, X):
        """
        Predict for new data.
        
        Args:
            X (pd.DataFrame): Feature matrix.
        Predict using the trained BEST models.
        """
        # X should match feature_cols. If missing cols, fill 0? 
        # For simplicity assume X has correct columns. 
        # In robust prod, we'd align columns.
        X_aligned = X.reindex(columns=self.feature_cols, fill_value=0)
        
        preds = {}
        probs = {}
        
        for name, model in self.best_models.items():
            if hasattr(model, "predict"):
                preds[name] = model.predict(X_aligned)
                if hasattr(model, "predict_proba"):
                    try:
                        probs[name] = model.predict_proba(X_aligned)[:, 1] # Prob of class 1
                    except:
                        probs[name] = preds[name]
                else:
                    probs[name] = preds[name] # For regressors
            else:
                print(f"Model {name} has no predict method.")
                
        return pd.DataFrame(preds), pd.DataFrame(probs)

    def predict_debug(self, X, loop_id):
        """
        Predicts for a single loop and prints details for debugging.
        """
        # Ensure we filter to just the row if X has index
        if str(loop_id) in X.index:
            X_row = X.loc[[str(loop_id)]]
        else:
            # Fallback if X is single row dataframe or mismatch
            X_row = X
            
        preds, probs = self.predict(X_row)

        print(f"\nDEBUG — Loop {loop_id}")
        for target in preds.columns:
            val = preds[target].iloc[0]
            prob = probs[target].iloc[0] if target in probs else 0.0
            print(f"{target}: pred={val}, prob={prob:.3f}")
        return preds, probs
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save both models and the feature list required for prediction
        data = {
            'best_models': self.best_models,
            'feature_cols': self.feature_cols
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Best models and feature configuration saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle backward compatibility or new format
        if isinstance(data, dict) and 'best_models' in data:
            self.best_models = data['best_models']
            self.feature_cols = data.get('feature_cols', [])
        else:
            # Fallback if old format was just the models dict (though we overwriting now)
            self.best_models = data
            self.feature_cols = []
            print("Warning: Loaded legacy model file without feature columns. Prediction might fail if features mismatch.")
            
        print(f"Models loaded from {filepath}")

if __name__ == "__main__":
    from utils_logger import setup_logging
    setup_logging("models")

    # Test class
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.rand(100, 5), columns=['f1','f2','f3','f4','f5'])
    y_dummy = pd.Series(np.random.randint(0, 2, 100), name='Oscillation_Flag')
    
    clf = LoopClassifier()
    clf.train(X_dummy, {'Oscillation_Flag': y_dummy})
    
    p, prob = clf.predict(X_dummy.iloc[:5])
    print("Predictions:\n", p)
