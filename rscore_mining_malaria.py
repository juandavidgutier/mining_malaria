# Importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy.stats as stats
from econml.dml import SparseLinearDML, LinearDML, CausalForestDML
from econml.orf import DMLOrthoForest
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error

# Compatibility with modern NumPy versions
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print(f"EconML version: {econml.__version__}")

data_all = pd.read_csv("D:/data_final.csv", encoding='latin-1')

# Remove rows with missing values
data_all = data_all.dropna()
print(f"Dimensions after dropna: {data_all.shape}")

# 1. Label Encoding for DANE
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(
    data_all[['DANE_labeled']]
)

# 2. Label Encoding for Deparment_DANE
le_year = LabelEncoder()
data_all['Deparment_DANE_labeled'] = le_year.fit_transform(data_all['Deparment_DANE'])
scaler_DDANE = MinMaxScaler()
data_all['Deparment_DANE_normalized'] = scaler_DDANE.fit_transform(
    data_all[['Deparment_DANE_labeled']]
)

# 3. Label Encoding for Deparment_Month
le_year = LabelEncoder()
data_all['Deparment_Month_labeled'] = le_year.fit_transform(data_all['Deparment_Month'])
scaler_DDANE = MinMaxScaler()
data_all['Deparment_Month_normalized'] = scaler_DDANE.fit_transform(
    data_all[['Deparment_Month_labeled']]
)

# 4. Label Encoding for DANE_period
le_year = LabelEncoder()
data_all['DANE_period_labeled'] = le_year.fit_transform(data_all['DANE_period'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_period_normalized'] = scaler_DDANE.fit_transform(
    data_all[['DANE_period_labeled']]
)

# 5. Label Encoding for DANE_Department_Month
le_year = LabelEncoder()
data_all['DANE_Department_Month_labeled'] = le_year.fit_transform(data_all['DANE_Department_Month'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Department_Month_normalized'] = scaler_DDANE.fit_transform(
    data_all[['DANE_Department_Month_labeled']]
)

std_mining = data_all['mining'].std()
print(f"std of mining: {std_mining}")

median_mining = data_all['mining'].median()
print(f"median of mining: {median_mining}")

scaler = StandardScaler()
data_all['forest'] = scaler.fit_transform(data_all[['forest']])
data_all['MPI'] = scaler.fit_transform(data_all[['MPI']])
data_all['temperature'] = scaler.fit_transform(data_all[['temperature']])
data_all['rainfall'] = scaler.fit_transform(data_all[['rainfall']])
data_all['deforest'] = scaler.fit_transform(data_all[['deforest']])
data_all['vector'] = scaler.fit_transform(data_all[['vector']])
data_all['fire'] = scaler.fit_transform(data_all[['fire']])
data_all['coca'] = scaler.fit_transform(data_all[['coca']])


# ============================================================================
# WRAPPER FOR PROBABILISTIC CLASSIFIERS
# ============================================================================

class ProbClassifierWrapper(BaseEstimator):
    """
    Wrapper for sklearn classifiers that returns probabilities in predict().

    For binary outcomes, EconML expects model_y.predict(X) to return
    E[Y|X] as continuous probabilities, not discrete classes.
    """
    def __init__(self, base_clf=None, calibrate=True, random_state=123):
        if base_clf is None:
            base_clf = RandomForestClassifier(
                n_estimators=200,
                n_jobs=1,
                random_state=random_state,
                class_weight='balanced'
            )
        self.base_clf = base_clf
        self.calibrate = calibrate
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """Fit the base classifier (with or without calibration)"""
        y = np.asarray(y).ravel()

        if self.calibrate:
            self.model_ = CalibratedClassifierCV(
                estimator=clone(self.base_clf),
                cv=3
            )
            self.model_.fit(X, y)
        else:
            self.model_ = clone(self.base_clf)
            self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Return positive class probabilities (P(Y=1|X))"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before predict()")
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        """Return full probability matrix"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before predict_proba()")
        return self.model_.predict_proba(X)


# ============================================================================
# DATA PREPARATION FOR ECONML
# ============================================================================

def prepare_data_vectors(df):
    """
    Prepares data matrices according to the user's exact specification.
    """
    y = df['excess'].astype(int).values.ravel()
    t = df['mining'].astype(float).values.ravel()
    W = df[['fire', 'coca']].values
    X = df[['altitude', 'DANE_normalized', 'Deparment_DANE_normalized', 'Deparment_Month_normalized', 'DANE_period_normalized', 'DANE_Department_Month_normalized']].values

    return y, t, W, X


# ============================================================================
# NUISANCE MODEL DIAGNOSTICS
# ============================================================================

def nuisance_diagnostics(y_tr, t_tr, W_tr, X_tr, y_val, t_val, W_val, X_val, random_state=123):
    """
    Diagnose the quality of nuisance models (first-stage models).
    """
    Z_tr = np.hstack([X_tr, W_tr])
    Z_val = np.hstack([X_val, W_val])

    # Model Y: Probabilistic classifier for binary outcome
    model_y = ProbClassifierWrapper(
        RandomForestClassifier(
            n_estimators=200,
            n_jobs=1,
            class_weight='balanced',
            random_state=random_state
        ),
        calibrate=True,
        random_state=random_state
    )
    model_y.fit(Z_tr, y_tr)
    y_pred_val = model_y.predict(Z_val)

    # Model T: Regressor for continuous treatment
    model_t = RandomForestRegressor(
        n_estimators=200,
        n_jobs=1,
        random_state=random_state
    )
    model_t.fit(Z_tr, t_tr)
    t_pred_val = model_t.predict(Z_val)

    # Residuals (orthogonalized component)
    y_res = y_val - y_pred_val
    t_res = t_val - t_pred_val

    diagnostics = {
        'y_pred_val_mean': np.mean(y_pred_val),
        'y_res_var': np.var(y_res),
        't_pred_val_mean': np.mean(t_pred_val),
        't_res_var': np.var(t_res),
        'y_t_res_corr': np.corrcoef(y_res, t_res)[0, 1] if len(y_res) > 1 else np.nan
    }

    return diagnostics, y_pred_val, t_pred_val, y_res, t_res


def make_sparselinear_models(random_state=123):
    """
    Generates SparseLinearDML models with a hyperparameter grid for nuisance models.

    CRITICAL: The hyperparameters n_estimators and max_depth apply ONLY
    to the nuisance models (RandomForest), NOT to SparseLinearDML.

    Returns:
        List[Tuple[str, SparseLinearDML]]: List of (name, model) configurations
    """
    models = []

    # Hyperparameter grid for Random Forest (nuisance models)
    hyperparameter_grid = [
        (3800, 45), (3600, 35), (3800, 30),
        (3500, 40), (3400, 30), (3600, 45)
    ]

    print(f"Generating {len(hyperparameter_grid)} model configurations...")

    for n_est, depth in hyperparameter_grid:
        name = f"SparseLinearDML_n{n_est}_d{depth}"

        # =====================================================================
        # NUISANCE MODEL Y (Outcome): Classifier for binary variable
        # =====================================================================
        base_clf_y = RandomForestClassifier(
            n_estimators=n_est,          # ✓ Grid hyperparameter
            max_depth=depth,             # ✓ Grid hyperparameter
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',         # Additional regularization
            n_jobs=1,
            class_weight='balanced',
            random_state=random_state
        )

        model_y_inst = ProbClassifierWrapper(
            base_clf=base_clf_y,
            calibrate=True,
            random_state=random_state
        )

        # =====================================================================
        # NUISANCE MODEL T (Treatment): Regressor for continuous variable
        # =====================================================================
        model_t_inst = RandomForestRegressor(
            n_estimators=n_est,          # ✓ Grid hyperparameter
            max_depth=depth,             # ✓ Grid hyperparameter
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=1,
            random_state=random_state
        )

        # =====================================================================
        # CAUSAL ESTIMATOR: SparseLinearDML
        # =====================================================================
        # NOTE: SparseLinearDML does NOT accept n_estimators or max_depth
        # It only receives pre-configured nuisance models and its own parameters

        try:
            sparselinear_dml = SparseLinearDML(
                model_y=model_y_inst,              # ✓ Nuisance model Y
                model_t=model_t_inst,              # ✓ Nuisance model T
                discrete_outcome=True,             # ✓ Outcome is binary (0/1)
                discrete_treatment=False,          # ✓ Treatment is continuous
                fit_cate_intercept=True,           # ✓ Intercept in CATE
                alpha='auto',                      # ✓ Automatic LASSO penalty
                max_iter=50000,                    # ✓ LASSO iterations
                tol=1e-4,                          # ✓ Convergence tolerance
                cv=5,                              # ✓ Cross-validation folds
                random_state=random_state          # ✓ Seed
                # ❌ DO NOT INCLUDE: n_estimators, max_depth, n_jobs
            )

            models.append((name, sparselinear_dml))
            print(f"  ✓ {name} configured successfully")

        except TypeError as e:
            print(f"  ❌ ERROR in {name}: {e}")
            print(f"     Verify you are not passing invalid arguments to SparseLinearDML")
            raise

    print(f"\n✓ Total models generated: {len(models)}")
    return models


# ============================================================================
# MODEL EVALUATION WITH RSCORER (UPDATED API)
# ============================================================================

def fit_and_evaluate_with_rscore(df, models_to_try, random_state=123):
    """
    Fits multiple DML models and evaluates them using RScorer.
    Compatible with EconML >= 0.13
    """
    print(f"\n{'='*80}")
    print("STARTING MODEL EVALUATION WITH RSCORER")
    print(f"{'='*80}")

    # Prepare data
    y, t, W, X = prepare_data_vectors(df)

    n = len(y)
    n_positives = int(y.sum())
    prevalence = n_positives / n
    var_t = np.var(t)

    print(f"\nDataset statistics:")
    print(f"  N observations: {n}")
    print(f"  Positive cases (Excess_cases_tp1=1): {n_positives} ({prevalence:.2%})")
    print(f"  Treatment variance (Deforestation_t): {var_t:.6f}")

    # Stratified train/validation split
    print(f"\nPerforming stratified split (60% train, 40% validation)...")

    try:
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W,
            test_size=0.4,
            random_state=random_state,
            stratify=y
        )
    except ValueError:
        print("⚠️ Stratification not possible. Using random split.")
        X_tr, X_val, t_tr, t_val, y_tr, y_val, W_tr, W_val = train_test_split(
            X, t, y, W,
            test_size=0.4,
            random_state=random_state
        )

    print(f"  Train: N={len(y_tr)}, Positive cases={int(y_tr.sum())} ({y_tr.mean():.2%})")
    print(f"  Validation: N={len(y_val)}, Positive cases={int(y_val.sum())} ({y_val.mean():.2%})")

    # Nuisance model diagnostics
    print(f"\n{'='*80}")
    print("NUISANCE MODEL DIAGNOSTICS (First-Stage)")
    print(f"{'='*80}")

    diag, y_pred_val, t_pred_val, y_res, t_res = nuisance_diagnostics(
        y_tr, t_tr, W_tr, X_tr,
        y_val, t_val, W_val, X_val,
        random_state=random_state
    )

    for key, value in diag.items():
        print(f"  {key}: {value:.6f}")

    # Baseline MSE (constant prediction)
    baseline_pred = y_val.mean()
    baseline_mse = mean_squared_error(y_val, np.full_like(y_val, baseline_pred, dtype=float))
    print(f"\n  Baseline (constant prediction): P(Y=1)={baseline_pred:.6f}, MSE={baseline_mse:.6f}")

    # Fit models in parallel
    print(f"\n{'='*80}")
    print("FITTING MODELS IN PARALLEL")
    print(f"{'='*80}")
    print(f"Number of models to evaluate: {len(models_to_try)}")

    def fit_single_model(name, model):
        """Helper function for parallel fitting"""
        try:
            model.fit(Y=y_tr, T=t_tr, X=X_tr, W=W_tr)
            return (name, model, None)
        except Exception as e:
            return (name, None, str(e))

    results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
        delayed(fit_single_model)(name, mdl) for name, mdl in models_to_try
    )

    # Process fitting results
    fitted_models = []
    failed_models = []

    for name, mdl, error in results:
        if error is not None:
            print(f"  ❌ ERROR in {name}: {error}")
            failed_models.append((name, error))
        else:
            print(f"  ✓ {name} fitted successfully")
            fitted_models.append((name, mdl))

    if len(fitted_models) == 0:
        print("\n❌ CRITICAL: No model was fitted successfully.")
        return {
            'n': n,
            'prevalence': prevalence,
            'var_treatment': var_t,
            'nuisance_diagnostics': diag,
            'baseline_mse': baseline_mse,
            'fitted_models': [],
            'rscores': [],
            'best_model': None,
            'failed_models': failed_models
        }

    # ========================================================================
    # EVALUATION WITH RSCORER - CORRECTED API
    # ========================================================================
    print(f"\n{'='*80}")
    print("EVALUATION WITH RSCORER (Cross-Validation Method)")
    print(f"{'='*80}")

    rscores = []

    for name, mdl in fitted_models:
        try:
            # METHOD 1: Use score() directly from the fitted model
            # This method uses the model's internal cross-validation
            rscore = mdl.score(Y=y_val, T=t_val, X=X_val, W=W_val)
            rscores.append((name, rscore))
            print(f"  {name}: R² = {rscore:.6f}")

        except AttributeError:
            # METHOD 2: Manual R² calculation if score() is unavailable
            try:
                # Predict CATE on validation set
                cate_pred = mdl.effect(X=X_val)

                # Compute outcome residuals
                Z_val = np.hstack([X_val, W_val])
                model_y_temp = ProbClassifierWrapper(
                    RandomForestClassifier(n_estimators=200, random_state=random_state),
                    calibrate=True,
                    random_state=random_state
                )
                model_y_temp.fit(Z_val, y_val)
                y_pred = model_y_temp.predict(Z_val)
                y_res = y_val - y_pred

                # Compute treatment residuals
                model_t_temp = RandomForestRegressor(n_estimators=200, random_state=random_state)
                model_t_temp.fit(Z_val, t_val)
                t_pred = model_t_temp.predict(Z_val)
                t_res = t_val - t_pred

                # R² score: correlation between CATE predictions and residuals
                if np.std(t_res) > 0 and np.std(y_res) > 0:
                    # Score based on MSE
                    mse_baseline = np.mean((y_res - np.mean(y_res))**2)
                    mse_model = np.mean((y_res - cate_pred * t_res)**2)
                    rscore = 1 - (mse_model / mse_baseline) if mse_baseline > 0 else 0.0
                else:
                    rscore = 0.0

                rscores.append((name, rscore))
                print(f"  {name}: R² = {rscore:.6f} (manual calculation)")

            except Exception as e2:
                print(f"  ❌ Error calculating R² for {name}: {e2}")
                rscores.append((name, np.nan))

        except Exception as e:
            print(f"  ❌ Error calculating RScore for {name}: {e}")
            rscores.append((name, np.nan))

    # Identify best model
    valid_rscores = [(n, s) for n, s in rscores if not np.isnan(s)]

    if len(valid_rscores) == 0:
        print("\n⚠️ No valid RScores could be computed.")
        best_model = None
    else:
        best_model = max(valid_rscores, key=lambda x: x[1])
        print(f"\n{'*'*80}")
        print(f"🏆 BEST MODEL: {best_model[0]}")
        print(f"   R² Score: {best_model[1]:.6f}")
        print(f"{'*'*80}")

    # Return complete results
    return {
        'n': n,
        'prevalence': prevalence,
        'var_treatment': var_t,
        'nuisance_diagnostics': diag,
        'baseline_mse': baseline_mse,
        'fitted_models': fitted_models,
        'rscores': rscores,
        'best_model': best_model,
        'failed_models': failed_models,
        'X_train': X_tr,
        'X_val': X_val,
        'y_train': y_tr,
        'y_val': y_val,
        't_train': t_tr,
        't_val': t_val,
        'W_train': W_tr,
        'W_val': W_val
    }


# ============================================================================
# VERIFICATION SCRIPT
# ============================================================================

print("="*80)
print("CONFIGURATION VERIFICATION")
print("="*80)

# 1. Check library versions
import econml, sklearn, numpy as np
print(f"\n📦 Library versions:")
print(f"  EconML: {econml.__version__}")
print(f"  Scikit-learn: {sklearn.__version__}")
print(f"  NumPy: {np.__version__}")

# 2. Verify ProbClassifierWrapper works
print(f"\n🔧 Testing ProbClassifierWrapper...")
from sklearn.datasets import make_classification
X_test, y_test = make_classification(n_samples=100, n_features=5, random_state=123)

try:
    wrapper = ProbClassifierWrapper(random_state=123)
    wrapper.fit(X_test, y_test)
    preds = wrapper.predict(X_test)
    assert len(preds) == len(y_test), "Dimension mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of range"
    print(f"  ✓ ProbClassifierWrapper works correctly")
    print(f"    Probability range: [{preds.min():.4f}, {preds.max():.4f}]")
except Exception as e:
    print(f"  ❌ Error in ProbClassifierWrapper: {e}")

# 3. Verify SparseLinearDML accepts correct arguments
print(f"\n🔍 Verifying SparseLinearDML arguments...")
import inspect
sig = inspect.signature(SparseLinearDML.__init__)
valid_params = list(sig.parameters.keys())
print(f"  Valid parameters: {valid_params[:10]}...")  # First 10

# Verify that n_estimators is NOT in the list
if 'n_estimators' in valid_params:
    print(f"  ⚠️ WARNING: Your EconML version may have a different API")
else:
    print(f"  ✓ Confirmed: n_estimators is NOT a parameter of SparseLinearDML")

# 4. Create a test model
print(f"\n🧪 Creating test model...")
try:
    test_model = SparseLinearDML(
        model_y=ProbClassifierWrapper(random_state=123),
        model_t=RandomForestRegressor(n_estimators=100, random_state=123),
        discrete_outcome=True,
        discrete_treatment=False,
        featurizer=PolynomialFeatures(degree=2, include_bias=False),
        random_state=123
    )
    print(f"  ✓ SparseLinearDML instantiated successfully")
except TypeError as e:
    print(f"  ❌ ERROR: {e}")
    print(f"     There is an issue with SparseLinearDML arguments")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETED")
print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*80)
print("="*80)
print("CAUSAL ANALYSIS: DEFORESTATION → LEISHMANIASIS")
print("Method: Double Machine Learning (SparseLinearDML)")
print("Evaluation: RScorer")
print("="*80)
print("="*80)

# Generate candidate models with specified hyperparameters
models_to_try = make_sparselinear_models(random_state=seed)

print(f"\nCandidate models generated: {len(models_to_try)}")
for name, _ in models_to_try:
    print(f"  - {name}")

# Run full analysis
results = fit_and_evaluate_with_rscore(data_all, models_to_try, random_state=seed)


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print("="*80)

print(f"\n📊 DATASET STATISTICS:")
print(f"  N observations: {results['n']}")
print(f"  Prevalence (Excess_cases_tp1=1): {results['prevalence']:.2%}")
print(f"  Treatment variance: {results['var_treatment']:.6f}")
print(f"  Baseline MSE: {results['baseline_mse']:.6f}")

print(f"\n🔧 NUISANCE MODEL DIAGNOSTICS:")
for key, value in results['nuisance_diagnostics'].items():
    print(f"  {key}: {value:.6f}")

print(f"\n📈 RSCORES (All models):")
sorted_rscores = sorted(results['rscores'], key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
for i, (name, score) in enumerate(sorted_rscores, 1):
    print(f"  {i}. {name}: {score:.6f}")

if results['best_model'] is not None:
    print(f"\n🏆 BEST MODEL:")
    print(f"  Name: {results['best_model'][0]}")
    print(f"  R² Score: {results['best_model'][1]:.6f}")

    # Extract best model for further analysis
    best_model_name = results['best_model'][0]
    best_model_obj = next((mdl for name, mdl in results['fitted_models'] if name == best_model_name), None)

    if best_model_obj is not None:
        print(f"\n📋 INTERPRETATION OF R² SCORE:")
        rscore_val = results['best_model'][1]
        if rscore_val > 0.1:
            print(f"  ✓ Excellent: The model captures significant causal effect heterogeneity")
        elif rscore_val > 0:
            print(f"  ✓ Moderate: There is effect heterogeneity, but limited")
        else:
            print(f"  ⚠️ Low: Little evidence of heterogeneity or overfitting")

        # ============================================================================
        # CAUSAL EFFECT ESTIMATION WITH BEST MODEL
        # ============================================================================

        print("\n" + "="*80)
        print("CAUSAL EFFECT ESTIMATION (CATE) WITH BEST MODEL")
        print("="*80)

        # Average Treatment Effect (ATE)
        ate = best_model_obj.ate(X=results['X_val'])
        ate_inference = best_model_obj.ate_inference(X=results['X_val'])
        ate_ci = ate_inference.conf_int_mean(alpha=0.05)

        print(f"\n📊 AVERAGE TREATMENT EFFECT (ATE):")
        print(f"  Estimate: {ate[0]:.6f}")
        print(f"  95% CI: [{ate_ci[0][0]:.6f}, {ate_ci[1][0]:.6f}]")

        # Interpretation
        print(f"\n💡 SUBSTANTIVE INTERPRETATION:")
        print(f"  A one standard deviation increase in deforestation is causally associated")
        print(f"  with a change of {ate[0]:.4f} units in the probability")
        print(f"  of excess leishmaniasis cases.")
        print(f"  The 95% confidence interval for this effect is [{ate_ci[0][0]:.6f}, {ate_ci[1][0]:.6f}].")

        # Conditional Average Treatment Effects (CATE)
        cate_val = best_model_obj.effect(X=results['X_val'])

        print(f"\n📈 CONDITIONAL AVERAGE TREATMENT EFFECTS (CATE):")
        print(f"  Mean: {np.mean(cate_val):.6f}")
        print(f"  Median: {np.median(cate_val):.6f}")
        print(f"  Std. Dev: {np.std(cate_val):.6f}")
        print(f"  Min: {np.min(cate_val):.6f}, Max: {np.max(cate_val):.6f}")

        # CATE distribution
        percentiles = np.percentile(cate_val, [10, 25, 50, 75, 90])
        print(f"\n  CATE Percentiles:")
        print(f"    P10: {percentiles[0]:.6f}")
        print(f"    P25: {percentiles[1]:.6f}")
        print(f"    P50 (median): {percentiles[2]:.6f}")
        print(f"    P75: {percentiles[3]:.6f}")
        print(f"    P90: {percentiles[4]:.6f}")

        # Heterogeneity analysis
        print(f"\n🔍 HETEROGENEITY ANALYSIS:")
        cate_range = np.max(cate_val) - np.min(cate_val)
        print(f"  CATE range: {cate_range:.6f}")

        if cate_range > 0.1:
            print(f"  ✓ Substantial heterogeneity in causal effect")
            print(f"    The effect of deforestation varies significantly across municipalities/time periods")
        else:
            print(f"  ⚠️ Limited heterogeneity in causal effect")
            print(f"    The effect of deforestation is relatively homogeneous")

else:
    print(f"\n⚠️ No valid best model identified.")

if len(results['failed_models']) > 0:
    print(f"\n❌ FAILED MODELS: {len(results['failed_models'])}")
    for name, error in results['failed_models'][:3]:
        print(f"  - {name}: {error[:100]}...")

print("\n" + "="*80)
print("--- SCRIPT COMPLETED SUCCESSFULLY ---")
print("="*80)