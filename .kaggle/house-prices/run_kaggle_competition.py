"""
Kaggle House Prices Competition - Full Pipeline Test

This script:
1. Registers the Kaggle training data
2. Invokes the training agent with no model preference (LLM chooses)
3. Stores the resulting model
4. Loads the Kaggle test data, applies transformations, and generates predictions
5. Creates a submission.csv file for Kaggle

Run: python .kaggle/house-prices/run_kaggle_competition.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models-tools" / "training"))

from utils import register_dataset, get_registered_dataset
from agents.training.agent import invoke_training_agent, resume_training_agent
from model_storage import load_model, get_model_info


def run_training_agent_auto_approve(goal: str, linked_datasets: list, user_model_preference: str = None):
    """
    Run the training agent with automatic approval of all HITL steps.
    This is useful for automated pipelines like Kaggle competitions.
    """
    print_step("STARTING", "Training agent with auto-approval of HITL steps")
    
    # Start the training agent
    result = invoke_training_agent(
        goal=goal,
        linked_datasets=linked_datasets,
        user_model_preference=user_model_preference
    )
    
    step_count = 0
    max_steps = 50  # Safety limit to prevent infinite loops
    
    # Loop to auto-approve each HITL interrupt
    while "__interrupt__" in result and step_count < max_steps:
        step_count += 1
        thread_id = result.get("_thread_id")
        
        # Extract interrupt info
        interrupt_data = result.get("__interrupt__", [])
        if interrupt_data:
            interrupt_obj = interrupt_data[0]
            interrupt_value = interrupt_obj.value if hasattr(interrupt_obj, "value") else interrupt_obj
            node = interrupt_value.get("node", "unknown") if isinstance(interrupt_value, dict) else "unknown"
            summary = interrupt_value.get("summary", "") if isinstance(interrupt_value, dict) else str(interrupt_value)
            
            print(f"\n  [STEP {step_count}] {node}")
            print(f"  Summary: {summary[:200]}..." if len(summary) > 200 else f"  Summary: {summary}")
            print(f"  → Auto-approving...")
        
        # Resume with approval
        result = resume_training_agent(decision=True, thread_id=thread_id)
        
        # Check for errors
        if result.get("error"):
            print(f"\n  ❌ Error: {result.get('error')}")
            break
    
    if step_count >= max_steps:
        print(f"\n  ⚠️ Reached max steps ({max_steps}), stopping.")
    
    print(f"\n  ✓ Completed {step_count} steps")
    return result


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

# Use existing model instead of training (set to None to train new)
USE_EXISTING_MODEL = None  # e.g., "xgboost_1769321495_v1"

# The goal prompt - crafted to guide the agent for this specific competition
# Enhanced with Kaggle competition best practices and domain knowledge
GOAL = """
## TASK: Kaggle House Prices Competition
Predict residential home sale prices (SalePrice) in Ames, Iowa using 79 housing features.

## CRITICAL: Evaluation Metric
Submissions are evaluated on **Root-Mean-Squared-Error (RMSE) between log(predicted) and log(actual)**.
This means:
1. **MUST train on log1p(SalePrice)** - transform target before training
2. Predictions will be expm1() transformed back to dollars
3. Errors on expensive and cheap houses affect score equally (log scale normalizes)
4. Focus on getting relative price ratios correct, not absolute values

## PROVEN FEATURE ENGINEERING (from top Kaggle solutions)

### 1. Size Features (Most Important Predictors)
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` (total living area)
- `log1p(GrLivArea)`, `log1p(TotalBsmtSF)`, `log1p(LotArea)` (log-transform skewed distributions)
- `TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`
- `TotalBaths = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`

### 2. Quality Features (2nd Most Important)
- **OverallQual is the #1 predictor** - use as-is AND create interactions
- Ordinal encode: ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC
  - Mapping: None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5
- Create `TotalQual = OverallQual + ExterQual_ord + KitchenQual_ord + BsmtQual_ord`

### 3. Critical Interaction Features
- `OverallQual * log1p(GrLivArea)` - quality-size interaction, very strong predictor
- `OverallQual * TotalSF` - another powerful interaction
- `Neighborhood_encoded * OverallQual` - location-quality interaction
- `YearBuilt * OverallQual` - newer high-quality homes command premium

### 4. Age Features
- `HouseAge = YrSold - YearBuilt` (age at time of sale)
- `RemodAge = YrSold - YearRemodAdd` (years since remodel)
- `IsRemodeled = (YearRemodAdd != YearBuilt).astype(int)` (was it remodeled?)
- `IsNew = (YrSold - YearBuilt <= 2).astype(int)` (new construction premium)

### 5. Neighborhood (Location Matters!)
- Neighborhood has ~25 categories with very different price levels
- Top neighborhoods: NoRidge, NridgHt, StoneBr have 2-3x higher prices
- Consider: target encoding or one-hot encoding
- Create `Neighborhood_median_price` as target-encoded feature (use training set only!)

### 6. Binary Flags
- `HasGarage = (GarageArea > 0).astype(int)`
- `HasBsmt = (TotalBsmtSF > 0).astype(int)`
- `Has2ndFloor = (2ndFlrSF > 0).astype(int)`
- `HasPool = (PoolArea > 0).astype(int)`
- `HasFireplace = (Fireplaces > 0).astype(int)`
- `CentralAir_bin = (CentralAir == 'Y').astype(int)`

### 7. Garage Features
- `GarageAge = YrSold - GarageYrBlt` (handle NA as no garage)
- `GarageFinish_ord`: None=0, Unf=1, RFn=2, Fin=3

### 8. Lot Features
- `log1p(LotFrontage)`, `log1p(LotArea)` - both are right-skewed
- `LotFrontage / LotArea` - lot shape ratio

## MISSING VALUE HANDLING (CRITICAL!)
In this dataset, NA often means "Not Applicable" NOT "Missing":
- PoolQC, MiscFeature, Alley, Fence: NA = "No pool/misc/alley/fence" → fill with "None"
- GarageType, GarageFinish, GarageQual, GarageCond: NA = "No garage" → fill with "None"
- BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2: NA = "No basement" → fill with "None"
- FireplaceQu: NA = "No fireplace" → fill with "None"
- MasVnrType, MasVnrArea: NA likely means "No masonry veneer" → fill with "None"/0
- LotFrontage: TRUE missing → impute with median by Neighborhood

Numeric NAs for basement/garage features (when house has no basement/garage):
- BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath → fill with 0
- GarageCars, GarageArea → fill with 0

## MODEL CONFIGURATION
**Use XGBoost with these hyperparameters (proven on this competition):**
- objective: 'reg:squarederror'
- learning_rate: 0.01-0.05 (lower is better with more trees)
- n_estimators: 1000-3000 with early_stopping_rounds: 50-100
- max_depth: 3-5 (shallow trees to avoid overfitting)
- min_child_weight: 3-5
- subsample: 0.8
- colsample_bytree: 0.7-0.8
- reg_alpha: 0.01-0.1 (L1 regularization)
- reg_lambda: 1-10 (L2 regularization)

**Key insight:** This is a small dataset (~1460 rows). Regularization is crucial to avoid overfitting.

## FEATURES TO EXCLUDE
- Id (identifier, no predictive value)
- Be careful with: MoSold, YrSold (can cause leakage if not used carefully)

## TARGET LEADERBOARD SCORE
Top 10% on Kaggle: log-RMSE < 0.12
Top 25%: log-RMSE < 0.13
Baseline (just OverallQual): log-RMSE ~0.18

Focus on feature engineering - that's what separates good scores from great scores in this competition.
""".strip()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step: str, msg: str = ""):
    """Print a step indicator."""
    print(f"\n[{step}] {msg}")


def preview_dataframe(df: pd.DataFrame, name: str, max_rows: int = 3):
    """Print a preview of a DataFrame."""
    print(f"\n  📊 {name}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print(f"  Dtypes summary: {df.dtypes.value_counts().to_dict()}")
    if max_rows > 0:
        print(f"  First {max_rows} rows:")
        print(df.head(max_rows).to_string(max_cols=8))


def apply_feature_engineering(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Apply feature engineering to match what the trained model expects.
    
    This function creates features with the exact names the model expects,
    stripping 'num__' prefix to get the raw column names.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    df = df.copy()
    
    # Extract the required column names (strip 'num__' and 'cat__' prefixes)
    required_cols = []
    for f in feature_names:
        if f.startswith('num__'):
            required_cols.append(f[5:])  # Remove 'num__' prefix
        elif f.startswith('cat__'):
            required_cols.append(f[5:])  # Remove 'cat__' prefix
        else:
            required_cols.append(f)
    
    print(f"  Model expects {len(required_cols)} input columns")
    
    # Fill missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Fill missing values for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna('None')
    
    # =========================================================================
    # Create engineered features with EXACT names the model expects
    # Based on the model's feature list
    # =========================================================================
    
    # Log transformations (match naming: log_ColName)
    if 'LotFrontage' in df.columns:
        df['log_LotFrontage'] = np.log1p(df['LotFrontage'].fillna(0))
    if 'LotArea' in df.columns:
        df['log_LotArea'] = np.log1p(df['LotArea'].fillna(0))
    if 'MasVnrArea' in df.columns:
        df['log_MasVnrArea'] = np.log1p(df['MasVnrArea'].fillna(0))
    if 'GrLivArea' in df.columns:
        df['log_GrLivArea'] = np.log1p(df['GrLivArea'].fillna(0))
    if '1stFlrSF' in df.columns:
        df['log_1stFlrSF'] = np.log1p(df['1stFlrSF'].fillna(0))
    if 'TotalBsmtSF' in df.columns:
        df['log_TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'].fillna(0))
    
    # Binning MSSubClass
    if 'MSSubClass' in df.columns:
        df['MSSubClass_binned'] = pd.cut(
            df['MSSubClass'], 
            bins=[-np.inf, 30, 60, 90, np.inf], 
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(0)
    
    # Computed features
    # Total square footage
    basement_sf = df['TotalBsmtSF'].fillna(0) if 'TotalBsmtSF' in df.columns else 0
    first_floor = df['1stFlrSF'].fillna(0) if '1stFlrSF' in df.columns else 0
    second_floor = df['2ndFlrSF'].fillna(0) if '2ndFlrSF' in df.columns else 0
    df['TotalSF'] = basement_sf + first_floor + second_floor
    
    # Age at sale
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['AgeAtSale'] = df['YrSold'] - df['YearBuilt']
    
    # Years since remodel
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
    
    # Total bathrooms
    full_bath = df['FullBath'].fillna(0) if 'FullBath' in df.columns else 0
    half_bath = df['HalfBath'].fillna(0) if 'HalfBath' in df.columns else 0
    bsmt_full = df['BsmtFullBath'].fillna(0) if 'BsmtFullBath' in df.columns else 0
    bsmt_half = df['BsmtHalfBath'].fillna(0) if 'BsmtHalfBath' in df.columns else 0
    df['TotalBaths'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half
    
    # Binary flags
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int) if 'GarageArea' in df.columns else 0
    df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int) if 'Fireplaces' in df.columns else 0
    df['Has2ndFloor'] = (df['2ndFlrSF'].fillna(0) > 0).astype(int) if '2ndFlrSF' in df.columns else 0
    df['HasPool'] = (df['PoolArea'].fillna(0) > 0).astype(int) if 'PoolArea' in df.columns else 0
    df['HasBsmt'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int) if 'TotalBsmtSF' in df.columns else 0
    
    # Interaction features
    if 'OverallQual' in df.columns and 'log_GrLivArea' in df.columns:
        df['OverallQual_x_logGrLivArea'] = df['OverallQual'] * df['log_GrLivArea']
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df['OverallQual_x_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    if 'OverallQual' in df.columns and 'TotalSF' in df.columns:
        df['OverallQual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
    if 'GarageCars' in df.columns and 'GrLivArea' in df.columns:
        df['GarageCars_x_GrLivArea'] = df['GarageCars'].fillna(0) * df['GrLivArea']
    
    # Ordinal encodings for quality columns (with _ord suffix)
    quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'None': 0}
    yes_no_map = {'N': 0, 'Y': 1}
    exposure_map = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3, 'None': 0}
    finish_map = {'Unf': 0, 'RFn': 1, 'Fin': 2, 'None': 0}
    paved_map = {'N': 0, 'P': 1, 'Y': 2}
    
    if 'ExterQual' in df.columns:
        df['ExterQual_ord'] = df['ExterQual'].map(quality_map).fillna(3)
    if 'ExterCond' in df.columns:
        df['ExterCond_ord'] = df['ExterCond'].map(quality_map).fillna(3)
    if 'BsmtQual' in df.columns:
        df['BsmtQual_ord'] = df['BsmtQual'].map(quality_map).fillna(0)
    if 'BsmtCond' in df.columns:
        df['BsmtCond_ord'] = df['BsmtCond'].map(quality_map).fillna(0)
    if 'HeatingQC' in df.columns:
        df['HeatingQC_ord'] = df['HeatingQC'].map(quality_map).fillna(3)
    if 'KitchenQual' in df.columns:
        df['KitchenQual_ord'] = df['KitchenQual'].map(quality_map).fillna(3)
    if 'GarageFinish' in df.columns:
        df['GarageFinish_ord'] = df['GarageFinish'].map(finish_map).fillna(0)
    if 'GarageQual' in df.columns:
        df['GarageQual_ord'] = df['GarageQual'].map(quality_map).fillna(0)
    if 'GarageCond' in df.columns:
        df['GarageCond_ord'] = df['GarageCond'].map(quality_map).fillna(0)
    if 'PavedDrive' in df.columns:
        df['PavedDrive_ord'] = df['PavedDrive'].map(paved_map).fillna(0)
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure_ord'] = df['BsmtExposure'].map(exposure_map).fillna(0)
    if 'CentralAir' in df.columns:
        df['CentralAir_bin'] = df['CentralAir'].map(yes_no_map).fillna(0)
    
    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the required columns in the right order
    return df[required_cols]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_kaggle_pipeline(use_existing_model: str = None):
    """Run the full Kaggle House Prices competition pipeline."""
    
    print_header("KAGGLE HOUSE PRICES COMPETITION")
    print("  Goal: Predict SalePrice for residential homes in Ames, Iowa")
    print("  Metric: RMSE on log(predicted) vs log(actual)")
    
    result = {}
    model_name = use_existing_model
    
    if use_existing_model:
        # Skip training, use existing model
        print_header("USING EXISTING MODEL")
        print(f"  Model: {use_existing_model}")
        model_name = use_existing_model
        
        # Verify model exists
        model_info = get_model_info(model_name)
        if not model_info:
            print(f"❌ Model '{model_name}' not found in registry!")
            return None
        
        print(f"  Type: {model_info.get('model_type')}")
        print(f"  Target: {model_info.get('target_column')}")
        print(f"  Features: {len(model_info.get('feature_names', []))}")
        
    else:
        # -------------------------------------------------------------------------
        # STEP 1: Load and Register Training Data
        # -------------------------------------------------------------------------
        print_header("STEP 1: Load and Register Training Data")
        
        if not TRAIN_FILE.exists():
            print(f"❌ Training file not found: {TRAIN_FILE}")
            return None
        
        df_train = pd.read_csv(TRAIN_FILE)
        print_step("LOADED", f"Training data from {TRAIN_FILE}")
        preview_dataframe(df_train, "Training Data")
        
        # Show target distribution
        print(f"\n  📈 Target Variable (SalePrice):")
        print(f"     Min: ${df_train['SalePrice'].min():,.0f}")
        print(f"     Max: ${df_train['SalePrice'].max():,.0f}")
        print(f"     Mean: ${df_train['SalePrice'].mean():,.0f}")
        print(f"     Median: ${df_train['SalePrice'].median():,.0f}")
        print(f"     Std: ${df_train['SalePrice'].std():,.0f}")
        
        # Log-transform target for better RMSE performance
        # The agent should handle this, but we can also do it here
        print_step("TRANSFORM", "Log-transforming SalePrice for RMSE optimization")
        df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
        print(f"  Log(SalePrice) range: {df_train['SalePrice'].min():.2f} to {df_train['SalePrice'].max():.2f}")
        
        # Register the dataset
        register_dataset("house_prices_train", df_train)
        print_step("REGISTERED", "Dataset registered as 'house_prices_train'")
        
        # -------------------------------------------------------------------------
        # STEP 2: Invoke Training Agent
        # -------------------------------------------------------------------------
        print_header("STEP 2: Invoke Training Agent")
        
        print_step("GOAL", "")
        print(f"  {GOAL[:200]}...")
        print_step("MODEL", "No preference - LLM will select based on goal")
        print_step("INVOKE", "Calling training agent with auto-approval...")
        print("\n" + "-" * 80)
        
        result = run_training_agent_auto_approve(
            goal=GOAL,
            linked_datasets=["house_prices_train"],
            user_model_preference="xgboost"  # Use XGBoost for tabular regression
        )
        
        print("-" * 80)
        
        # -------------------------------------------------------------------------
        # STEP 3: Review Results
        # -------------------------------------------------------------------------
        print_header("STEP 3: Training Results")
        
        print_step("SUMMARY", "")
        print(f"  Selected Model: {result.get('selected_model')}")
        model_explanation = result.get('model_explanation') or ''
        print(f"  Model Explanation: {model_explanation[:150]}..." if model_explanation else "  Model Explanation: N/A")
        print(f"  Collected Dataset: {result.get('collected_dataset_ref')}")
        print(f"  Cleaned Dataset: {result.get('cleaned_dataset_ref')}")
        print(f"  Train Dataset: {result.get('transformed_train_ref')}")
        print(f"  Model Weights: {result.get('model_weights_path')}")
        print(f"  Report Path: {result.get('report_path')}")
        
        training_metrics = result.get("training_metrics", {})
        if training_metrics:
            print_step("METRICS", "")
            print(f"  Success: {training_metrics.get('success')}")
            print(f"  Model Name: {training_metrics.get('model_name')}")
            print(f"  Iterations: {training_metrics.get('num_iterations', 0)}")
            
            # For regression, look for R² or RMSE
            val_acc = training_metrics.get('val_accuracy')
            if val_acc:
                print(f"  Val Score: {val_acc}")
            test_acc = training_metrics.get('test_accuracy')
            if test_acc:
                print(f"  Test Score: {test_acc}")
        
        # Check for errors
        if result.get("error"):
            print(f"\n❌ Error: {result.get('error')}")
            return result
        
        model_name = result.get("model_weights_path")
        if not model_name:
            print("\n❌ No model was trained!")
            return result
    
    # -------------------------------------------------------------------------
    # STEP 4: Load Test Data and Apply Feature Engineering
    # -------------------------------------------------------------------------
    print_header("STEP 4: Prepare Test Data")
    
    if not TEST_FILE.exists():
        print(f"❌ Test file not found: {TEST_FILE}")
        return result
    
    df_test_raw = pd.read_csv(TEST_FILE)
    print_step("LOADED", f"Test data from {TEST_FILE}")
    preview_dataframe(df_test_raw, "Test Data (Raw)")
    
    # Save the Id column for submission
    test_ids = df_test_raw["Id"].copy()
    
    # Load the trained model
    print_step("LOAD MODEL", f"Loading model '{model_name}'")
    
    try:
        model = load_model(model_name)
        model_info = get_model_info(model_name)
        print(f"  Model type: {model_info.get('model_type')}")
        print(f"  Features: {len(model_info.get('feature_names', []))} features")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return result
    
    # Get the feature names the model expects
    feature_names = model_info.get("feature_names", [])
    print(f"  Model expects features: {feature_names[:10]}...")
    
    # Apply the same feature engineering to test data
    print_step("FEATURE ENGINEERING", "Applying transformations to test data")
    
    df_test = df_test_raw.drop(columns=["Id"], errors="ignore").copy()
    
    # Apply the same feature engineering that was applied to training data
    # The function will strip 'num__' prefix and create matching columns
    X_test = apply_feature_engineering(df_test, feature_names)
    
    print(f"  Final test shape: {X_test.shape}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Generate Predictions
    # -------------------------------------------------------------------------
    print_header("STEP 5: Generate Predictions")
    
    print_step("PREDICT", "Generating predictions...")
    
    try:
        predictions = model.predict(X_test)
        print(f"  Generated {len(predictions)} predictions")
        
        # Reverse log transform (we log-transformed the target)
        predictions_dollars = np.expm1(predictions)
        print(f"  Price range: ${predictions_dollars.min():,.0f} to ${predictions_dollars.max():,.0f}")
        print(f"  Mean predicted price: ${predictions_dollars.mean():,.0f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    # -------------------------------------------------------------------------
    # STEP 6: Create Submission File
    # -------------------------------------------------------------------------
    print_header("STEP 6: Create Submission File")
    
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions_dollars
    })
    
    submission.to_csv(SUBMISSION_FILE, index=False)
    print_step("SAVED", f"Submission saved to {SUBMISSION_FILE}")
    
    preview_dataframe(submission, "Submission Preview", max_rows=5)
    
    print(f"\n  📁 File: {SUBMISSION_FILE}")
    print(f"  📊 Rows: {len(submission)}")
    print(f"  ✅ Ready to upload to Kaggle!")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print_header("PIPELINE COMPLETE")
    
    print(f"""
  📋 Summary:
     - Test samples: {len(df_test_raw)}
     - Model: {model_name}
     - Submission: {SUBMISSION_FILE}
  
  📤 Next Steps:
     1. Go to https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit
     2. Upload {SUBMISSION_FILE.name}
     3. Check your score!
  
  💡 Tips for improvement:
     - Try different models (run with user_model_preference="xgboost" or "random_forest")
     - Feature engineering: add more interaction terms
     - Handle missing values more carefully
     - Try ensemble methods
""")
    
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kaggle House Prices competition pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Only load data, don't train")
    parser.add_argument("--model", type=str, help="Use existing model instead of training new")
    args = parser.parse_args()
    
    if args.dry_run:
        print_header("DRY RUN - Data Loading Only")
        
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
        
        preview_dataframe(df_train, "Training Data")
        preview_dataframe(df_test, "Test Data")
        
        print(f"\n  SalePrice stats:")
        print(df_train["SalePrice"].describe())
    else:
        result = run_kaggle_pipeline(use_existing_model=args.model)
        
        if result is None:
            print("\n❌ Pipeline failed!")
        elif isinstance(result, dict) and result.get("training_metrics", {}).get("success"):
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n⚠️ Pipeline completed - check output above")
