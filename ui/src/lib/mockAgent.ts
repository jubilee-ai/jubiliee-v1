/**
 * Mock training agent for demo/offline mode (useAgentState hook).
 */

import type { TrainingAgentState, StepInfo, ConfirmationRequest } from "@/types/agent"
import type { Dataset, ModelType } from "@/lib/api"

export const STEP_DEFINITIONS: Array<{ id: string; name: string; description: string }> = [
  { id: "data_collection", name: "Data Collection", description: "Load or collect the dataset" },
  { id: "select_model", name: "Model Selection", description: "Choose the ML model type for this task" },
  { id: "cleaning", name: "Cleaning", description: "Clean and standardize the data" },
  { id: "label_split_definition", name: "Label & Split", description: "Define target column and train/val/test splits" },
  { id: "feature_selection_specification", name: "Feature Selection", description: "Analyze data and specify features" },
  { id: "feature_engineering_executor", name: "Feature Engineering", description: "Execute feature transformations" },
  { id: "training_approval", name: "Training Config", description: "Propose hyperparameters and strategy" },
  { id: "training", name: "Training", description: "Train model and evaluate metrics" },
  { id: "generate_report", name: "Report", description: "Save the final training report" },
]

const STEP_ORDER = STEP_DEFINITIONS.map((s) => s.id)

export const AVAILABLE_DATASETS: Dataset[] = [
  { file: "csv/Loan_default.csv", format: "CSV", name: "Loan Default", description: "Loan default prediction dataset" },
  { file: "csv/insurance.csv", format: "CSV", name: "Insurance", description: "Insurance claims dataset" },
]

export const AVAILABLE_MODELS: ModelType[] = [
  { id: "logistic_regression", name: "Logistic Regression", description: "Binary/multiclass classification" },
  { id: "random_forest", name: "Random Forest", description: "Classification/regression" },
  { id: "xgboost", name: "XGBoost", description: "High-performance tabular data" },
  { id: "naive_bayes", name: "Naive Bayes", description: "Fast probabilistic classifier" },
  { id: "glm", name: "GLM", description: "Poisson/Gamma/Tweedie regression" },
  { id: "survival_analysis", name: "Survival Analysis", description: "Time-to-event prediction" },
]

export function createInitialState(): TrainingAgentState {
  return {
    goal: "",
    linked_datasets: null,
    user_model_preference: null,
    selected_model: null,
    model_explanation: null,
    model_regen_count: 0,
    collected_dataset_ref: null,
    cleaned_dataset_ref: null,
    cleaning_transformations: [],
    cleaning_summary: null,
    label_definition: null,
    split_indices: null,
    train_dataset_ref: null,
    val_dataset_ref: null,
    test_dataset_ref: null,
    feature_spec: null,
    analysis_trace: [],
    transformed_dataset_ref: null,
    transformed_train_ref: null,
    transformed_val_ref: null,
    transformed_test_ref: null,
    feature_validation_passed: false,
    human_confirmed: false,
    training_params: null,
    model_weights_path: null,
    training_metrics: null,
    training_iteration: 0,
    feature_redo_requested: false,
    feature_redo_recommendation: null,
    feature_redo_reason: null,
    feature_redo_iteration: 0,
    report_path: null,
    audit_trace: [],
    explanations: [],
    current_step: "idle",
    error: null,
  }
}

export function createInitialSteps(): StepInfo[] {
  return STEP_DEFINITIONS.map((def) => ({
    id: def.id,
    name: def.name,
    description: def.description,
    status: "pending" as const,
  }))
}

export function getNextStep(stepId: string): string | null {
  const idx = STEP_ORDER.indexOf(stepId)
  if (idx < 0 || idx >= STEP_ORDER.length - 1) return null
  return STEP_ORDER[idx + 1]
}

export function getConfirmationRequest(
  stepId: string,
  _state: TrainingAgentState,
  stepMetrics?: Record<string, string | number>,
  stepDetails?: string
): ConfirmationRequest {
  const def = STEP_DEFINITIONS.find((s) => s.id === stepId)
  return {
    step: stepId,
    stepName: def?.name ?? stepId,
    summary: stepDetails ?? `Step ${stepId} completed`,
    details: (stepMetrics ?? {}) as Record<string, unknown>,
  }
}

interface ExecuteStepResult {
  stepMetrics?: Record<string, string | number>
  stepDetails?: string
  confirmationRequired: boolean
  [key: string]: unknown
}

export async function executeStep(
  stepId: string,
  state: TrainingAgentState
): Promise<ExecuteStepResult> {
  // Simulate async work
  await new Promise((r) => setTimeout(r, 300))

  const def = STEP_DEFINITIONS.find((s) => s.id === stepId)
  const base: ExecuteStepResult = {
    stepMetrics: {},
    stepDetails: `Mock completion of ${def?.name ?? stepId}`,
    confirmationRequired: stepId === "select_model" || stepId === "data_collection",
  }

  switch (stepId) {
    case "select_model":
      return {
        ...base,
        selected_model: state.user_model_preference ?? "logistic_regression",
        model_explanation: "Mock: selected for demo",
      }
    case "data_collection":
      return {
        ...base,
        collected_dataset_ref: state.linked_datasets?.[0] ?? "mock_dataset",
      }
    case "cleaning":
      return { ...base, cleaned_dataset_ref: "mock_cleaned", cleaning_summary: "Mock cleaned" }
    case "label_split_definition":
      return {
        ...base,
        label_definition: {
          target_column: "target",
          prediction_horizon: null,
          grain: "row",
          as_of_cutoff: null,
          split_strategy: "random",
          forbidden_columns: [],
        },
        train_dataset_ref: "mock_train",
        val_dataset_ref: "mock_val",
        test_dataset_ref: "mock_test",
      }
    case "feature_selection_specification":
      return {
        ...base,
        feature_spec: { features: [] },
        analysis_trace: [],
      }
    case "feature_engineering_executor":
      return {
        ...base,
        transformed_train_ref: "mock_transformed_train",
        transformed_val_ref: "mock_transformed_val",
        transformed_test_ref: "mock_transformed_test",
        feature_validation_passed: true,
      }
    case "training_approval":
      return {
        ...base,
        training_params: {},
      }
    case "training":
      return {
        ...base,
        model_weights_path: "mock_model.joblib",
        training_metrics: {
          model_name: "mock_model",
          model_type: "logistic_regression",
          test_accuracy: 0.85,
          test_roc_auc: 0.9,
        },
      }
    case "generate_report":
      return {
        ...base,
        report_path: "/reports/mock_report.json",
      }
    default:
      return base
  }
}
