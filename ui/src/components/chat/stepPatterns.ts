// Step detection patterns - ordered from most specific to least specific
// Each entry has keywords that should be unique to that step

export const STEP_PATTERNS: Array<{ stepId: string; patterns: RegExp[] }> = [
  // Report - check first, very specific phrases
  {
    stepId: "generate_report",
    patterns: [
      /training pipeline complete/i,
      /report generated/i,
      /generate_report/i,
      /pipeline complete.*view report/i,
    ]
  },
  // Training Approval - configuration proposal
  {
    stepId: "training_approval",
    patterns: [
      /training config/i,
      /training configuration/i,
      /hyperparameters/i,
      /training plan/i,
      /training_approval/i,
      /training strategy/i,
    ]
  },
  // Training - specific training result patterns
  {
    stepId: "training",
    patterns: [
      /model trained/i,
      /test accuracy[:\s]/i,
      /test r[²2][:\s]/i,
      /roc-auc[:\s]/i,
      /training iteration/i,
      /model performance/i,
      /val_accuracy/i,
      /test_accuracy/i,
    ]
  },
  {
    stepId: "feature_specification_and_engineering",
    patterns: [
      /features specified.+engineered/i,
      /feature specification \+ engineering/i,
      /specify and materialize features/i,
    ],
  },
  // Feature Engineering - specific to execution
  {
    stepId: "feature_engineering_executor",
    patterns: [
      /features for training/i,
      /feature engineering complete/i,
      /features created/i,
      /features engineered/i,
      /feature engineering executor/i,
      /transformed dataset/i,
    ]
  },
  // Feature Selection - analysis phase
  {
    stepId: "feature_selection_specification",
    patterns: [
      /feature selection complete/i,
      /feature selection specification/i,
      /features specified/i,
      /feature analysis/i,
      /analyzing features/i,
      /correlation analysis/i,
    ]
  },
  // Label/Split Definition
  {
    stepId: "label_split_definition",
    patterns: [
      /label definition/i,
      /label.+split/i,
      /split definition/i,
      /target column[:\s]/i,
      /train\/val\/test/i,
      /split strategy/i,
      /70\/15\/15/i,
    ]
  },
  // Cleaning
  {
    stepId: "cleaning",
    patterns: [
      /cleaning complete/i,
      /data cleaning/i,
      /cleaning.+standardization/i,
      /transformations applied/i,
      /cleaned dataset/i,
      /missing values (filled|imputed|handled)/i,
    ]
  },
  // Data Collection
  {
    stepId: "data_collection",
    patterns: [
      /data collection complete/i,
      /dataset loaded/i,
      /loading dataset/i,
      /loaded.*rows/i,
      /\d+\s+rows.*\d+\s+columns/i,
    ]
  },
  // Model Selection - last because "model" is common
  {
    stepId: "select_model",
    patterns: [
      /model selection complete/i,
      /selected.*as the optimal model/i,
      /selected model[:\s]/i,
      /model:\s*(logistic_regression|random_forest|xgboost|gradient_boost|naive_bayes)/i,
      /choosing.*model/i,
    ]
  },
]

/**
 * Detects which step a message is about based on its content
 * @param content - The message content to analyze
 * @returns The step ID if detected, null otherwise
 */
export function detectStepFromMessage(content: string): string | null {
  // Don't make messages clickable if they already have "View Report" CTA
  // These are final completion messages that have the View Report button
  if (/view report/i.test(content) || /click.*report/i.test(content)) {
    return null
  }
  
  for (const { stepId, patterns } of STEP_PATTERNS) {
    if (patterns.some(pattern => pattern.test(content))) {
      return stepId
    }
  }
  return null
}

/**
 * Maps step IDs to keywords for finding related messages
 */
export const STEP_KEYWORDS: Record<string, string[]> = {
  "select_model": ["Model Selection", "selected model", "xgboost", "random_forest", "logistic_regression", "naive_bayes"],
  "data_collection": ["Data Collection", "Dataset loaded", "dataset"],
  "cleaning": ["Cleaning", "cleaned", "transformations"],
  "label_split_definition": ["Label", "Split", "target column"],
  "feature_selection_specification": ["Feature Selection", "features specified"],
  "feature_specification_and_engineering": ["Features", "specified", "engineered"],
  "feature_engineering_executor": ["Feature Engineering", "features created"],
  "training_approval": ["Training Config", "hyperparameters", "training plan", "training strategy"],
  "training": ["Training", "trained", "R²", "accuracy", "RMSE"],
  "generate_report": ["Report", "complete"],
}
