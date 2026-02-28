// Types matching TrainingAgentState from agents/training/agent.py

export interface LabelDefinition {
  target_column: string
  prediction_horizon: string | null
  grain: string
  as_of_cutoff: string | null
  split_strategy: "random" | "time_based" | "entity_based"
  forbidden_columns: string[]
}

export interface FeatureSpec {
  features: Array<{
    name: string
    formula: string
    source_tables?: string[]
    window?: string | null
    grain?: string
    as_of_constraint?: string | null
    encoding: string
  }>
}

export interface TrainingIteration {
  iteration?: number
  model_name?: string
  tool?: string
  params?: Record<string, unknown>
  hyperparams?: Record<string, unknown>
  metrics?: Record<string, unknown>
  // Classification metrics
  val_accuracy?: number
  val_roc_auc?: number
  train_accuracy?: number
  // Regression metrics
  val_r2?: number
  val_rmse?: number
  val_mae?: number
  test_r2?: number
  test_rmse?: number
  test_mae?: number
  train_r2?: number
  train_rmse?: number
  train_mae?: number
  success?: boolean
  error?: string | null
}

export interface TrainingMetrics {
  success?: boolean
  model_name?: string
  model_type?: string
  
  // Classification metrics
  val_accuracy?: number
  val_roc_auc?: number
  test_accuracy?: number
  test_roc_auc?: number
  
  // Regression metrics
  val_r2?: number
  val_rmse?: number
  val_mae?: number
  test_r2?: number
  test_rmse?: number
  test_mae?: number
  train_r2?: number
  train_rmse?: number
  train_mae?: number
  
  // Iterations
  iterations?: TrainingIteration[]
  num_iterations?: number
  best_iteration?: number
  
  // Summary and recommendations
  summary?: string
  recommendations?: string
  feature_redo_requested?: boolean
}

// Feature Analysis Types
export interface FeatureCorrelation {
  feature: string
  correlation: number
}

export interface HighCorrelationPair {
  feature1: string
  feature2: string
  correlation: number
}

export interface NumericSummary {
  column: string
  mean?: number
  median?: number
  std?: number
  min?: number
  max?: number
  p5?: number
  p95?: number
  skew?: number
  outliers_pct?: number
}

export interface HistogramBin {
  range: string
  count: number
  pct: number
}

export interface DistributionStat {
  column: string
  n?: number
  n_missing?: number
  mean?: number
  std?: number
  min?: number
  max?: number
  skewness?: number
  kurtosis?: number
  is_normal?: string
  shape?: string
  modality?: string
  histogram?: HistogramBin[]
  percentiles?: Record<string, number>
}

export interface GroupData {
  value: string
  count?: number
  mean?: number
}

export interface GroupSummary {
  column: string
  n_groups: number
  total_rows?: number
  groups: GroupData[]
  overall_mean?: number
}

export interface LorenzPoint {
  pct_entities: number
  pct_value: number
}

export interface ConcentrationStat {
  column: string
  gini?: number
  gini_interpretation?: string
  mean?: number
  median?: number
  top_1pct_share?: number
  top_10pct_share?: number
  top_50pct_share?: number
  pareto_80pct?: number
  lorenz_curve?: LorenzPoint[]
}

export interface FeatureHealth {
  column: string
  null_pct?: number
  skew?: number
  leakage_risk?: string
  redundancy_group?: string
}

export interface CorrelationMatrix {
  columns: string[]
  matrix: Record<string, Record<string, number>>
}

export interface KeyStats {
  dataset_overview: {
    rows?: number
    columns?: number
    numeric_columns?: number
    categorical_columns?: number
  }
  target_analysis: {
    type?: "numeric" | "categorical"
    column?: string
    mean?: number
    std?: number
    min?: number
    max?: number
    median?: number
    unique_values?: number
    top_value?: string
    top_freq?: number
  }
  schema?: Array<{
    column: string
    dtype: string
    null_pct: number
    unique?: number
  }>
  numeric_summaries: NumericSummary[]
  feature_correlations: FeatureCorrelation[]
  correlation_matrix?: CorrelationMatrix
  high_correlation_pairs: HighCorrelationPair[]
  distribution_stats: DistributionStat[]
  feature_health?: FeatureHealth[]
  diagnostics_summary?: {
    features_checked: number
    drop: number
    transform: number
    keep_as_is: number
  }
  leakage_warnings: string[]
  categorical_summaries: Array<{
    column: string
    groups: string[]
    group_count: number
  }>
  group_summaries?: GroupSummary[]
  concentration_analysis: ConcentrationStat[]
  feature_importances?: Array<{feature: string; importance: number}>
  summary_text: string
}

export interface FeatureAnalysisTrace {
  step: string
  analysis_results?: Record<string, unknown>
  key_stats?: KeyStats
  validation?: Record<string, unknown>
  is_redo?: boolean
  redo_recommendation?: string
}

export interface AuditTraceItem {
  step: string
  [key: string]: unknown
}

export interface TrainingAgentState {
  // Inputs
  goal: string
  linked_datasets: string[] | null
  user_model_preference: string | null

  // Step 1: Model Selection
  selected_model: string | null
  model_explanation: string | null
  model_regen_count: number

  // Step 2: Data Collection
  collected_dataset_ref: string | null

  // Step 3: Cleaning & Standardization
  cleaned_dataset_ref: string | null
  cleaning_transformations: Array<Record<string, unknown>>
  cleaning_summary: string | null

  // Step 3.5: Label & Split Definition
  label_definition: LabelDefinition | null
  split_indices: Record<string, unknown> | null
  train_dataset_ref: string | null
  val_dataset_ref: string | null
  test_dataset_ref: string | null

  // Step 4: Feature Selection & Specification
  feature_spec: FeatureSpec | null
  analysis_trace: Array<Record<string, unknown>>

  // Step 5: Feature Engineering
  transformed_dataset_ref: string | null
  transformed_train_ref: string | null
  transformed_val_ref: string | null
  transformed_test_ref: string | null
  feature_validation_passed: boolean

  // Step 6: Human Confirmation
  human_confirmed: boolean

  // Step 7: Training
  training_params: Record<string, unknown> | null
  model_weights_path: string | null
  training_metrics: TrainingMetrics | null
  training_iteration: number

  // Feature Engineering Redo
  feature_redo_requested: boolean
  feature_redo_recommendation: string | null
  feature_redo_reason: string | null
  feature_redo_iteration: number

  // Step 8: Report
  report_path: string | null

  // Audit & Trace
  audit_trace: AuditTraceItem[]
  explanations: string[]

  // Control flow
  current_step: string
  error: string | null
}

// UI-specific types
export type StepStatus = "pending" | "running" | "awaiting_confirmation" | "completed" | "error" | "skipped" | "stale"

export interface StepInfo {
  id: string
  name: string
  description: string
  status: StepStatus
  metrics?: Record<string, string | number>
  details?: string
  subtitle?: string
  runCount?: number
  startTime?: number
  endTime?: number
}

export interface ChatMessage {
  id: string
  role: "user" | "agent" | "system"
  content: string
  timestamp: number
  /** True while the message is still receiving streamed tokens */
  _streaming?: boolean
  links?: Array<{
    type: "dataset" | "model" | "step"
    id: string
    label: string
  }>
}

export interface Dataset {
  name: string
  description: string
  file: string
  rows?: number
  columns?: string[]
  format: string
}

export interface TrainedModel {
  model_name: string
  model_type: string
  description?: string
  metrics?: {
    train_accuracy?: number
    test_accuracy?: number
    test_roc_auc?: number
  }
}

// Human-in-the-loop actions
export type ConfirmationAction = "accept" | "redo" | "accept_all"

export interface ConfirmationRequest {
  step: string
  stepName: string
  summary: string
  details: Record<string, unknown>
}
