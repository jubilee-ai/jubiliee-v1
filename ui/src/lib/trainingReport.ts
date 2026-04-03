import type { TrainingAgentState } from "@/types/agent"

/** Model name for the training report header and for `GET .../trained-models/{name}/report`. */
export function trainingReportModelLabel(state: TrainingAgentState): string | null {
  const m = state.training_metrics?.model_name
  if (typeof m === "string" && m.trim()) return m.trim()
  const p = state.model_weights_path
  if (typeof p === "string" && p.trim()) {
    const seg = p.split(/[/\\]/).pop() ?? p
    const stem = seg.replace(/\.(joblib|pkl|pt)$/i, "")
    return stem || null
  }
  return null
}

/**
 * Convert a stored report JSON (from the backend `/trained-models/{name}/report` endpoint)
 * into a partial `TrainingAgentState` that `FinalReport` can render.
 */
export function reportJsonToAgentState(report: Record<string, unknown>): TrainingAgentState {
  const model = (report.model ?? {}) as Record<string, unknown>
  const data = (report.data ?? {}) as Record<string, unknown>
  const labelDef = (report.label_definition ?? null) as Record<string, unknown> | null
  const tr = (report.training_results ?? {}) as Record<string, unknown>
  const valMetrics = (tr.validation_metrics ?? {}) as Record<string, unknown>
  const testMetrics = (tr.test_metrics ?? {}) as Record<string, unknown>

  return {
    goal: str(report.goal),
    linked_datasets: null,
    user_model_preference: null,
    selected_model: str(model.type),
    model_explanation: str(model.explanation),
    model_regen_count: 0,
    collected_dataset_ref: str(data.collected_dataset),
    cleaned_dataset_ref: str(data.cleaned_dataset),
    cleaning_transformations: [],
    cleaning_summary: null,
    label_definition: labelDef
      ? {
          target_column: str(labelDef.target_column) ?? "",
          split_strategy: str(labelDef.split_strategy) ?? "random",
          grain: str(labelDef.grain),
        }
      : null,
    split_indices: null,
    train_dataset_ref: null,
    val_dataset_ref: null,
    test_dataset_ref: null,
    feature_spec: null,
    analysis_trace: [],
    transformed_dataset_ref: null,
    transformed_train_ref: str(data.train_dataset),
    transformed_val_ref: str(data.val_dataset),
    transformed_test_ref: str(data.test_dataset),
    feature_validation_passed: false,
    feature_pipeline_mode: null,
    human_confirmed: true,
    training_params: null,
    model_weights_path: str(model.name),
    training_metrics: {
      success: tr.success === true,
      model_name: str(model.name),
      model_type: str(model.type),
      val_accuracy: num(valMetrics.accuracy),
      val_roc_auc: num(valMetrics.roc_auc),
      test_accuracy: num(testMetrics.accuracy),
      test_roc_auc: num(testMetrics.roc_auc),
      val_r2: num(valMetrics.r2),
      val_rmse: num(valMetrics.rmse),
      val_mae: num(valMetrics.mae),
      test_r2: num(testMetrics.r2),
      test_rmse: num(testMetrics.rmse),
      test_mae: num(testMetrics.mae),
      iterations: Array.isArray(tr.iterations) ? tr.iterations : [],
      num_iterations: typeof tr.num_iterations === "number" ? tr.num_iterations : undefined,
      best_iteration: tr.best_iteration as Record<string, unknown> | number | undefined,
      summary: str(tr.summary),
    },
    training_iteration: 0,
    feature_redo_requested: false,
    feature_redo_recommendation: null,
    feature_redo_reason: null,
    feature_redo_iteration: 0,
    report_path: str(report.report_path),
    audit_trace: Array.isArray(report.audit_trace) ? report.audit_trace : [],
    explanations: [],
    current_step: "generate_report",
    error: null,
  } as TrainingAgentState
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v.trim() : null
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined
}
