import type { TrainingAgentState, StepInfo, TrainingIteration } from "@/types/agent"
import { formatNumber, formatPercent } from "@/lib/utils"

/**
 * Helper to extract iteration metrics from various possible structures
 */
export function getIterationMetrics(iter: TrainingIteration): {
  train_accuracy?: number
  val_accuracy?: number
  val_roc_auc?: number
  val_r2?: number
  val_rmse?: number
  val_mae?: number
  test_r2?: number
  test_rmse?: number
  test_mae?: number
  model_name?: string
  tool?: string
  success?: boolean
  error?: string | null
} {
  // Check top-level properties first
  const topLevel = {
    train_accuracy: iter.train_accuracy,
    val_accuracy: iter.val_accuracy,
    val_roc_auc: iter.val_roc_auc,
    val_r2: iter.val_r2,
    val_rmse: iter.val_rmse,
    val_mae: iter.val_mae,
    test_r2: iter.test_r2,
    test_rmse: iter.test_rmse,
    test_mae: iter.test_mae,
    model_name: iter.model_name,
    tool: iter.tool,
    success: iter.success,
    error: iter.error,
  }

  // If metrics object exists, merge values (top-level takes precedence)
  const metricsObj = iter.metrics as Record<string, unknown> | undefined
  if (metricsObj) {
    return {
      train_accuracy: topLevel.train_accuracy ?? (metricsObj.train_accuracy as number | undefined),
      val_accuracy: topLevel.val_accuracy ?? (metricsObj.val_accuracy as number | undefined),
      val_roc_auc: topLevel.val_roc_auc ?? (metricsObj.roc_auc as number | undefined) ?? (metricsObj.val_roc_auc as number | undefined),
      val_r2: topLevel.val_r2 ?? (metricsObj.val_r2 as number | undefined),
      val_rmse: topLevel.val_rmse ?? (metricsObj.val_rmse as number | undefined),
      val_mae: topLevel.val_mae ?? (metricsObj.val_mae as number | undefined),
      test_r2: topLevel.test_r2 ?? (metricsObj.test_r2 as number | undefined),
      test_rmse: topLevel.test_rmse ?? (metricsObj.test_rmse as number | undefined),
      test_mae: topLevel.test_mae ?? (metricsObj.test_mae as number | undefined),
      model_name: topLevel.model_name,
      tool: topLevel.tool,
      success: topLevel.success,
      error: topLevel.error,
    }
  }

  return topLevel
}

/**
 * Render any value as a string safely
 */
export function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "N/A"
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "object") {
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

/**
 * Generate a text-based report for clipboard/export
 */
export function generateTextReport(state: TrainingAgentState, steps: StepInfo[]): string {
  const metrics = state.training_metrics
  const completedSteps = steps.filter((s) => s.status === "completed").length
  return `
TRAINING REPORT
===============

Goal: ${state.goal}
Model: ${metrics?.model_name} (${metrics?.model_type})
Report Path: ${state.report_path}

METRICS
-------
(Test metrics below are for the saved model "${metrics?.model_name || "N/A"}". The training summary may mention other runs that were not kept.)
Test Accuracy: ${formatPercent(metrics?.test_accuracy)}
Test ROC-AUC: ${formatNumber(metrics?.test_roc_auc, 3)}
Validation Accuracy: ${formatPercent(metrics?.val_accuracy)}
Validation ROC-AUC: ${formatNumber(metrics?.val_roc_auc, 3)}

PIPELINE
--------
Steps Completed: ${completedSteps}/${steps.length}

DATA
----
Dataset: ${state.collected_dataset_ref}
Target Column: ${state.label_definition?.target_column}
Split Strategy: ${state.label_definition?.split_strategy}

FEATURE DEFINITIONS (logical, ${state.feature_spec?.features.length || 0})
One-hot and similar encodings expand to more model input columns than this list. Prefer training approval / data summary n_features when present.
${state.feature_spec?.features.map((f) => `- ${f.name}: ${f.formula}`).join("\n") || "None"}

TRAINING ITERATIONS
-------------------
${metrics?.iterations?.map((it) => `Iteration ${it.iteration}: Accuracy=${formatPercent(it.val_accuracy)}, AUC=${formatNumber(it.val_roc_auc, 3)}`).join("\n") || "None"}
Best Iteration: ${metrics?.best_iteration}

Generated: ${new Date().toISOString()}
`.trim()
}

/**
 * Generate a JSON report for download/export
 */
export function generateJsonReport(state: TrainingAgentState, steps: StepInfo[]): object {
  return {
    generated_at: new Date().toISOString(),
    goal: state.goal,
    model: {
      name: state.training_metrics?.model_name,
      type: state.training_metrics?.model_type,
      explanation: state.model_explanation,
    },
    metrics: {
      test: {
        accuracy: state.training_metrics?.test_accuracy,
        roc_auc: state.training_metrics?.test_roc_auc,
        r2: state.training_metrics?.test_r2,
        rmse: state.training_metrics?.test_rmse,
        mae: state.training_metrics?.test_mae,
      },
      validation: {
        accuracy: state.training_metrics?.val_accuracy,
        roc_auc: state.training_metrics?.val_roc_auc,
        r2: state.training_metrics?.val_r2,
        rmse: state.training_metrics?.val_rmse,
        mae: state.training_metrics?.val_mae,
      },
      iterations: state.training_metrics?.iterations,
      best_iteration: state.training_metrics?.best_iteration,
      summary: state.training_metrics?.summary,
    },
    data: {
      collected_dataset: state.collected_dataset_ref,
      cleaned_dataset: state.cleaned_dataset_ref,
      train_dataset: state.transformed_train_ref,
      val_dataset: state.transformed_val_ref,
      test_dataset: state.transformed_test_ref,
    },
    label_definition: state.label_definition,
    feature_spec: state.feature_spec,
    cleaning_transformations: state.cleaning_transformations,
    audit_trace: state.audit_trace,
    steps: steps.map((s) => ({
      id: s.id,
      name: s.name,
      status: s.status,
      duration_ms: s.endTime && s.startTime ? s.endTime - s.startTime : null,
    })),
  }
}
