import type { StepInfo, TrainingAgentState } from "@/types/agent"

/** Steps excluded from checklist progress (dots, X/Y) — still tracked internally for sync. */
export const CHECKLIST_EXCLUDE_IDS = new Set<string>([
  "generate_report",
  "select_model",
  /** Merged with `training` into one checklist row / dot in ProgressPanel */
  "training_approval",
])

/**
 * Steps not shown in the checklist for unsupervised runs: feature pipeline (skill preprocesses in
 * training) and label/split (no target or holdout — same dataset as Cleaning, nothing to inspect).
 */
export const UNSUPERVISED_HIDDEN_STEP_IDS = new Set<string>([
  "label_split_definition",
  "feature_specification_and_engineering",
  "feature_selection_specification",
  "feature_engineering_executor",
  "feature_experiment_runner",
])

/** Steps omitted from the execution trace list in the final report. */
export const TRACE_EXCLUDE_IDS = new Set<string>(["select_model"])

export function shouldHideFeatureStepsForUnsupervised(agentState: TrainingAgentState): boolean {
  return (
    agentState.selected_model === "unsupervised" ||
    agentState.label_definition?.split_strategy === "none"
  )
}

/** Checklist / sidebar: drop feature steps when the run is unsupervised — keeps full `steps` in the hook for stream sync. */
export function filterVisiblePipelineSteps(
  steps: StepInfo[],
  agentState: TrainingAgentState,
): StepInfo[] {
  if (!shouldHideFeatureStepsForUnsupervised(agentState)) return steps
  return steps.filter((s) => !UNSUPERVISED_HIDDEN_STEP_IDS.has(s.id))
}
