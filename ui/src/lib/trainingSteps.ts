import type { StepInfo, TrainingAgentState } from "@/types/agent"

/** Steps excluded from checklist progress (dots, X/Y) — still tracked internally for sync. */
export const CHECKLIST_EXCLUDE_IDS = new Set<string>(["generate_report", "select_model"])

/** Feature pipeline steps — not shown in the checklist for unsupervised runs (skill preprocesses in training). */
export const UNSUPERVISED_HIDDEN_STEP_IDS = new Set<string>([
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
