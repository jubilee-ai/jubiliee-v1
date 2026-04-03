/** Steps excluded from checklist progress (dots, X/Y) — still tracked internally for sync. */
export const CHECKLIST_EXCLUDE_IDS = new Set<string>(["generate_report", "select_model"])

/** Steps omitted from the execution trace list in the final report. */
export const TRACE_EXCLUDE_IDS = new Set<string>(["select_model"])
