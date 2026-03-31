import { toast } from "sonner"

const AUTO_ACCEPT_TOAST_ID = "jubilee-auto-accept"

/** After the user explicitly accepts a pipeline step. */
export function toastStepAccepted() {
  toast.success("Step accepted — training continues", {
    description: "You can leave this page. This experiment stays in the sidebar.",
    duration: 6500,
  })
}

/** After the user chooses accept-all for remaining steps. */
export function toastAcceptAllMode() {
  toast.success("Auto-approving the rest", {
    description: "We’ll keep going without pausing. Come back anytime — this run stays in the sidebar.",
    duration: 7500,
  })
}

/** Shown when the stream auto-resumes after accept-all (deduped so it doesn’t spam). */
export function toastAutoAcceptContinued() {
  toast.message("Still running — safe to step away", {
    id: AUTO_ACCEPT_TOAST_ID,
    description: "Jubilee is auto-approving steps. Progress is saved on this experiment.",
    duration: 4500,
  })
}

/** After starting an async / background training run from the plan card. */
export function toastBackgroundRunStarted() {
  toast.success("Background run started", {
    description: "Training continues off the chat screen. Watch progress under this experiment in the sidebar (moon icon).",
    duration: 8000,
  })
}
