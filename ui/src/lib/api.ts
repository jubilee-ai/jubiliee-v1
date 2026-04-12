/**
 * API client for the Jubilee Training Agent FastAPI backend.
 */

import {
  USE_MODEL_RISK_MOCK,
  getMockModelRiskDetail,
  getMockModelRiskDashboard,
  getMockModelRiskInventory,
} from "@/lib/modelRiskMock"

const API_BASE = "" // Relative; proxied by nginx in Docker or same-origin in dev

export interface Dataset {
  id?: string
  name: string
  file?: string
  storage_key?: string
  source_type?: string
  trainable?: boolean
  format?: string
  rows?: number
  columns?: string[]
  description?: string
  use_case?: string
}

export interface DatasetPreview {
  name: string
  limit: number
  returned: number
  columns: string[]
  rows: Array<Record<string, unknown>>
}

export interface ModelType {
  id: string
  name: string
  description?: string
}

export interface TrainedModelEntry {
  model_name: string
  model_type: string
  description?: string
  metrics: Record<string, number>
  feature_names: string[]
  target_column: string
  hyperparameters: Record<string, unknown>
  training_samples: number
  classes: string[]
  created_at: string
  updated_at: string
  version: number
  experiment_id?: string | null
  experiment_name?: string | null
  /** True when `trained_models/{model_name}_report.json` exists on the server. */
  report_available?: boolean
}

/** Unified SSE payloads from POST /api/chat (orchestrator and/or training graph). */
export interface AgentStreamEvent {
  type: string
  experiment_id?: string
  ts?: string

  // Token streaming
  content?: string

  // Tool events
  tool?: string
  args?: Record<string, unknown>
  result?: string
  headline?: string

  // Pipeline step events
  node?: string
  progress?: number
  summary?: unknown
  details?: unknown
  state?: Record<string, unknown>

  // HITL
  state_snapshot?: Record<string, unknown>
  review_prompt?: string

  // Prediction
  model?: string
  rows_predicted?: number
  result_ref?: string

  // Dataset
  ref?: string
  dataset?: string

  // Dedup
  stream_step_key?: string

  // Legacy (keep during migration)
  thread_id?: string
  phase?: string
  message?: string

  // Pipeline completion flag
  pipeline_completed?: boolean

  /** True only for LangGraph training pipeline streams (not orchestrator chat). */
  training_graph?: boolean

  // Error
  error?: string

  /** task_plan.proposed (background intake, no tools) */
  goal?: string
  dataset_refs?: string[]
  dataset_labels?: string[]
  preferences?: string | null
  recap_steps?: string[]
}

export interface AgentStreamRequest {
  message: string
  experiment_id?: string
  linked_datasets?: string[] | null
  model_preference?: string | null
  /** Prior lab chat turns so the training planner can synthesize from the full dialogue */
  conversation?: Array<{ role: string; content: string }>
  resume?: {
    approved: boolean
    feedback?: string
  }
  /** Keep orchestrator/tools (e.g. propose_training_plan); do not start the training graph. */
  force_orchestrator?: boolean
  /** Tool-free planning agent; server emits task_plan.proposed instead of propose_training_plan. */
  background_intake?: boolean
}


export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`)
    const data = await res.json()
    return data?.status === "ok"
  } catch {
    return false
  }
}

export async function getDatasets(): Promise<Dataset[]> {
  const res = await fetch(`${API_BASE}/api/datasets`)
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.status}`)
  return res.json()
}

const DATASET_PREVIEW_LIMIT = 25

export async function getDatasetPreview(
  name: string,
  opts?: { limit?: number },
): Promise<DatasetPreview> {
  const limit = opts?.limit ?? DATASET_PREVIEW_LIMIT
  const res = await fetch(
    `${API_BASE}/api/datasets/${encodeURIComponent(name)}/preview?limit=${limit}`,
  )
  if (!res.ok) {
    let detail = `Preview failed (${res.status})`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === "string") detail = body.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function uploadDataset(
  file: File,
  opts?: { name?: string; description?: string },
): Promise<Dataset> {
  const form = new FormData()
  form.append("file", file)
  if (opts?.name?.trim()) form.append("name", opts.name.trim())
  if (opts?.description?.trim()) form.append("description", opts.description.trim())
  const res = await fetch(`${API_BASE}/api/datasets/upload`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) {
    let detail = `Upload failed (${res.status})`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === "string") detail = body.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function getModelTypes(): Promise<ModelType[]> {
  const res = await fetch(`${API_BASE}/api/models`)
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`)
  return res.json()
}

/** Trained model registry entries keyed by model name. */
export async function getTrainedModels(): Promise<Record<string, TrainedModelEntry>> {
  const res = await fetch(`${API_BASE}/api/trained-models`)
  if (!res.ok) throw new Error(`Failed to fetch trained models: ${res.status}`)
  return res.json()
}

export async function getTrainedModelReport(modelName: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/trained-models/${encodeURIComponent(modelName)}/report`)
  if (!res.ok) throw new Error(`Failed to fetch trained model report: ${res.status}`)
  return res.json()
}

function parseSSELine(line: string): unknown {
  if (line.startsWith("data: ")) {
    const json = line.slice(6).trim()
    if (json === "[DONE]" || json === "") return null
    try {
      return JSON.parse(json) as unknown
    } catch {
      return null
    }
  }
  return null
}

// =========================================================================
// Experiments API
// =========================================================================

export interface ExperimentSummary {
  id: string
  name: string
  goal?: string | null
  status: string
  chat_thread_id: string
  linked_datasets?: string[] | null
  created_at?: string | null
  updated_at?: string | null
  last_message?: string | null
  lab_mode?: string | null
  task_status?: string | null
}

export interface ExperimentDetail extends ExperimentSummary {
  chat_history: Array<Record<string, unknown>>
  training_state?: Record<string, unknown> | null
  training_context?: Record<string, unknown> | null
}

export async function createExperiment(
  name?: string,
  linked_datasets?: string[],
): Promise<ExperimentSummary> {
  const res = await fetch(`${API_BASE}/api/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, linked_datasets }),
  })
  if (!res.ok) throw new Error(`Failed to create experiment: ${res.status}`)
  return res.json()
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const res = await fetch(`${API_BASE}/api/experiments`)
  if (!res.ok) throw new Error(`Failed to list experiments: ${res.status}`)
  return res.json()
}

export async function getExperiment(id: string): Promise<ExperimentDetail> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}`)
  if (!res.ok) throw new Error(`Failed to get experiment: ${res.status}`)
  return res.json()
}

export async function updateExperiment(
  id: string,
  updates: {
    name?: string
    status?: string
    goal?: string
    linked_datasets?: string[]
    training_state_merge?: Record<string, unknown>
  },
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
  if (!res.ok) throw new Error(`Failed to update experiment: ${res.status}`)
}

export async function startExperimentAsyncTrain(
  experimentId: string,
  body?: { user_model_preference?: string | null; conversation?: Array<{ role: string; content: string }> },
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${experimentId}/async-train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => "")
    throw new Error(`Async train failed: ${res.status}${detail ? ` ${detail}` : ""}`)
  }
}

export async function deleteExperiment(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error(`Failed to delete experiment: ${res.status}`)
}

// =========================================================================
// Model Risk Management (demo API)
// =========================================================================

export type ModelRiskTier = "low" | "medium" | "high" | "critical"
export type ModelRiskPortalRole = "owner" | "validation" | "board"
export type ModelRiskMonitoringStatus = "healthy" | "watch" | "breach" | "unknown"

export interface ModelRiskInventoryItem {
  model_name: string
  model_type?: string
  risk_tier: ModelRiskTier
  lifecycle_stage: string
  monitoring_status: ModelRiskMonitoringStatus
  governance_profile: string
  last_reviewed_at?: string | null
  /** One-line summary for registry list (demo / optional). */
  intended_use_summary?: string
  next_review_due?: string | null
}

export type ModelRiskLifecycleNodeStatus = "complete" | "current" | "pending" | "skipped"

export interface ModelRiskLifecycleStage {
  id: string
  label: string
  status: ModelRiskLifecycleNodeStatus
  entered_at?: string | null
  notes?: string | null
}

export type ModelRiskApprovalStatus = "pending" | "approved" | "rejected"

export interface ModelRiskApproval {
  id: string
  stage: string
  status: ModelRiskApprovalStatus
  requested_at: string
  decided_at?: string | null
  actor_role?: string | null
  comment?: string | null
}

export interface ModelRiskPsiCsiRow {
  feature: string
  psi: number
  csi?: number | null
}

export interface ModelRiskBacktestRow {
  period: string
  metric: string
  value: number
  benchmark?: number | null
  pass: boolean
}

export interface ModelRiskMonitoring {
  status: ModelRiskMonitoringStatus
  as_of: string
  psi_csi: ModelRiskPsiCsiRow[]
  backtest: ModelRiskBacktestRow[]
  narrative?: string | null
}

export interface ModelRiskDocument {
  id: string
  title: string
  kind: string
  href?: string | null
  updated_at?: string | null
}

export interface ModelRiskReview {
  id: string
  review_type: string
  scheduled_for?: string | null
  completed_at?: string | null
  outcome?: string | null
  owner?: string | null
}

export interface ModelRiskMonitoringPipeline {
  id: string
  name: string
  schedule_cron: string
  last_run_at: string
  status: "ok" | "warning" | "failed"
}

export interface ModelRiskInferenceCall {
  id: string
  at: string
  input_preview: Record<string, string | number | boolean>
  output_preview: Record<string, string | number | boolean>
  latency_ms: number
}

export interface ModelRiskPerformanceKpi {
  label: string
  value: number
  unit?: string
  window?: string
  trend?: "up" | "down" | "flat"
}

export interface ModelRiskLineDefenseDocs {
  owner: ModelRiskDocument
  mrm: ModelRiskDocument
  audit: ModelRiskDocument
}

export interface ModelRiskProfileDetail {
  model_name: string
  model_type?: string
  risk_tier: ModelRiskTier
  governance_profile: string
  lifecycle_stage: string
  lifecycle_stages: ModelRiskLifecycleStage[]
  monitoring: ModelRiskMonitoring
  approvals: ModelRiskApproval[]
  documents: ModelRiskDocument[]
  reviews: ModelRiskReview[]
  metadata?: Record<string, unknown>
  /** Business purpose and boundaries (MRM profile). */
  intended_use?: string
  /** Short line for registry list cards; falls back to truncated intended_use. */
  intended_use_summary?: string
  usage_guidance?: string
  prohibited_use?: string
  monitoring_pipelines?: ModelRiskMonitoringPipeline[]
  inference_log?: ModelRiskInferenceCall[]
  performance_kpis?: ModelRiskPerformanceKpi[]
  line_defense_docs?: ModelRiskLineDefenseDocs
}

export interface ModelRiskDashboardItem {
  model_name: string
  snippet: string
  risk_tier: ModelRiskTier
  monitoring_status: ModelRiskMonitoringStatus
}

export interface ModelRiskDashboard {
  portal: ModelRiskPortalRole
  headline: string
  counts: {
    models: number
    pending_approvals: number
    monitoring_watch: number
    breaches: number
  }
  items: ModelRiskDashboardItem[]
}

export interface ModelRiskApprovalRequest {
  approval_id: string
  decision: "approved" | "rejected"
  comment?: string
}

export interface ModelRiskProfilePatch {
  risk_tier?: ModelRiskTier
  governance_profile?: string
  lifecycle_stage?: string
}

export async function getModelRiskInventory(): Promise<ModelRiskInventoryItem[]> {
  if (USE_MODEL_RISK_MOCK) return getMockModelRiskInventory()
  const res = await fetch(`${API_BASE}/api/model-risk/inventory`)
  if (!res.ok) throw new Error(`Model risk inventory failed: ${res.status}`)
  return res.json()
}

export async function getModelRiskDetail(modelName: string): Promise<ModelRiskProfileDetail> {
  if (USE_MODEL_RISK_MOCK) return getMockModelRiskDetail(modelName)
  const res = await fetch(
    `${API_BASE}/api/model-risk/models/${encodeURIComponent(modelName)}`,
  )
  if (!res.ok) throw new Error(`Model risk detail failed: ${res.status}`)
  return res.json()
}

export async function getModelRiskDashboard(
  portal: ModelRiskPortalRole = "owner",
): Promise<ModelRiskDashboard> {
  if (USE_MODEL_RISK_MOCK) return getMockModelRiskDashboard(portal)
  const q = new URLSearchParams({ portal })
  const res = await fetch(`${API_BASE}/api/model-risk/dashboard?${q}`)
  if (!res.ok) throw new Error(`Model risk dashboard failed: ${res.status}`)
  return res.json()
}

export async function postModelRiskApproval(
  modelName: string,
  body: ModelRiskApprovalRequest,
): Promise<void> {
  if (USE_MODEL_RISK_MOCK) return
  const res = await fetch(
    `${API_BASE}/api/model-risk/models/${encodeURIComponent(modelName)}/approvals`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  )
  if (!res.ok) throw new Error(`Model risk approval failed: ${res.status}`)
}

export async function patchModelRiskProfile(
  modelName: string,
  body: ModelRiskProfilePatch,
): Promise<void> {
  if (USE_MODEL_RISK_MOCK) return
  const res = await fetch(
    `${API_BASE}/api/model-risk/models/${encodeURIComponent(modelName)}/profile`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  )
  if (!res.ok) throw new Error(`Model risk profile update failed: ${res.status}`)
}

export async function saveExperimentMessages(
  id: string,
  messages: Array<Record<string, unknown>>,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}/messages`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  })
  if (!res.ok) throw new Error(`Failed to save messages: ${res.status}`)
}

// =========================================================================
// Chat streaming (with experiment_id support)
// =========================================================================

export function streamChat(
  req: AgentStreamRequest,
  onEvent: (event: AgentStreamEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController()
  const body = JSON.stringify(req)

  fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Chat stream failed: ${res.status}`)
      const reader = res.body?.getReader()
      if (!reader) throw new Error("No response body")
      const decoder = new TextDecoder()
      let buffer = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""
        for (const line of lines) {
          const parsed = parseSSELine(line)
          if (parsed) onEvent(parsed as AgentStreamEvent)
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(err instanceof Error ? err : new Error(String(err)))
    })

  return controller
}
