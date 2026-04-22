/**
 * API client for the Jubilee Training Agent FastAPI backend.
 */

const API_BASE = "" // Relative; proxied by nginx in Docker or same-origin in dev

// ---------------------------------------------------------------------------
// Auth token integration
// ---------------------------------------------------------------------------

type TokenGetter = () => Promise<string | null>
let _getToken: TokenGetter | null = null

/** Called from SignedInWithOrg to provide the Clerk getToken function. */
export function setTokenGetter(fn: TokenGetter): void {
  _getToken = fn
}

/**
 * Wrapper around fetch that attaches the Clerk JWT as a Bearer token.
 * Falls through to plain fetch for unauthenticated endpoints (e.g. health).
 */
async function authFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers)
  if (_getToken) {
    try {
      let token = await _getToken()
      if (!token) {
        await new Promise((r) => setTimeout(r, 75))
        token = await _getToken()
      }
      if (token) headers.set("Authorization", `Bearer ${token}`)
    } catch {
      /* token retrieval failed; proceed without auth */
    }
  }
  return fetch(input, { ...init, headers })
}

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
  dataset_info?: Record<string, unknown>

  // Orchestrator analysis sidecars (`analysis.result`, `analysis.chart`)
  kind?: string
  payload?: unknown
  spec?: Record<string, unknown>

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
  /** `"chat"` forces orchestrator; `"train"` forces the training graph. */
  mode?: "chat" | "train"
  /** Resume a specific training thread after a HITL interrupt. */
  resume_training?: {
    thread_id: string
    approved: boolean
    feedback?: string
  }
  /** Tool-free planning agent; server emits task_plan.proposed instead of propose_training_plan. */
  background_intake?: boolean

  // ---- Legacy fields (deprecated) -----------------------------------------
  /** @deprecated prefer `mode: "chat"`. */
  force_orchestrator?: boolean
  /** @deprecated prefer `resume_training` with the explicit thread id. */
  resume?: {
    approved: boolean
    feedback?: string
  }
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
  const res = await authFetch(`${API_BASE}/api/datasets`)
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.status}`)
  return res.json()
}

const DATASET_PREVIEW_LIMIT = 25

export async function getDatasetPreview(
  name: string,
  opts?: { limit?: number },
): Promise<DatasetPreview> {
  const limit = opts?.limit ?? DATASET_PREVIEW_LIMIT
  const res = await authFetch(
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
  const res = await authFetch(`${API_BASE}/api/datasets/upload`, {
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
  const res = await authFetch(`${API_BASE}/api/models`)
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`)
  return res.json()
}

/** Trained model registry entries keyed by model name. */
export async function getTrainedModels(): Promise<Record<string, TrainedModelEntry>> {
  const res = await authFetch(`${API_BASE}/api/trained-models`)
  if (!res.ok) throw new Error(`Failed to fetch trained models: ${res.status}`)
  return res.json()
}

export async function getTrainedModelReport(modelName: string): Promise<Record<string, unknown>> {
  const res = await authFetch(`${API_BASE}/api/trained-models/${encodeURIComponent(modelName)}/report`)
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
  const res = await authFetch(`${API_BASE}/api/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, linked_datasets }),
  })
  if (!res.ok) throw new Error(`Failed to create experiment: ${res.status}`)
  return res.json()
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  const res = await authFetch(`${API_BASE}/api/experiments`)
  if (!res.ok) throw new Error(`Failed to list experiments: ${res.status}`)
  return res.json()
}

export async function getExperiment(id: string): Promise<ExperimentDetail> {
  const res = await authFetch(`${API_BASE}/api/experiments/${id}`)
  if (!res.ok) throw new Error(`Failed to get experiment: ${res.status}`)
  return res.json()
}

export interface ExperimentArtifactsResponse {
  datasets: Dataset[]
  models: Array<{
    model_name: string
    model_type?: string
    version?: number
    metrics?: Record<string, number>
    storage_key?: string | null
    is_current?: boolean
    created_at?: string
  }>
}

export async function getExperimentArtifacts(
  experimentId: string,
): Promise<ExperimentArtifactsResponse> {
  const res = await authFetch(`${API_BASE}/api/experiments/${experimentId}/artifacts`)
  if (!res.ok) throw new Error(`Failed to load experiment artifacts: ${res.status}`)
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
  const res = await authFetch(`${API_BASE}/api/experiments/${id}`, {
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
  const res = await authFetch(`${API_BASE}/api/experiments/${experimentId}/async-train`, {
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
  const res = await authFetch(`${API_BASE}/api/experiments/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error(`Failed to delete experiment: ${res.status}`)
}

export async function saveExperimentMessages(
  id: string,
  messages: Array<Record<string, unknown>>,
): Promise<void> {
  const res = await authFetch(`${API_BASE}/api/experiments/${id}/messages`, {
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

  authFetch(`${API_BASE}/api/chat`, {
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
