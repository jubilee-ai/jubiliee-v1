/**
 * API client for the Jubilee Training Agent FastAPI backend.
 */

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

  // Error
  error?: string
}

export interface AgentStreamRequest {
  message: string
  experiment_id?: string
  linked_datasets?: string[] | null
  model_preference?: string | null
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
  const res = await fetch(`${API_BASE}/api/datasets`)
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.status}`)
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
  updates: { name?: string; status?: string; goal?: string; linked_datasets?: string[] },
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  })
  if (!res.ok) throw new Error(`Failed to update experiment: ${res.status}`)
}

export async function deleteExperiment(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/experiments/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error(`Failed to delete experiment: ${res.status}`)
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
