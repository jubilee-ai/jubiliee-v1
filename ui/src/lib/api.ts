/**
 * API client for the Jubilee Training Agent FastAPI backend.
 */

const API_BASE = "" // Relative; proxied by nginx in Docker or same-origin in dev

export interface Dataset {
  id?: string
  name?: string
  description?: string
  file: string
  format?: string
  rows?: number
  columns?: string[]
}

export interface ModelType {
  id: string
  name: string
  description?: string
}

export interface StreamEvent {
  type: string
  node?: string
  progress?: number
  thread_id?: string
  summary?: unknown
  details?: unknown
  state?: Record<string, unknown>
  state_snapshot?: Record<string, unknown>
  dataset?: string
  ref?: string
  error?: string
}

export interface ChatStreamEvent {
  type: string
  thread_id?: string
  content?: string
  tool?: string
  node?: string
  progress?: number
  summary?: unknown
  state?: Record<string, unknown>
  error?: string
}

export interface TrainRequest {
  goal: string
  linked_datasets?: string[] | null
  user_model_preference?: string | null
}

export interface TrainResponse {
  job_id: string
  status: string
  message: string
}

export interface JobStatus {
  job_id: string
  status: string
  progress: number
  current_step?: string | null
  state?: Record<string, unknown> | null
  error?: string | null
  started_at?: string | null
  completed_at?: string | null
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

/** Trained model registry entries keyed by model name (same shape as backend catalog). */
export async function getTrainedModels(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/trained-models`)
  if (!res.ok) throw new Error(`Failed to fetch trained models: ${res.status}`)
  return res.json()
}

export async function startTraining(req: TrainRequest): Promise<TrainResponse> {
  const res = await fetch(`${API_BASE}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      goal: req.goal,
      linked_datasets: req.linked_datasets ?? null,
      user_model_preference: req.user_model_preference ?? null,
    }),
  })
  if (!res.ok) throw new Error(`Failed to start training: ${res.status}`)
  return res.json()
}

export async function getTrainingStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/train/${jobId}`)
  if (!res.ok) throw new Error(`Failed to get training status: ${res.status}`)
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

export function streamTraining(
  req: { goal: string; linked_datasets?: string[] | null; user_model_preference?: string | null; hitl?: boolean },
  onEvent: (event: StreamEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController()
  const body = JSON.stringify({
    goal: req.goal,
    linked_datasets: req.linked_datasets ?? null,
    user_model_preference: req.user_model_preference ?? null,
    hitl: req.hitl ?? true,
  })

  fetch(`${API_BASE}/api/train-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Stream failed: ${res.status}`)
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
          if (parsed) onEvent(parsed as StreamEvent)
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(err instanceof Error ? err : new Error(String(err)))
    })

  return controller
}

export function streamResumeTraining(
  req: { thread_id: string; approved?: boolean; feedback?: string },
  onEvent: (event: StreamEvent) => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController()
  const body = JSON.stringify(req)

  fetch(`${API_BASE}/api/train-resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`Resume stream failed: ${res.status}`)
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
          if (parsed) onEvent(parsed as StreamEvent)
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(err instanceof Error ? err : new Error(String(err)))
    })

  return controller
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
  updates: { name?: string; status?: string },
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
  req: { message: string; thread_id?: string; experiment_id?: string; training_context?: string },
  onEvent: (event: ChatStreamEvent) => void,
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
          if (parsed) onEvent(parsed as ChatStreamEvent)
        }
      }
    })
    .catch((err) => {
      if (err?.name !== "AbortError") onError(err instanceof Error ? err : new Error(String(err)))
    })

  return controller
}
