/**
 * Training/chat agent hook (FastAPI backend).
 *
 * Shipped shape: one active experiment in React state; applyExperimentDetail swaps that slice.
 * Target doc ui/architecture-reference.html describes an ideal per-experiment map +
 * ExperimentSessionContext in the reducer — stronger isolation for concurrent sessions, not required
 * for a single active experiment.
 */

import { useState, useCallback, useRef, useEffect } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { experimentKeys } from "@/lib/queries"
import type {
  TrainingAgentState,
  StepInfo,
  ChatMessage,
  AnalysisInsight,
  AttachedDatasetSnapshot,
  ConfirmationRequest,
  ConfirmationAction,
  TaskPlanSummary,
  ChatTaskPlanPayload,
} from "@/types/agent"
import {
  checkHealth,
  getDatasets,
  getModelTypes,
  getTrainedModels,
  streamChat,
  getExperiment,
  saveExperimentMessages,
  createExperiment,
  startExperimentAsyncTrain,
  type Dataset,
  type ModelType,
  type TrainedModelEntry,
  type AgentStreamEvent,
  type ExperimentDetail,
  updateExperiment,
} from "@/lib/api"
import { uid, suggestExperimentTitleFromUserMessage } from "@/lib/utils"
import { buildStepDetailMarkdown } from "@/hooks/stepStreamDetails"
import { looksLikeLeakedPlanJson, stripLeakedPlanJson } from "@/lib/planDisplay"
import {
  toastAcceptAllMode,
  toastAutoAcceptContinued,
  toastBackgroundRunStarted,
  toastStepAccepted,
} from "@/lib/trainingToasts"

function agentDebug(event: string, payload?: Record<string, unknown>) {
  const ts = new Date().toISOString()
  if (payload) {
    console.log(`[hook:useRealAgent][${ts}] ${event}`, payload)
    return
  }
  console.log(`[hook:useRealAgent][${ts}] ${event}`)
}

/** Ephemeral merged "thinking" lines from graph custom stream (cleared on each new `started`). */
const GRAPH_THINKING_MSG_ID = "__graph_thinking__"

// Step definitions matching the agent's tool set (agent_simple.py _STEP_ORDER)
const STEP_DEFINITIONS = [
  { id: "data_collection", name: "Data Collection", description: "Load or collect the dataset" },
  { id: "select_model", name: "Model Selection", description: "Choose the ML model type for this task" },
  { id: "cleaning", name: "Cleaning", description: "Clean and standardize the data" },
  { id: "label_split_definition", name: "Label & Split", description: "Define target column and train/val/test splits" },
  { id: "feature_specification_and_engineering", name: "Features", description: "Specify features and build transformed datasets" },
  { id: "feature_selection_specification", name: "Feature Selection", description: "Analyze data and specify features" },
  { id: "feature_engineering_executor", name: "Feature Engineering", description: "Execute feature transformations" },
  { id: "feature_experiment_runner", name: "Feature experiments", description: "Compare feature-set variants with scout models" },
  { id: "evaluate_models", name: "Model comparison", description: "Compare candidate estimators on validation data" },
  { id: "training_approval", name: "Training plan", description: "Estimator, hyperparameters, and validation setup" },
  { id: "training", name: "Training", description: "Train model and evaluate metrics" },
  { id: "generate_report", name: "Report", description: "Save the final training report" },
]

function createInitialSteps(): StepInfo[] {
  return STEP_DEFINITIONS.map((def) => ({
    id: def.id,
    name: def.name,
    description: def.description,
    status: "pending",
  }))
}

const STEP_ORDER_IDS = STEP_DEFINITIONS.map((s) => s.id)

const DEFAULT_RUNNING_HINT = "Working"

/** Label & Split step copy: supervised needs target + splits; unsupervised uses full cleaned data only */
function labelSplitStepDescription(
  selectedModel: string | null | undefined,
  splitStrategy: string | null | undefined,
): string {
  if (selectedModel === "unsupervised" || splitStrategy === "none") {
    return "Prepare the full dataset for unsupervised training (no target or split)"
  }
  return "Define target column and train/val/test splits"
}

/** Shown next to the loading indicator while a checklist step is active */
const STEP_LOADING_HINTS: Record<string, string> = {
  data_collection: "Loading your dataset…",
  select_model: "Continuing setup…",
  cleaning: "Cleaning and standardizing columns…",
  label_split_definition: "Defining the target and train/validation/test splits…",
  feature_specification_and_engineering: "Specifying and building features…",
  feature_selection_specification: "Analyzing columns, correlations, and leakage…",
  feature_engineering_executor: "Encoding features and checking matrix shapes…",
  feature_experiment_runner: "Running feature experiments and picking the best variant…",
  evaluate_models: "Comparing candidate models on validation data…",
  training_approval: "Preparing training configuration…",
  training: "Training models and comparing validation metrics…",
  generate_report: "Finishing up…",
}

const PHASE_LOADING_HINTS: Record<string, string> = {
  ...STEP_LOADING_HINTS,
  planner: "Planning your run…",
  evaluator: "Reviewing options…",
  dispatch: "Setting things up…",
}

function normalizeRunningHint(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  if (!trimmed) return null
  return /[.!?…]$/.test(trimmed) ? trimmed : `${trimmed}…`
}

function compactStatusValue(value: unknown, maxLength = 56): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim().replace(/\s+/g, " ")
  if (!trimmed) return null
  if (trimmed.length <= maxLength) return trimmed
  return `${trimmed.slice(0, maxLength - 1).trimEnd()}…`
}

function labelFromPathLikeValue(value: unknown, maxLength = 40): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  if (!trimmed) return null
  const short = trimmed.split("/").filter(Boolean).pop() || trimmed
  return compactStatusValue(short, maxLength)
}

function isGenericToolHeadline(headline: string, toolName: string): boolean {
  const normalized = headline.toLowerCase().replace(/[.…!?]+$/g, "").trim()
  const raw = toolName.toLowerCase()
  const humanized = raw.replace(/_/g, " ")
  return normalized === `running ${raw}` || normalized === `running ${humanized}`
}

function getToolRunningHint(
  toolName: string,
  args?: Record<string, unknown>,
  headline?: string,
): string {
  const query = compactStatusValue(args?.query)
  const source = compactStatusValue(args?.source, 20)?.toLowerCase()
  const datasetLabel =
    labelFromPathLikeValue(args?.dataset_ref) ??
    labelFromPathLikeValue(args?.identifier) ??
    labelFromPathLikeValue(args?.dataset) ??
    labelFromPathLikeValue(args?.ref)
  const modelLabel =
    compactStatusValue(args?.model_name, 32) ??
    compactStatusValue(args?.model, 32) ??
    compactStatusValue(args?.model_type, 32)

  switch (toolName) {
    case "search_datasets":
      if (query) return `Searching datasets for "${query}"`
      if (source) return `Searching ${source} datasets`
      return "Searching datasets"
    case "curate_dataset":
      if (datasetLabel) return `Preparing dataset ${datasetLabel}`
      return "Preparing dataset"
    case "get_dataset_info":
      if (datasetLabel) return `Loading details for ${datasetLabel}`
      return "Loading dataset details"
    case "list_dataset_files":
      if (datasetLabel) return `Checking files for ${datasetLabel}`
      return "Checking dataset files"
    case "propose_training_plan":
      return "Planning your run"
    case "chart_tool":
    case "correlation_matrix_tool":
    case "group_summary_tool":
    case "eda_report_tool":
    case "distribution_analysis_tool":
    case "feature_diagnostics_tool":
    case "data_validation_tool":
    case "trend_analysis_tool":
    case "concentration_analysis_tool":
      if (datasetLabel) return `Analyzing ${datasetLabel}`
      return "Analyzing your data"
    case "train_model":
      if (modelLabel) return `Training ${modelLabel}`
      return "Training model"
    case "check_trained_models":
      return "Checking trained models"
    case "predict":
      if (datasetLabel) return `Running predictions for ${datasetLabel}`
      return "Running predictions"
    case "get_model_details":
      if (modelLabel) return `Loading ${modelLabel} details`
      return "Loading model details"
    default: {
      const normalizedHeadline = normalizeRunningHint(headline)
      if (normalizedHeadline && !isGenericToolHeadline(normalizedHeadline, toolName)) {
        return normalizedHeadline
      }
      return DEFAULT_RUNNING_HINT
    }
  }
}

/**
 * Shared node_complete step transitions for train stream and chat stream (plan phase parity).
 * Mirrors prior stream handler behavior: complete node, stale downstream on re-run,
 * fill prior pending, start next pending.
 */
function applyNodeCompleteToSteps(
  prev: StepInfo[],
  nodeName: string,
  _event: AgentStreamEvent,
): StepInfo[] {
  const nodeIndex = STEP_ORDER_IDS.indexOf(nodeName)
  const currentStepState = prev.find((s) => s.id === nodeName)
  const isRerun = (currentStepState?.runCount || 0) >= 1

  const next: StepInfo[] = prev.map((step, index): StepInfo => {
    if (step.id === nodeName) {
      return {
        ...step,
        status: "completed",
        endTime: Date.now(),
        runCount: (step.runCount || 0) + 1,
      }
    }
    if (nodeIndex < 0) {
      return step
    }
    if (isRerun && index > nodeIndex && (step.status === "completed" || step.status === "stale")) {
      return { ...step, status: "stale" }
    }
    if (!isRerun) {
      if (index < nodeIndex && step.status === "pending") {
        return { ...step, status: "completed" }
      }
      if (index === nodeIndex + 1 && step.status === "pending") {
        return { ...step, status: "running", startTime: Date.now() }
      }
    }
    return step
  })
  // Parent row for the Features group: the graph runs feature_selection_specification next, not this id.
  if (nodeName === "feature_selection_specification") {
    return next.map((step): StepInfo =>
      step.id === "feature_specification_and_engineering" && step.status === "running"
        ? { ...step, status: "completed", endTime: Date.now() }
        : step,
    )
  }
  return next
}

function subtitleFromSkippedEvent(summary: unknown): string {
  if (!summary || typeof summary !== "object" || Array.isArray(summary)) return ""
  const rec = summary as Record<string, unknown>
  const r = rec.reason ?? rec.message ?? rec.detail
  if (typeof r === "string" && r.trim()) return r.length > 120 ? `${r.slice(0, 117)}…` : r
  return ""
}

function applyNodeSkippedToSteps(prev: StepInfo[], nodeName: string): StepInfo[] {
  return prev.map((step) =>
    step.id === nodeName ? { ...step, status: "skipped" as const, endTime: Date.now() } : step,
  )
}

function markStepsDonePreservingSkipped(prev: StepInfo[]): StepInfo[] {
  return prev.map((step) =>
    step.status === "skipped"
      ? step
      : { ...step, status: "completed" as const, endTime: step.endTime || Date.now() },
  )
}

/** Reconcile checklist steps from async task metadata on the experiment. */
function mergeTaskProgressIntoSteps(
  prev: StepInfo[],
  ts: Record<string, unknown>,
): StepInfo[] {
  const st = ts.task_status as string | undefined
  const events = Array.isArray(ts.task_step_events)
    ? (ts.task_step_events as Array<{ node: string; type?: string }>)
    : []
  const done = new Set(events.filter((e) => e?.type !== "skipped").map((e) => e.node))
  const skipped = new Set(events.filter((e) => e?.type === "skipped").map((e) => e.node))
  const cur = (ts.task_current_node as string | null) || null
  const hasTerminalResults =
    (ts.training_metrics != null && typeof ts.training_metrics === "object") ||
    typeof ts.model_weights_path === "string" ||
    typeof ts.report_path === "string"

  if (ts.lab_mode !== "task" && events.length === 0 && !hasTerminalResults) return prev

  if (st === "completed" || (!st && hasTerminalResults)) {
    return prev.map((s) => {
      if (skipped.has(s.id)) return { ...s, status: "skipped" as const, endTime: s.endTime ?? Date.now() }
      return { ...s, status: "completed" as const, endTime: s.endTime ?? Date.now() }
    })
  }
  if (st === "failed") {
    return prev.map((s) => {
      if (skipped.has(s.id)) return { ...s, status: "skipped" as const }
      if (done.has(s.id)) return { ...s, status: "completed" as const }
      if (s.id === cur) return { ...s, status: "error" as const }
      return { ...s, status: "pending" as const }
    })
  }
  // task_current_node is last-completed (see backend merge on step.complete), not the active step.
  let runningId: string | null = null
  for (const id of STEP_ORDER_IDS) {
    if (!done.has(id) && !skipped.has(id)) {
      runningId = id
      break
    }
  }
  return prev.map((s) => {
    if (skipped.has(s.id)) {
      return { ...s, status: "skipped" as const, endTime: s.endTime ?? Date.now() }
    }
    if (done.has(s.id)) {
      return { ...s, status: "completed" as const, endTime: s.endTime ?? Date.now() }
    }
    if (runningId !== null && s.id === runningId) {
      return { ...s, status: "running" as const, startTime: s.startTime ?? Date.now() }
    }
    return { ...s, status: "pending" as const }
  })
}

function userTextForApi(m: ChatMessage): string {
  if (m.role === "user" && m.apiPayload?.trim()) return m.apiPayload.trim()
  return m.content.trim()
}

/** Build user/agent turns for the backend planner (`messages` is pre-send snapshot; `latestUserText` is the outgoing instruction). */
/** Normalize plan fields from tool JSON or task_plan.proposed SSE into a card payload. */
function chatTaskPlanFromPlanFields(data: {
  goal?: string
  dataset_refs?: string[]
  dataset_labels?: string[]
  preferences?: string | null
  recap_steps?: string[]
}): ChatTaskPlanPayload | null {
  if (!data.goal?.trim()) return null
  const refs = data.dataset_refs ?? []
  const plan: TaskPlanSummary = {
    goal: data.goal.trim(),
    datasetLabels: data.dataset_labels?.length ? data.dataset_labels : refs,
    preferences: data.preferences ?? null,
    steps: data.recap_steps?.length ? data.recap_steps : [],
  }
  return { plan, datasetRefs: refs }
}

/** Drops trailing empty/streaming/leaked-plan agent bubbles before attaching a plan message. */
function pruneAgentMessagesBeforeTaskPlan(prev: ChatMessage[]): ChatMessage[] {
  const next = [...prev]
  while (next.length > 0) {
    const last = next[next.length - 1]
    if (last.role !== "agent") break
    if (last.taskPlan) break
    const c = last.content ?? ""
    if (
      last._streaming ||
      looksLikeLeakedPlanJson(c) ||
      stripLeakedPlanJson(c).trim() === ""
    ) {
      next.pop()
      continue
    }
    break
  }
  const lastAfter = next[next.length - 1]
  if (lastAfter?.role === "agent" && lastAfter.content && !lastAfter.taskPlan) {
    const stripped = stripLeakedPlanJson(lastAfter.content)
    if (stripped !== lastAfter.content) {
      next[next.length - 1] = { ...lastAfter, content: stripped }
    }
  }
  return next
}

function buildPlanningConversation(
  messages: ChatMessage[],
  latestUserText: string,
): Array<{ role: "user" | "agent"; content: string }> {
  const trimmed = latestUserText.trim()
  const turns: Array<{ role: "user" | "agent"; content: string }> = []
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "agent") continue
    const c = m.role === "user" ? userTextForApi(m) : m.content.trim()
    if (c) turns.push({ role: m.role, content: c })
  }
  const last = turns[turns.length - 1]
  if (trimmed && (!last || last.role !== "user" || last.content !== trimmed)) {
    return [...turns, { role: "user", content: trimmed }]
  }
  if (turns.length > 0) return turns
  return [{ role: "user", content: trimmed || "Training run" }]
}

function createInitialState(): TrainingAgentState {
  return {
    goal: "",
    linked_datasets: null,
    user_model_preference: null,
    selected_model: null,
    model_explanation: null,
    model_regen_count: 0,
    collected_dataset_ref: null,
    cleaned_dataset_ref: null,
    cleaning_transformations: [],
    cleaning_summary: null,
    label_definition: null,
    split_indices: null,
    train_dataset_ref: null,
    val_dataset_ref: null,
    test_dataset_ref: null,
    feature_spec: null,
    analysis_trace: [],
    transformed_dataset_ref: null,
    transformed_train_ref: null,
    transformed_val_ref: null,
    transformed_test_ref: null,
    feature_validation_passed: false,
    feature_pipeline_mode: null,
    human_confirmed: false,
    training_params: null,
    model_weights_path: null,
    training_metrics: null,
    training_iteration: 0,
    feature_redo_requested: false,
    feature_redo_recommendation: null,
    feature_redo_reason: null,
    feature_redo_iteration: 0,
    report_path: null,
    audit_trace: [],
    explanations: [],
    current_step: "idle",
    error: null,
    hitl_auto_approve: false,
  }
}

export interface UseRealAgentOptions {
  /** Called after a new experiment row is created (auto-create on send / link dataset). */
  onExperimentEnsured?: (id: string) => void
}

export interface UseRealAgentReturn {
  // State
  agentState: TrainingAgentState
  steps: StepInfo[]
  messages: ChatMessage[]
  isRunning: boolean
  isBackendConnected: boolean
  currentJobId: string | null
  progress: number
  confirmationRequest: ConfirmationRequest | null
  experimentId: string | null
  
  // Data
  datasets: Dataset[]
  modelTypes: ModelType[]
  trainedModels: TrainedModelEntry[]
  trainedModelsLoading: boolean
  
  // Actions
  startAgent: (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => Promise<void>
  sendMessage: (
    content: string,
    opts?: {
      user_model_preference?: string
      linked_datasets_override?: string[]
      /** Start “background task intake”: orchestrator only until the user starts a run from the plan card. */
      begin_background_intake?: boolean
      force_orchestrator?: boolean
      persist_linked_datasets?: string[]
      /** Shown as conversation topic; `content` is still sent to the API */
      displayTopic?: string
    },
  ) => void
  /** True while assigning a background task: chat is for clarifications only; no training graph in-thread. */
  backgroundIntakeActive: boolean
  /** True after Approve on the plan card until the experiment is in task lab mode (API + reload). */
  startingHandsOffTask: boolean
  linkedDatasets: string[]
  /** Resolved dataset snapshots from the latest SSE turn (shown above chat). */
  attachedDatasetSnapshots: AttachedDatasetSnapshot[]
  updateLinkedDatasets: (ids: string[]) => void
  handleConfirmation: (action: ConfirmationAction, comment?: string) => void
  reset: () => void
  /** Ping `/api/health` only (for status banner + pre-flight). Does not refetch datasets/models. */
  checkConnection: () => Promise<boolean>
  /** Refetch dataset catalog (e.g. when opening the Datasets tab). */
  refreshDatasets: () => Promise<void>
  /** Refetch model types (e.g. when opening the Models tab). */
  refreshModelTypes: () => Promise<void>
  /** Refetch trained model registry. */
  refreshTrainedModels: () => Promise<void>
  setExperimentId: (id: string | null) => void
  /** Replace local lab state from a full experiment row (from GET /experiments/:id or TanStack Query). */
  applyExperimentDetail: (exp: ExperimentDetail) => void
  saveCurrentMessages: () => Promise<void>
  /** Clear local session and deselect experiment (experiment row remains in the list). */
  leaveLabSession: () => void
  /** Human-readable hint for what the pipeline is doing right now */
  runningStepHint: string | null
  /** Pull latest `training_state` from the server (e.g. while an async task runs). */
  refreshExperimentTraining: () => Promise<void>
  markTaskPlanResolved: (messageId: string) => void
  startGuidedTrainingFromPlan: (plan: TaskPlanSummary, refs: string[]) => void
  startHandsOffTrainingFromPlan: (plan: TaskPlanSummary, refs: string[]) => Promise<void>
}

export function useRealAgent(options?: UseRealAgentOptions): UseRealAgentReturn {
  const queryClient = useQueryClient()
  const onExperimentEnsuredRef = useRef(options?.onExperimentEnsured)
  onExperimentEnsuredRef.current = options?.onExperimentEnsured
  const [agentState, setAgentState] = useState<TrainingAgentState>(createInitialState())
  const [steps, setSteps] = useState<StepInfo[]>(createInitialSteps())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isBackendConnected, setIsBackendConnected] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [modelTypes, setModelTypes] = useState<ModelType[]>([])
  const [trainedModels, setTrainedModels] = useState<TrainedModelEntry[]>([])
  const [trainedModelsLoading, setTrainedModelsLoading] = useState(false)

  const datasetsLoadedRef = useRef(false)
  const modelTypesLoadedRef = useRef(false)
  const trainedModelsLoadedRef = useRef(false)

  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null)
  const [acceptAllMode, setAcceptAllMode] = useState(false)
  const [experimentId, setExperimentId] = useState<string | null>(null)
  const [linkedDatasets, setLinkedDatasets] = useState<string[]>([])
  const [attachedDatasetSnapshots, setAttachedDatasetSnapshots] = useState<AttachedDatasetSnapshot[]>([])
  const [backgroundIntakeActive, setBackgroundIntakeActive] = useState(false)
  const [startingHandsOffTask, setStartingHandsOffTask] = useState(false)
  const backgroundIntakeActiveRef = useRef(false)

  const setBackgroundIntake = useCallback((active: boolean) => {
    backgroundIntakeActiveRef.current = active
    setBackgroundIntakeActive(active)
  }, [])
  
  // Use a ref to track accept-all mode to avoid stale closure issues in callbacks
  const acceptAllModeRef = useRef(acceptAllMode)
  acceptAllModeRef.current = acceptAllMode

  /** Set after `startGuidedTrainingFromPlan` exists — used from stream handler for accept-all + plan. */
  const startGuidedTrainingFromPlanRef = useRef<(plan: TaskPlanSummary, refs: string[]) => void>(() => {})
  
  const messagesRef = useRef<ChatMessage[]>([])
  messagesRef.current = messages

  const experimentIdRef = useRef<string | null>(experimentId)
  experimentIdRef.current = experimentId

  const linkedDatasetsRef = useRef<string[]>(linkedDatasets)
  linkedDatasetsRef.current = linkedDatasets

  const loadGenerationRef = useRef(0)
  const streamControllerRef = useRef<AbortController | null>(null)
  const linkedDatasetsPatchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const emittedStepsRef = useRef<Set<string>>(new Set())
  /** Prevents duplicate “pipeline finished” messages (e.g. React Strict Mode double-invoke). */
  const pipelineCompletionEmittedRef = useRef(false)
  const agentStateRef = useRef(agentState)
  agentStateRef.current = agentState
  /** Stream end can run before React applies training_metrics to state; cache from step.complete. */
  const lastTrainingSummaryRef = useRef<Record<string, unknown> | null>(null)
  /** Drop post-tool assistant tokens (e.g. “I’ve set up a plan…”) after propose_training_plan. */
  const suppressPostPlanTokensRef = useRef(false)
  const activeToolNameRef = useRef<string | null>(null)

  const [runningStepHint, setRunningStepHint] = useState<string | null>(null)
  const [activeToolHint, setActiveToolHint] = useState<string | null>(null)
  /** Overrides checklist hint while the graph emits step.progress (phase matches a pipeline id). */
  const [progressPhaseHint, setProgressPhaseHint] = useState<string | null>(null)

  // Add a message to the chat (optional step metadata for expandable details + report CTA)
  const addMessage = useCallback(
    (
      role: ChatMessage["role"],
      content: string,
      meta?: Partial<
        Pick<
          ChatMessage,
          "stepId" | "detailMarkdown" | "showReportButton" | "taskPlan" | "apiPayload" | "linkedDatasetKeys"
        >
      >,
    ) => {
      setMessages((prev) => [
        ...prev,
        {
          id: uid("msg"),
          role,
          content,
          timestamp: Date.now(),
          ...meta,
        },
      ])
    },
    [],
  )

  const mergeChatIntoExperimentCache = useCallback(
    (eid: string, serialized: Array<Record<string, unknown>>) => {
      agentDebug("merge-chat-into-query-cache", {
        experimentId: eid,
        serializedCount: serialized.length,
      })
      queryClient.setQueryData<ExperimentDetail>(experimentKeys.detail(eid), (prev) => {
        if (prev) {
          agentDebug("merge-chat-hit-existing-detail", {
            experimentId: eid,
            prevChatCount: Array.isArray(prev.chat_history) ? prev.chat_history.length : null,
          })
          return { ...prev, chat_history: serialized }
        }
        // Do not synthesize partial experiment detail rows in cache.
        // Selection should rely on real GET /experiments/:id responses.
        agentDebug("merge-chat-skip-no-detail", { experimentId: eid })
        return prev
      })
    },
    [queryClient],
  )

  const saveCurrentMessages = useCallback(async () => {
    const eid = experimentIdRef.current
    const msgs = messagesRef.current
    if (!eid || msgs.length === 0) {
      agentDebug("save-current-messages-skip", {
        experimentId: eid,
        messagesCount: msgs.length,
      })
      return
    }
    agentDebug("save-current-messages-start", {
      experimentId: eid,
      messagesCount: msgs.length,
    })

    const serialized = msgs.map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      timestamp: m.timestamp,
      ...(m.stepId ? { stepId: m.stepId } : {}),
      ...(m.detailMarkdown ? { detailMarkdown: m.detailMarkdown } : {}),
      ...(m.showReportButton ? { showReportButton: m.showReportButton } : {}),
      ...(m.taskPlan ? { task_plan: m.taskPlan } : {}),
      ...(m.taskPlanResolved ? { task_plan_resolved: true } : {}),
      ...(m.apiPayload ? { api_payload: m.apiPayload } : {}),
      ...(m.linkedDatasetKeys?.length ? { linked_dataset_keys: m.linkedDatasetKeys } : {}),
    }))

    mergeChatIntoExperimentCache(eid, serialized)

    try {
      await saveExperimentMessages(eid, serialized)
      agentDebug("save-current-messages-success", {
        experimentId: eid,
        savedCount: serialized.length,
      })
    } catch {
      agentDebug("save-current-messages-error", {
        experimentId: eid,
        savedCount: serialized.length,
      })
      // best-effort
    }
  }, [mergeChatIntoExperimentCache])

  const refreshDatasets = useCallback(async () => {
    if (datasetsLoadedRef.current) {
      getDatasets().then(setDatasets).catch(() => {})
      return
    }
    try {
      const data = await getDatasets()
      setDatasets(data)
      datasetsLoadedRef.current = true
    } catch {
      setDatasets([])
    }
  }, [])

  const refreshModelTypes = useCallback(async () => {
    if (modelTypesLoadedRef.current) {
      getModelTypes().then(setModelTypes).catch(() => {})
      return
    }
    try {
      const data = await getModelTypes()
      setModelTypes(data)
      modelTypesLoadedRef.current = true
    } catch {
      setModelTypes([])
    }
  }, [])

  const refreshTrainedModels = useCallback(async () => {
    if (trainedModelsLoadedRef.current) {
      getTrainedModels()
        .then((data) => setTrainedModels(Object.values(data)))
        .catch(() => {})
      return
    }
    setTrainedModelsLoading(true)
    try {
      const data = await getTrainedModels()
      setTrainedModels(Object.values(data))
      trainedModelsLoadedRef.current = true
    } catch {
      setTrainedModels([])
    } finally {
      setTrainedModelsLoading(false)
    }
  }, [])

  /** Health check only — avoids hammering datasets/models on a timer. */
  const checkConnection = useCallback(async () => {
    try {
      const connected = await checkHealth()
      setIsBackendConnected(connected)
      return connected
    } catch {
      setIsBackendConnected(false)
      return false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const connected = await checkHealth()
      if (cancelled) return
      setIsBackendConnected(connected)
      if (connected) {
        const [datasetsData, modelsData, trainedData] = await Promise.all([
          getDatasets().catch(() => [] as Dataset[]),
          getModelTypes().catch(() => [] as ModelType[]),
          getTrainedModels().catch(() => ({} as Record<string, TrainedModelEntry>)),
        ])
        if (!cancelled) {
          setDatasets(datasetsData)
          datasetsLoadedRef.current = true
          setModelTypes(modelsData)
          modelTypesLoadedRef.current = true
          setTrainedModels(Object.values(trainedData))
          trainedModelsLoadedRef.current = true
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const desc = labelSplitStepDescription(
      agentState.selected_model,
      agentState.label_definition?.split_strategy ?? null,
    )
    setSteps((prev) =>
      prev.map((s) => (s.id === "label_split_definition" ? { ...s, description: desc } : s)),
    )
  }, [agentState.selected_model, agentState.label_definition?.split_strategy])

  useEffect(() => {
    if (!isRunning) {
      setRunningStepHint(null)
      setProgressPhaseHint(null)
      return
    }
    if (confirmationRequest) {
      setRunningStepHint("Waiting for your review…")
      return
    }
    if (activeToolHint) {
      setRunningStepHint(activeToolHint)
      return
    }
    if (progressPhaseHint) {
      setRunningStepHint(progressPhaseHint)
      return
    }
    // Prefer the latest pipeline step when several rows are still marked running (e.g. Features parent + sub-step).
    let running: StepInfo | undefined
    let bestIdx = -1
    for (const s of steps) {
      if (s.status !== "running") continue
      const idx = STEP_ORDER_IDS.indexOf(s.id)
      if (idx >= 0 && idx > bestIdx) {
        bestIdx = idx
        running = s
      }
    }
    if (running === undefined) {
      running = steps.find((s) => s.status === "running")
    }
    if (running) {
      const hintKey = running.id === "training_approval" ? "training" : running.id
      let hint = STEP_LOADING_HINTS[hintKey] ?? `Running ${running.name}…`
      if (
        running.id === "label_split_definition" &&
        (agentState.selected_model === "unsupervised" ||
          agentState.label_definition?.split_strategy === "none")
      ) {
        hint = "Preparing full dataset for unsupervised training…"
      }
      setRunningStepHint(hint)
      return
    }
    setRunningStepHint(DEFAULT_RUNNING_HINT)
  }, [
    activeToolHint,
    isRunning,
    steps,
    confirmationRequest,
    progressPhaseHint,
    agentState.selected_model,
    agentState.label_definition?.split_strategy,
  ])

  // Derive a brief subtitle for a completed step
  const computeStepSubtitle = useCallback((nodeName: string, summary?: Record<string, unknown>): string => {
    if (!summary) return ""
    switch (nodeName) {
      case "select_model":
        return ""
      case "data_collection": {
        const rows = summary.rows
        const colList = summary.columns
        const ncol = Array.isArray(colList)
          ? colList.length
          : typeof colList === "number"
            ? colList
            : null
        const colsStr = ncol != null ? String(ncol) : "?"
        if (rows === undefined || rows === null || rows === "") return ""
        return `${Number(rows).toLocaleString()} rows, ${colsStr} cols`
      }
      case "cleaning":
      case "cleaning_and_standardization":
        return summary.num_transformations != null
          ? (Number(summary.num_transformations) === 0 ? "No changes needed" : `${summary.num_transformations} transformations`)
          : ""
      case "label_split_definition": {
        if (summary.split_strategy === "none") return ""
        return summary.target_column ? `Target: ${summary.target_column}` : ""
      }
      case "feature_selection_specification":
        return summary.num_features
          ? `${summary.num_features} features`
          : ""
      case "feature_engineering_executor":
        return "Exploratory analysis · matrix ready"
      case "feature_specification_and_engineering":
        return "Features & EDA · matrix built"
      case "feature_experiment_runner": {
        if (summary.skipped) return "skipped"
        const bv = summary.best_variant_name
        const tv = summary.total_variants
        const parts: string[] = []
        if (bv) parts.push(String(bv))
        if (tv != null) parts.push(`${tv} setups`)
        return parts.join(" · ")
      }
      case "evaluate_models": {
        const nm = summary.best_model
        const n = summary.num_models
        if (typeof n === "number" && n > 0) {
          return nm ? `${n} models · best: ${nm}` : `${n} models compared`
        }
        return ""
      }
      case "training_approval": {
        const model = summary.model_type || ""
        const hp = summary.hyperparameters
        const hpCount = hp && typeof hp === "object" ? Object.keys(hp).length : 0
        return model ? `${model}, ${hpCount} params` : ""
      }
      case "training": {
        const parts: string[] = []
        if (summary.test_accuracy != null) parts.push(`${(Number(summary.test_accuracy) * 100).toFixed(1)}% accuracy`)
        if (summary.test_roc_auc != null) parts.push(`AUC: ${Number(summary.test_roc_auc).toFixed(3)}`)
        if (summary.test_r2 != null) parts.push(`R²: ${Number(summary.test_r2).toFixed(4)}`)
        if (summary.test_rmse != null) parts.push(`RMSE: ${Number(summary.test_rmse).toFixed(0)}`)
        return parts.join(", ") || (summary.success ? "Completed" : "Failed")
      }
      case "generate_report":
        return "Complete"
      default:
        return ""
    }
  }, [])

  const applyAgentStreamEvent = useCallback((event: AgentStreamEvent) => {
    console.log("[agent-stream]", event)

    if (event.type === "start" || (event.type === "stream.start" && event.pipeline_completed == null)) {
      setIsRunning(true)
      if (event.type === "stream.start" && !event.training_graph) {
        setAttachedDatasetSnapshots([])
      }
      if (event.type === "stream.start" && event.training_graph) {
        setProgressPhaseHint(null)
        emittedStepsRef.current = new Set()
        setProgress(0)
        setSteps(() => {
          const initial = createInitialSteps()
          return initial.map((step, index) =>
            index === 0 ? { ...step, status: "running", startTime: Date.now() } : step,
          )
        })
      }
      return
    }
    if (event.type === "dataset.resolved") {
      const raw = event.dataset_info as Record<string, unknown> | undefined
      const cols = raw?.columns as AttachedDatasetSnapshot["columns"]
      const snap: AttachedDatasetSnapshot = {
        ref: String(event.ref || ""),
        rows: typeof raw?.rows === "number" ? raw.rows : undefined,
        n_columns:
          typeof raw?.n_columns === "number"
            ? raw.n_columns
            : Array.isArray(cols)
              ? cols.length
              : undefined,
        columns: Array.isArray(cols) ? cols : undefined,
        sample: Array.isArray(raw?.sample) ? (raw.sample as Record<string, unknown>[]) : undefined,
      }
      setAttachedDatasetSnapshots((prev) => [...prev.filter((x) => x.ref !== snap.ref), snap])
      return
    }
    if (event.type === "analysis.result") {
      const insight: AnalysisInsight = {
        tool: event.tool || "unknown",
        kind: event.kind || "unknown",
        summary: typeof event.summary === "string" ? event.summary : undefined,
        payload: event.payload,
      }
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        if (last && last.role === "agent") {
          next[next.length - 1] = {
            ...last,
            analyses: [...(last.analyses || []), insight],
          }
          return next
        }
        return [
          ...next,
          {
            id: uid("msg"),
            role: "agent",
            content: "",
            timestamp: Date.now(),
            _streaming: true,
            analyses: [insight],
          },
        ]
      })
      return
    }
    if (event.type === "analysis.chart") {
      const spec = event.spec
      if (!spec || typeof spec !== "object") return
      const toolNm = event.tool || "chart_tool"
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        if (!last || last.role !== "agent") return prev
        const analyses = [...(last.analyses || [])]
        let merged = false
        for (let i = analyses.length - 1; i >= 0; i--) {
          const a = analyses[i]
          if (a.kind === "chart" && (a.tool === toolNm || toolNm === "chart_tool")) {
            analyses[i] = { ...a, chartSpec: spec as Record<string, unknown> }
            merged = true
            break
          }
        }
        if (!merged) {
          analyses.push({
            tool: toolNm,
            kind: "chart",
            chartSpec: spec as Record<string, unknown>,
          })
        }
        next[next.length - 1] = { ...last, analyses }
        return next
      })
      return
    }
    if (event.type === "token") {
      if (suppressPostPlanTokensRef.current) {
        return
      }
      const phase = event.phase
      const isThinkingPhase = phase === "planner" || phase === "evaluator" || phase === "select_model"
      if (isThinkingPhase) {
        return
      }

      setMessages((prev) => {
        const last = prev[prev.length - 1]
        const chunk = event.content || ""
        if (last && last.role === "agent" && last._streaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, content: last.content + chunk },
          ]
        }
        return [
          ...prev,
          { id: uid("msg"), role: "agent", content: chunk, timestamp: Date.now(), _streaming: true },
        ]
      })
      return
    }
    if (event.type === "tool_call" || event.type === "tool.start") {
      const toolName = event.tool || "unknown"
      activeToolNameRef.current = toolName
      setActiveToolHint(normalizeRunningHint(getToolRunningHint(toolName, event.args, event.headline)) ?? DEFAULT_RUNNING_HINT)
      return
    }
    if (event.type === "task_plan.proposed") {
      const payload = chatTaskPlanFromPlanFields({
        goal: event.goal,
        dataset_refs: event.dataset_refs,
        dataset_labels: event.dataset_labels,
        preferences: event.preferences ?? null,
        recap_steps: event.recap_steps,
      })
      if (!payload) return
      suppressPostPlanTokensRef.current = true
      if (acceptAllModeRef.current) {
        setMessages((prev) => [
          ...pruneAgentMessagesBeforeTaskPlan(prev),
          {
            id: uid("msg"),
            role: "agent",
            content: "Starting your training run…",
            timestamp: Date.now(),
          },
        ])
        setTimeout(() => {
          startGuidedTrainingFromPlanRef.current(payload.plan, payload.datasetRefs)
        }, 0)
        return
      }
      setMessages((prev) => {
        const next = pruneAgentMessagesBeforeTaskPlan(prev)
        return [
          ...next,
          {
            id: uid("msg"),
            role: "agent",
            content: "",
            timestamp: Date.now(),
            taskPlan: payload,
          },
        ]
      })
      return
    }
    if (event.type === "tool_result" || event.type === "tool.end") {
      if (!event.tool || event.tool === activeToolNameRef.current) {
        activeToolNameRef.current = null
        setActiveToolHint(null)
      }
      if (event.tool === "propose_training_plan" && event.result != null) {
        try {
          const raw = typeof event.result === "string" ? event.result : String(event.result)
          const data = JSON.parse(raw) as {
            error?: string
            goal?: string
            dataset_refs?: string[]
            dataset_labels?: string[]
            preferences?: string | null
            recap_steps?: string[]
          }
          if (data.error) {
            return
          }
          const payload = chatTaskPlanFromPlanFields(data)
          if (!payload) return
          suppressPostPlanTokensRef.current = true
          if (acceptAllModeRef.current) {
            setMessages((prev) => [
              ...pruneAgentMessagesBeforeTaskPlan(prev),
              {
                id: uid("msg"),
                role: "agent",
                content: "Starting your training run…",
                timestamp: Date.now(),
              },
            ])
            setTimeout(() => {
              startGuidedTrainingFromPlanRef.current(payload.plan, payload.datasetRefs)
            }, 0)
          } else {
            setMessages((prev) => {
              const next = pruneAgentMessagesBeforeTaskPlan(prev)
              return [
                ...next,
                {
                  id: uid("msg"),
                  role: "agent",
                  content: "",
                  timestamp: Date.now(),
                  taskPlan: payload,
                },
              ]
            })
          }
        } catch {
          // ignore malformed tool JSON
        }
      }
      return
    }
    if (event.type === "predict.start") {
      const modelLabel = compactStatusValue(event.model, 32)
      setActiveToolHint(modelLabel ? `Running predictions with ${modelLabel}` : "Running predictions")
      return
    }
    if (event.type === "predict.complete") {
      setActiveToolHint(null)
      setMessages((prev) => [
        ...prev,
        {
          id: uid("msg"),
          role: "agent",
          content: "",
          timestamp: Date.now(),
          prediction: {
            model: event.model || "unknown",
            rows_predicted: event.rows_predicted || 0,
            headline: event.headline || "Prediction complete",
            result_ref: event.result_ref,
          },
        },
      ])
      return
    }
    if (event.type === "training_started") {
      emittedStepsRef.current = new Set()
      setProgress(0)
      setSteps(() => {
        const initial = createInitialSteps()
        return initial.map((step, index) =>
          index === 0 ? { ...step, status: "running", startTime: Date.now() } : step,
        )
      })
      return
    }
    if (event.type === "training_completed") {
      setProgress(100)
      setSteps((prev) => markStepsDonePreservingSkipped(prev))
      return
    }
    if (event.type === "end" || (event.type === "stream.end" && !event.pipeline_completed)) {
      suppressPostPlanTokensRef.current = false
      activeToolNameRef.current = null
      setActiveToolHint(null)
      setProgressPhaseHint(null)
      setMessages((prev) =>
        prev.map((m) => (m._streaming ? { ...m, _streaming: undefined } : m)),
      )
      setIsRunning(false)
      setTimeout(() => saveCurrentMessages(), 100)
      return
    }

    if (event.type === "thinking") {
      return
    }
    if (event.type === "step.progress") {
      const ph = (event as { phase?: string }).phase
      const messageHint = normalizeRunningHint(event.message)
      if (messageHint) {
        setProgressPhaseHint(messageHint)
        return
      }
      if (ph && PHASE_LOADING_HINTS[ph]) {
        setProgressPhaseHint(PHASE_LOADING_HINTS[ph])
        return
      }
      setProgressPhaseHint(DEFAULT_RUNNING_HINT)
      return
    }

    if (event.type === "started" || (event.type === "stream.start" && event.pipeline_completed != null)) {
      setMessages((prev) => prev.filter((m) => m.id !== GRAPH_THINKING_MSG_ID))
      setProgress(0)
      // Graph pipeline sends started with node "planner" before any step node runs;
      // avoid showing "Data collection" as running during planning.
      if (event.node !== "planner") {
        setSteps((prev) =>
          prev.map((step, index) => {
            if (index === 0) {
              return { ...step, status: "running", startTime: Date.now() }
            }
            return step
          }),
        )
      }
    } else if (event.type === "interrupt" || event.type === "review.required") {
      // Human-in-the-loop interrupt - pause for user approval
      console.log("[stream] Interrupt received:", event)
      
      const nodeName = event.node || "unknown"
      let summary: string
      if (typeof event.summary === "string") {
        summary = event.summary.trim()
        if (!summary && typeof event.message === "string" && event.message.trim()) {
          summary = event.message.trim()
        }
      } else if (nodeName === "training_approval" && event.summary && typeof event.summary === "object") {
        const s = event.summary as Record<string, unknown>
        const lines: string[] = []
        if (s.model_type) lines.push(`**Model:** ${s.model_type}`)
        if (s.task_type) lines.push(`**Task:** ${s.task_type}`)
        const hp = s.hyperparameters as Record<string, unknown> | undefined
        if (hp && typeof hp === "object") {
          lines.push(`**Hyperparameters:** ${Object.keys(hp).length} configured`)
        }
        summary = lines.join("\n")
      } else {
        summary = JSON.stringify(event.summary || {}, null, 2)
      }
      const stateSnapshot = (event.state_snapshot || {}) as Record<string, unknown>
      
      // Update step to awaiting confirmation with details
      // Also mark preceding pending steps as completed (they must have run)
      const stepOrder = STEP_DEFINITIONS.map((s) => s.id)
      const interruptIndex = stepOrder.indexOf(nodeName)
      
      setSteps((prev) =>
        prev.map((step, index) => {
          if (step.id === nodeName) {
            return { 
              ...step, 
              status: "awaiting_confirmation", 
              endTime: Date.now(),
              details: summary,
            }
          }
          if (index < interruptIndex && step.status === "pending") {
            return { ...step, status: "completed" }
          }
          return step
        })
      )
      
      // Update agent state with snapshot data
      if (stateSnapshot && Object.keys(stateSnapshot).length > 0) {
        setAgentState((prev) => ({
          ...prev,
          ...stateSnapshot as Partial<TrainingAgentState>,
        }))
      }
      
      // If in accept-all mode, auto-accept
      if (acceptAllModeRef.current) {
        console.log("[stream] Auto-accepting in accept-all mode")
        toastAutoAcceptContinued()
        setSteps((prev) =>
          prev.map((step) =>
            step.id === nodeName ? { ...step, status: "completed" } : step
          )
        )
        setTimeout(() => {
          streamControllerRef.current = streamChat(
            {
              message: "",
              experiment_id: experimentIdRef.current || undefined,
              resume: { approved: true },
            },
            applyAgentStreamEvent,
            (error: Error) => {
              setIsRunning(false)
              addMessage("system", `Stream error: ${error.message}`)
            },
          )
        }, 100)
        return
      }
      
      // Set confirmation request for UI with rich details
      const stepDef = STEP_DEFINITIONS.find((s) => s.id === nodeName)
      setConfirmationRequest({
        step: nodeName,
        stepName: stepDef?.name || nodeName,
        summary: summary,
        details: stateSnapshot,
      })
      
      // Pause running state so the input is enabled
      setIsRunning(false)
      
    } else if (event.type === "dataset_loaded" || event.type === "dataset.resolved") {
      return
    } else if (event.type === "dataset_error" || event.type === "dataset.error") {
      addMessage("system", `Could not load dataset: ${event.dataset || "unknown"}`)
    } else if (event.type === "node_complete" || event.type === "step.complete") {
      setProgressPhaseHint(null)
      const nodeName = event.node || "unknown"
      const nodeProgress = event.progress || 0
      const stepKey =
        typeof (event as { stream_step_key?: string }).stream_step_key === "string"
          ? (event as { stream_step_key: string }).stream_step_key
          : nodeName

      // Deduplicate per pipeline execution (same step name can repeat after replan/amend)
      if (emittedStepsRef.current.has(stepKey)) {
        console.log("[stream] Skipping duplicate node_complete for", stepKey)
        return
      }
      emittedStepsRef.current.add(stepKey)
      
      setProgress(nodeProgress)

      setSteps((prev) => applyNodeCompleteToSteps(prev, nodeName, event))

      // Compute a one-line subtitle for the step dropdown
      const summary = event.summary as Record<string, unknown> | undefined
      const subtitle = computeStepSubtitle(nodeName, summary)
      if (subtitle) {
        setSteps((prev) =>
          prev.map((step) =>
            step.id === nodeName ? { ...step, subtitle } : step
          )
        )
      }
      
      // Update agent state from event
      if (event.state) {
        setAgentState((prev) => ({
          ...prev,
          ...(event.state as Partial<TrainingAgentState>),
        }))
      }

      if (nodeName === "training" && summary) {
        lastTrainingSummaryRef.current = summary as Record<string, unknown>
      }

      // Graph orchestration nodes: update checklist/state only; no chat line.
      if (nodeName === "planner" || nodeName === "dispatcher" || nodeName === "evaluator") {
        return
      }

      // Selection + engineering are one user-facing step: only chat when engineering finishes.
      if (nodeName === "feature_selection_specification") {
        return
      }

      // Report write is internal; completion is surfaced via "Pipeline finished" + View Report only.
      if (nodeName === "generate_report") {
        return
      }

      // Model-family setup is internal; no user-facing chat line (headline/details are generic on the wire).
      if (nodeName === "select_model") {
        return
      }

      // Unsupervised label/split is graph bookkeeping only (no target or split); Cleaning already covered the dataset.
      if (nodeName === "label_split_definition") {
        const summ = summary as { split_strategy?: string } | undefined
        const evState = event.state as Partial<TrainingAgentState> | undefined
        const splitStrat =
          summ?.split_strategy ??
          evState?.label_definition?.split_strategy ??
          agentStateRef.current.label_definition?.split_strategy
        const selected =
          evState?.selected_model ?? agentStateRef.current.selected_model
        if (selected === "unsupervised" || splitStrat === "none") {
          return
        }
      }

      // "Ready to train **…** with N hyperparameters" is redundant when the user already chose auto-accept.
      if (nodeName === "training_approval" && acceptAllModeRef.current) {
        return
      }

      const headline =
        typeof event.headline === "string" && event.headline.trim()
          ? event.headline.trim()
          : `${STEP_DEFINITIONS.find((s) => s.id === nodeName)?.name || nodeName} complete`

      const detailMarkdown = buildStepDetailMarkdown(nodeName, event)
      const stepIdForMessage =
        nodeName === "cleaning_and_standardization" ? "cleaning" : nodeName

      // Data collection: single chat line; placeholder headlines still get a line when we have a ref (See preview CTA).
      if (nodeName === "data_collection") {
        const evState = event.state as Partial<TrainingAgentState> | undefined
        const collectedRef =
          (typeof evState?.collected_dataset_ref === "string" && evState.collected_dataset_ref.trim()) ||
          (typeof agentStateRef.current.collected_dataset_ref === "string" &&
            agentStateRef.current.collected_dataset_ref.trim()) ||
          ""
        const isPlaceholder =
          !headline.trim() ||
          /^dataset ready\.?$/i.test(headline) ||
          /^dataset collected\.?$/i.test(headline)
        let line = headline
        if (isPlaceholder) {
          if (!collectedRef) {
            return
          }
          const short = collectedRef.split("/").filter(Boolean).pop() || collectedRef
          line = `Loaded dataset **${short}**`
        }
        setMessages((prev) => {
          const base = prev.filter(
            (m) => !(m.role === "agent" && m.stepId === "data_collection"),
          )
          return [
            ...base,
            {
              id: uid("msg"),
              role: "agent",
              content: line,
              timestamp: Date.now(),
              stepId: "data_collection",
              detailMarkdown: detailMarkdown || undefined,
            },
          ]
        })
        return
      }

      addMessage("agent", headline, {
        stepId: stepIdForMessage,
        detailMarkdown: detailMarkdown || undefined,
      })
    } else if (event.type === "node_skipped" || event.type === "step.skipped") {
      const nodeName = event.node || "unknown"
      if (emittedStepsRef.current.has(`skipped:${nodeName}`)) {
        console.log("[stream] Skipping duplicate node_skipped for", nodeName)
        return
      }
      emittedStepsRef.current.add(`skipped:${nodeName}`)

      setSteps((prev) => {
        let next = applyNodeSkippedToSteps(prev, nodeName)
        if (
          nodeName === "feature_selection_specification" ||
          nodeName === "feature_engineering_executor"
        ) {
          next = applyNodeSkippedToSteps(next, "feature_specification_and_engineering")
        }
        return next
      })
      const skipSubtitle = subtitleFromSkippedEvent(event.summary)
      if (skipSubtitle) {
        setSteps((prev) =>
          prev.map((step) => (step.id === nodeName ? { ...step, subtitle: skipSubtitle } : step)),
        )
      }
    } else if (event.type === "completed" || (event.type === "stream.end" && event.pipeline_completed === true)) {
      setProgressPhaseHint(null)
      setProgress(100)
      setIsRunning(false)

      setSteps((prev) => markStepsDonePreservingSkipped(prev))

      if (pipelineCompletionEmittedRef.current) {
        return
      }
      pipelineCompletionEmittedRef.current = true

      const fromStep = lastTrainingSummaryRef.current
      const m = agentStateRef.current.training_metrics
      const acc = fromStep?.test_accuracy ?? m?.test_accuracy
      const auc = fromStep?.test_roc_auc ?? m?.test_roc_auc
      const r2 = fromStep?.test_r2 ?? m?.test_r2
      const rmse = fromStep?.test_rmse ?? m?.test_rmse

      let recap = ""
      if (acc != null) {
        recap = `Best test result: **${(Number(acc) * 100).toFixed(1)}% accuracy**`
        if (auc != null) recap += `, ROC-AUC **${Number(auc).toFixed(3)}**`
      } else if (r2 != null) {
        recap = `Best test result: R² **${Number(r2).toFixed(4)}**`
        if (rmse != null) recap += `, RMSE **${Number(rmse).toFixed(0)}**`
      } else {
        recap = "Your run finished successfully."
      }

      addMessage(
        "agent",
        `**Pipeline finished.** ${recap}`,
        { showReportButton: true },
      )
      setTimeout(() => saveCurrentMessages(), 100)
    } else if (event.type === "error") {
      setProgressPhaseHint(null)
      setIsRunning(false)
      
      // Mark current running step as error
      setSteps((prev) => prev.map((step) => 
        step.status === "running" ? { ...step, status: "error" } : step
      ))
      
      addMessage("system", `Error: ${event.error || "Unknown error"}`)
    }
  }, [addMessage, computeStepSubtitle, saveCurrentMessages])

  const ensureExperimentId = useCallback(
    async (
      suggestedName?: string | null,
      linkedDatasetsForCreate?: string[] | null,
    ): Promise<string | null> => {
      if (experimentIdRef.current) return experimentIdRef.current
      try {
        const name = suggestedName?.trim() || undefined
        const ld =
          linkedDatasetsForCreate != null && linkedDatasetsForCreate.length > 0
            ? linkedDatasetsForCreate
            : undefined
        const exp = await createExperiment(name, ld)
        setExperimentId(exp.id)
        experimentIdRef.current = exp.id
        void queryClient.invalidateQueries({ queryKey: experimentKeys.list() })
        onExperimentEnsuredRef.current?.(exp.id)
        return exp.id
      } catch {
        return null
      }
    },
    [queryClient],
  )

  const scheduleLinkedDatasetsPatch = useCallback((ids: string[]) => {
    const eid = experimentIdRef.current
    if (!eid) return
    if (linkedDatasetsPatchTimerRef.current) clearTimeout(linkedDatasetsPatchTimerRef.current)
    linkedDatasetsPatchTimerRef.current = setTimeout(() => {
      void updateExperiment(eid, { linked_datasets: ids })
    }, 400)
  }, [])

  const updateLinkedDatasets = useCallback(
    (ids: string[]) => {
      setLinkedDatasets(ids)
      if (experimentIdRef.current) {
        scheduleLinkedDatasetsPatch(ids)
      }
    },
    [scheduleLinkedDatasetsPatch],
  )

  // Start training with streaming
  const startAgentStreaming = useCallback(async (goal: string, linkedDatasets?: string[], modelPreference?: string, hitl?: boolean) => {
    const connected = await checkConnection()
    if (!connected) {
      addMessage("system", "Backend not connected. Please start the FastAPI server with: uvicorn app:app --reload")
      return
    }

    let runEid = experimentIdRef.current
    if (!runEid) {
      const hasUserMessage = messagesRef.current.some((m) => m.role === "user")
      if (!hasUserMessage) {
        addMessage(
          "system",
          "Send a message in chat first to create an experiment before starting training.",
        )
        return
      }
      const linked =
        linkedDatasets != null && linkedDatasets.length > 0 ? linkedDatasets : undefined
      runEid = await ensureExperimentId(suggestExperimentTitleFromUserMessage(goal), linked ?? null)
    }
    if (!runEid) {
      addMessage("system", "Could not create an experiment. Check the backend connection.")
      return
    }

    setIsRunning(true)
    setProgress(0)
    setSteps(createInitialSteps())
    emittedStepsRef.current = new Set()
    pipelineCompletionEmittedRef.current = false
    lastTrainingSummaryRef.current = null

    const initialState = createInitialState()
    initialState.goal = goal
    initialState.linked_datasets = linkedDatasets || null
    initialState.user_model_preference = modelPreference || null
    setAgentState(initialState)

    const hitlLabel = hitl === false ? " (no human review)" : ""
    addMessage("agent", `Starting training with goal: "${goal}"\n\nStreaming progress updates in real-time${hitlLabel}...`)

    const conversation = buildPlanningConversation(messagesRef.current, goal)
    streamControllerRef.current = streamChat(
      {
        message: goal,
        mode: "train",
        linked_datasets: linkedDatasets ?? null,
        model_preference: modelPreference ?? null,
        experiment_id: runEid,
        conversation,
      },
      applyAgentStreamEvent,
      (error: Error) => {
        setIsRunning(false)
        addMessage("system", `Stream error: ${error.message}`)
      },
    )
  }, [addMessage, checkConnection, applyAgentStreamEvent, ensureExperimentId])

  const handleConfirmation = useCallback((action: ConfirmationAction, comment?: string) => {
    if (!confirmationRequest || !experimentId) {
      console.warn("[handleConfirmation] No confirmation request or experiment_id")
      return
    }

    const currentStep = confirmationRequest.step
    const stepDef = STEP_DEFINITIONS.find((s) => s.id === currentStep)

    const resumeStream = (approved: boolean, feedback?: string) => {
      streamControllerRef.current = streamChat(
        {
          message: "",
          experiment_id: experimentId,
          resume: { approved, feedback },
        },
        applyAgentStreamEvent,
        (error: Error) => {
          setIsRunning(false)
          addMessage("system", `Stream error: ${error.message}`)
        },
      )
    }

    switch (action) {
      case "accept":
        toastStepAccepted()
        addMessage("system", "Step accepted")
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "completed" } : step
          )
        )
        setConfirmationRequest(null)
        setIsRunning(true)
        resumeStream(true)
        break

      case "accept_all":
        toastAcceptAllMode()
        addMessage("system", "Auto-accepting remaining steps")
        setAcceptAllMode(true)
        acceptAllModeRef.current = true
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "completed" } : step
          )
        )
        setConfirmationRequest(null)
        setIsRunning(true)
        resumeStream(true)
        break

      case "redo":
        addMessage("system", `↻ Requested redo${comment ? `: ${comment}` : ""}`)
        addMessage("agent", `Got it. I'll redo **${stepDef?.name || currentStep}**${comment ? ` with your feedback: "${comment}"` : ""}.`)
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "running", startTime: Date.now() } : step
          )
        )
        setConfirmationRequest(null)
        setIsRunning(true)
        resumeStream(false, comment || "Please redo this step.")
        break
    }
  }, [confirmationRequest, experimentId, addMessage, applyAgentStreamEvent])

  const startAgent = useCallback(async (goal: string, linkedDatasets?: string[], modelPreference?: string, hitl?: boolean) => {
    await startAgentStreaming(goal, linkedDatasets, modelPreference, hitl)
  }, [startAgentStreaming])

  const sendMessage = useCallback(
    (
      content: string,
      opts?: {
        user_model_preference?: string
        linked_datasets_override?: string[]
        begin_background_intake?: boolean
        force_orchestrator?: boolean
        persist_linked_datasets?: string[]
        displayTopic?: string
        /** Use `"train"` only when starting the interactive training graph (guided plan, explicit run). */
        mode?: "chat" | "train"
      },
    ) => {
      suppressPostPlanTokensRef.current = false
      const topic = opts?.displayTopic?.trim()

      const forceOrch =
        backgroundIntakeActiveRef.current || opts?.force_orchestrator === true

      const linkedForApi: string[] | null = forceOrch
        ? null
        : (() => {
            const raw =
              opts?.linked_datasets_override != null
                ? [...opts.linked_datasets_override]
                : [...linkedDatasetsRef.current]
            return raw.length > 0 ? raw : null
          })()

      const linkedForCreate: string[] | undefined = (() => {
        if (opts?.linked_datasets_override != null) {
          const o = opts.linked_datasets_override.filter(Boolean)
          return o.length > 0 ? o : undefined
        }
        if (opts?.begin_background_intake && opts?.persist_linked_datasets?.length) {
          return [...opts.persist_linked_datasets]
        }
        if (linkedForApi && linkedForApi.length > 0) {
          return [...linkedForApi]
        }
        const fromRef = [...linkedDatasetsRef.current].filter(Boolean)
        return fromRef.length > 0 ? fromRef : undefined
      })()

      const attachedAtSend: string[] = (() => {
        if (opts?.linked_datasets_override != null) {
          return opts.linked_datasets_override.filter(Boolean)
        }
        if (opts?.begin_background_intake && opts?.persist_linked_datasets?.length) {
          return [...opts.persist_linked_datasets]
        }
        return [...linkedDatasetsRef.current].filter(Boolean)
      })()

      const datasetBubbleMeta =
        attachedAtSend.length > 0 ? { linkedDatasetKeys: attachedAtSend } : {}

      if (topic) {
        addMessage("user", topic, { apiPayload: content, ...datasetBubbleMeta })
      } else {
        addMessage("user", content, datasetBubbleMeta)
      }

      if (isRunning) {
        addMessage("agent", "Please wait — a task is still running.")
        return
      }

      if (attachedAtSend.length > 0) {
        setLinkedDatasets([])
      }

      streamControllerRef.current?.abort()

      if (opts?.begin_background_intake) {
        setBackgroundIntake(true)
      }

      void (async () => {
        setIsRunning(true)
        const eid = await ensureExperimentId(
          suggestExperimentTitleFromUserMessage(topic || content),
          linkedForCreate ?? null,
        )
        if (!eid) {
          setIsRunning(false)
          addMessage("system", "Could not create an experiment. Check the backend connection.")
          return
        }

        if (opts?.begin_background_intake && opts.persist_linked_datasets?.length) {
          const ids = [...opts.persist_linked_datasets]
          void updateExperiment(eid, { linked_datasets: ids })
        }

        if (
          !forceOrch &&
          opts?.mode === "train" &&
          linkedForApi &&
          linkedForApi.length > 0
        ) {
          void updateExperiment(eid, { linked_datasets: [] })
          setSteps(createInitialSteps())
          emittedStepsRef.current = new Set()
          pipelineCompletionEmittedRef.current = false
          lastTrainingSummaryRef.current = null
          setProgress(0)
        }

        const conversation = buildPlanningConversation(messagesRef.current, content)
        streamControllerRef.current = streamChat(
          {
            message: content,
            experiment_id: eid,
            linked_datasets: linkedForApi,
            model_preference: opts?.user_model_preference ?? null,
            conversation,
            force_orchestrator: forceOrch,
            background_intake:
              backgroundIntakeActiveRef.current || opts?.begin_background_intake === true,
            mode: opts?.mode,
          },
          applyAgentStreamEvent,
          (error: Error) => {
            setIsRunning(false)
            addMessage("system", `Chat error: ${error.message}`)
          },
        )
      })()
    },
    [addMessage, isRunning, applyAgentStreamEvent, ensureExperimentId, setBackgroundIntake],
  )

  // Reset everything
  const reset = useCallback(() => {
    if (streamControllerRef.current) {
      streamControllerRef.current.abort()
      streamControllerRef.current = null
    }
    setAgentState(createInitialState())
    setSteps(createInitialSteps())
    setMessages([])
    setIsRunning(false)
    setCurrentJobId(null)
    setProgress(0)
    setConfirmationRequest(null)
    setAcceptAllMode(false)
    acceptAllModeRef.current = false
    emittedStepsRef.current = new Set()
    pipelineCompletionEmittedRef.current = false
    lastTrainingSummaryRef.current = null
    setLinkedDatasets([])
    setBackgroundIntake(false)
    suppressPostPlanTokensRef.current = false
    setProgressPhaseHint(null)
  }, [setBackgroundIntake])

  const leaveLabSession = useCallback(() => {
    agentDebug("leave-lab-session", {
      previousExperimentId: experimentIdRef.current,
      previousMessagesCount: messagesRef.current.length,
    })
    ++loadGenerationRef.current
    reset()
    setExperimentId(null)
    experimentIdRef.current = null
  }, [reset])

  const applyExperimentDetail = useCallback((exp: ExperimentDetail) => {
    agentDebug("apply-experiment-detail-start", {
      experimentId: exp.id,
      chatHistoryCount: Array.isArray(exp.chat_history) ? exp.chat_history.length : null,
      hasTrainingState: !!exp.training_state,
      linkedDatasetsCount: Array.isArray(exp.linked_datasets) ? exp.linked_datasets.length : 0,
    })
    streamControllerRef.current?.abort()

    setExperimentId(exp.id)

    if (exp.chat_history && Array.isArray(exp.chat_history)) {
      setMessages(
        exp.chat_history
          .filter((m: Record<string, unknown>) => m.id !== GRAPH_THINKING_MSG_ID)
          .map((m: Record<string, unknown>) => ({
            id: (m.id as string) || uid("msg"),
            role: (m.role as ChatMessage["role"]) || "agent",
            content: (m.content as string) || "",
            timestamp: (m.timestamp as number) || Date.now(),
            ...(typeof m.step_id === "string" ? { stepId: m.step_id } : {}),
            ...(typeof m.stepId === "string" ? { stepId: m.stepId } : {}),
            ...(typeof m.detail_markdown === "string" ? { detailMarkdown: m.detail_markdown } : {}),
            ...(typeof m.detailMarkdown === "string" ? { detailMarkdown: m.detailMarkdown } : {}),
            ...(m.show_report_button === true ? { showReportButton: true } : {}),
            ...(m.showReportButton === true ? { showReportButton: true } : {}),
            ...(m.task_plan && typeof m.task_plan === "object"
              ? { taskPlan: m.task_plan as ChatTaskPlanPayload }
              : {}),
            ...(m.task_plan_resolved === true ? { taskPlanResolved: true } : {}),
            ...(typeof m.api_payload === "string" ? { apiPayload: m.api_payload } : {}),
            ...(typeof m.apiPayload === "string" ? { apiPayload: m.apiPayload } : {}),
            ...(Array.isArray(m.linked_dataset_keys)
              ? { linkedDatasetKeys: (m.linked_dataset_keys as unknown[]).map(String) }
              : {}),
            ...(Array.isArray(m.linkedDatasetKeys)
              ? { linkedDatasetKeys: (m.linkedDatasetKeys as unknown[]).map(String) }
              : {}),
          })),
      )
    } else {
      setMessages([])
    }

    const ts = exp.training_state && typeof exp.training_state === "object"
      ? (exp.training_state as Record<string, unknown>)
      : null
    if (ts) {
      // Replace state per experiment to avoid stale fields leaking across switches.
      setAgentState({ ...createInitialState(), ...(ts as Partial<TrainingAgentState>) })
    } else {
      setAgentState(createInitialState())
    }

    const initialSteps = createInitialSteps()
    let resolvedSteps = ts ? mergeTaskProgressIntoSteps(initialSteps, ts) : initialSteps
    setIsRunning(false)
    setStartingHandsOffTask(false)
    setCurrentJobId(null)
    setProgress(0)
    setAcceptAllMode(false)
    emittedStepsRef.current = new Set()
    const rawLd = exp.linked_datasets
    setLinkedDatasets(Array.isArray(rawLd) ? rawLd.map(String) : [])
    backgroundIntakeActiveRef.current = false
    setBackgroundIntakeActive(false)
    suppressPostPlanTokensRef.current = false

    const pending = ts?.pending_interrupt as
      | { node: string; summary?: string; message?: string; state_snapshot?: Record<string, unknown> }
      | null
      | undefined
    if (pending && typeof pending === "object" && pending.node) {
      const nodeName = pending.node
      const stepDef = STEP_DEFINITIONS.find((s) => s.id === nodeName)
      const interruptIndex = STEP_ORDER_IDS.indexOf(nodeName)
      resolvedSteps = resolvedSteps.map((step, index) => {
        if (step.id === nodeName) {
          return { ...step, status: "awaiting_confirmation" as const, endTime: step.endTime ?? Date.now() }
        }
        if (index < interruptIndex && step.status === "pending") {
          return { ...step, status: "completed" as const }
        }
        return step
      })
      setConfirmationRequest({
        step: nodeName,
        stepName: stepDef?.name || nodeName,
        summary: typeof pending.summary === "string" ? pending.summary : "",
        details: (pending.state_snapshot || {}) as Record<string, unknown>,
      })
      agentDebug("restored-pending-interrupt", { node: nodeName })
    } else {
      setConfirmationRequest(null)
    }

    setSteps(resolvedSteps)
    agentDebug("apply-experiment-detail-finish", {
      experimentId: exp.id,
      appliedMessagesCount: Array.isArray(exp.chat_history) ? exp.chat_history.length : 0,
    })
  }, [])

  const refreshExperimentTraining = useCallback(async () => {
    const id = experimentIdRef.current
    if (!id) return
    try {
      const exp = await getExperiment(id)
      const ts = exp.training_state && typeof exp.training_state === "object"
        ? (exp.training_state as Record<string, unknown>)
        : null
      if (ts) {
        setAgentState((prev) => ({ ...prev, ...(ts as Partial<TrainingAgentState>) }))
        setSteps((prev) => mergeTaskProgressIntoSteps(prev, ts))
      }
    } catch {
      // ignore
    }
  }, [])

  const markTaskPlanResolved = useCallback((messageId: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === messageId ? { ...m, taskPlanResolved: true } : m)),
    )
  }, [])

  const startGuidedTrainingFromPlan = useCallback(
    (plan: TaskPlanSummary, refs: string[]) => {
      setBackgroundIntake(false)
      const content = plan.goal.trim() || "Run the agreed training plan."
      sendMessage(content, {
        user_model_preference: plan.preferences ?? undefined,
        linked_datasets_override: refs,
        mode: "train",
      })
    },
    [sendMessage, setBackgroundIntake],
  )
  startGuidedTrainingFromPlanRef.current = startGuidedTrainingFromPlan

  const startHandsOffTrainingFromPlan = useCallback(
    async (plan: TaskPlanSummary, refs: string[]) => {
      setStartingHandsOffTask(true)
      setBackgroundIntake(false)
      try {
        const connected = await checkConnection()
        if (!connected) {
          addMessage("system", "Backend not connected.")
          return
        }
        const eid = experimentIdRef.current
        if (!eid) {
          addMessage(
            "system",
            "Send a message in chat first so your experiment exists before starting a background task.",
          )
          return
        }
        await updateExperiment(eid, { goal: plan.goal, linked_datasets: refs })
        await updateExperiment(eid, {
          training_state_merge: {
            lab_mode: "task",
            task_plan: { ...plan, datasetRefs: refs },
            task_status: "pending",
          },
        })
        const conv = buildPlanningConversation(messagesRef.current, plan.goal)
        await startExperimentAsyncTrain(eid, {
          user_model_preference: plan.preferences ?? null,
          conversation: conv.map((c) => ({ role: c.role, content: c.content })),
        })
        const exp = await queryClient.fetchQuery({
          queryKey: experimentKeys.detail(eid),
          queryFn: () => getExperiment(eid),
        })
        applyExperimentDetail(exp)
        toastBackgroundRunStarted()
      } catch (e) {
        addMessage(
          "system",
          `Could not start background task: ${e instanceof Error ? e.message : String(e)}`,
        )
      } finally {
        setStartingHandsOffTask(false)
      }
    },
    [addMessage, checkConnection, applyExperimentDetail, queryClient, setBackgroundIntake],
  )

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamControllerRef.current) {
        streamControllerRef.current.abort()
      }
    }
  }, [])

  return {
    agentState,
    steps,
    messages,
    isRunning,
    isBackendConnected,
    currentJobId,
    progress,
    confirmationRequest,
    experimentId,
    backgroundIntakeActive,
    startingHandsOffTask,
    linkedDatasets,
    attachedDatasetSnapshots,
    updateLinkedDatasets,
    datasets,
    modelTypes,
    trainedModels,
    trainedModelsLoading,
    startAgent,
    sendMessage,
    handleConfirmation,
    reset,
    checkConnection,
    refreshDatasets,
    refreshModelTypes,
    refreshTrainedModels,
    setExperimentId,
    applyExperimentDetail,
    saveCurrentMessages,
    leaveLabSession,
    runningStepHint,
    refreshExperimentTraining,
    markTaskPlanResolved,
    startGuidedTrainingFromPlan,
    startHandsOffTrainingFromPlan,
  }
}
