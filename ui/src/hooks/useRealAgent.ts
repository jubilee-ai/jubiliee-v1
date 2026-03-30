/**
 * Training/chat agent hook (FastAPI backend).
 *
 * Shipped shape: one active experiment in React state; loadExperiment swaps that slice.
 * Target doc ui/architecture-reference.html describes an ideal per-experiment map +
 * ExperimentSessionContext in the reducer — stronger isolation for concurrent sessions, not required
 * for a single active experiment.
 */

import { useState, useCallback, useRef, useEffect } from "react"
import type {
  TrainingAgentState,
  StepInfo,
  ChatMessage,
  ConfirmationRequest,
  ConfirmationAction,
} from "@/types/agent"
import {
  checkHealth,
  getDatasets,
  getModelTypes,
  streamChat,
  getExperiment,
  saveExperimentMessages,
  createExperiment,
  type Dataset,
  type ModelType,
  type AgentStreamEvent,
  updateExperiment,
} from "@/lib/api"
import {
  uid,
  suggestExperimentTitleFromLinkedDatasets,
  suggestExperimentTitleFromUserMessage,
} from "@/lib/utils"
import { buildStepDetailMarkdown } from "@/hooks/stepStreamDetails"

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
  { id: "training_approval", name: "Training Config", description: "Propose hyperparameters and strategy" },
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

/** Shown next to the loading indicator while a checklist step is active */
const STEP_LOADING_HINTS: Record<string, string> = {
  data_collection: "Loading your dataset…",
  select_model: "Choosing the model family…",
  cleaning: "Cleaning and standardizing columns…",
  label_split_definition: "Defining the target and train/validation/test splits…",
  feature_specification_and_engineering: "Specifying and building features…",
  feature_selection_specification: "Analyzing columns, correlations, and leakage…",
  feature_engineering_executor: "Encoding features and checking matrix shapes…",
  feature_experiment_runner: "Running feature experiments and picking the best variant…",
  training_approval: "Preparing training configuration…",
  training: "Training models and comparing validation metrics…",
  generate_report: "Writing the final report…",
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

  return prev.map((step, index) => {
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

/** Build user/agent turns for the backend planner (`messages` is pre-send snapshot; `latestUserText` is the outgoing instruction). */
function buildPlanningConversation(
  messages: ChatMessage[],
  latestUserText: string,
): Array<{ role: "user" | "agent"; content: string }> {
  const trimmed = latestUserText.trim()
  const turns: Array<{ role: "user" | "agent"; content: string }> = []
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "agent") continue
    const c = m.content.trim()
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
  
  // Actions
  startAgent: (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => Promise<void>
  sendMessage: (content: string, opts?: { user_model_preference?: string }) => void
  linkedDatasets: string[]
  updateLinkedDatasets: (ids: string[]) => void
  handleConfirmation: (action: ConfirmationAction, comment?: string) => void
  reset: () => void
  /** Ping `/api/health` only (for status banner + pre-flight). Does not refetch datasets/models. */
  checkConnection: () => Promise<boolean>
  /** Refetch dataset catalog (e.g. when opening the Datasets tab). */
  refreshDatasets: () => Promise<void>
  /** Refetch model types (e.g. when opening the Models tab). */
  refreshModelTypes: () => Promise<void>
  setExperimentId: (id: string | null) => void
  loadExperiment: (id: string) => Promise<void>
  saveCurrentMessages: () => Promise<void>
  /** Clear local session and deselect experiment (experiment row remains in the list). */
  leaveLabSession: () => void
  /** Human-readable hint for what the pipeline is doing right now */
  runningStepHint: string | null
}

export function useRealAgent(options?: UseRealAgentOptions): UseRealAgentReturn {
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


  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null)
  const [acceptAllMode, setAcceptAllMode] = useState(false)
  const [experimentId, setExperimentId] = useState<string | null>(null)
  const [linkedDatasets, setLinkedDatasets] = useState<string[]>([])
  
  // Use a ref to track accept-all mode to avoid stale closure issues in callbacks
  const acceptAllModeRef = useRef(acceptAllMode)
  acceptAllModeRef.current = acceptAllMode
  
  const messagesRef = useRef<ChatMessage[]>([])
  messagesRef.current = messages

  const experimentIdRef = useRef<string | null>(experimentId)
  experimentIdRef.current = experimentId

  const linkedDatasetsRef = useRef<string[]>(linkedDatasets)
  linkedDatasetsRef.current = linkedDatasets

  const streamControllerRef = useRef<AbortController | null>(null)
  const linkedDatasetsPatchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const emittedStepsRef = useRef<Set<string>>(new Set())
  /** Prevents duplicate “pipeline finished” messages (e.g. React Strict Mode double-invoke). */
  const pipelineCompletionEmittedRef = useRef(false)
  const agentStateRef = useRef(agentState)
  agentStateRef.current = agentState
  /** Stream end can run before React applies training_metrics to state; cache from step.complete. */
  const lastTrainingSummaryRef = useRef<Record<string, unknown> | null>(null)

  const [runningStepHint, setRunningStepHint] = useState<string | null>(null)

  // Add a message to the chat (optional step metadata for expandable details + report CTA)
  const addMessage = useCallback(
    (
      role: ChatMessage["role"],
      content: string,
      meta?: Partial<Pick<ChatMessage, "stepId" | "detailMarkdown" | "showReportButton">>,
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

  const saveCurrentMessages = useCallback(async () => {
    const eid = experimentIdRef.current
    const msgs = messagesRef.current
    if (!eid || msgs.length === 0) return
    try {
      await saveExperimentMessages(
        eid,
        msgs.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          timestamp: m.timestamp,
          ...(m.stepId ? { stepId: m.stepId } : {}),
          ...(m.detailMarkdown ? { detailMarkdown: m.detailMarkdown } : {}),
          ...(m.showReportButton ? { showReportButton: m.showReportButton } : {}),
        })),
      )
    } catch {
      // best-effort save
    }
  }, [])

  const refreshDatasets = useCallback(async () => {
    try {
      setDatasets(await getDatasets())
    } catch {
      setDatasets([])
    }
  }, [])

  const refreshModelTypes = useCallback(async () => {
    try {
      setModelTypes(await getModelTypes())
    } catch {
      setModelTypes([])
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

  // One-time: health + catalog when the app loads (not on every periodic ping).
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const connected = await checkHealth()
      if (cancelled) return
      setIsBackendConnected(connected)
      if (connected) {
        const [datasetsData, modelsData] = await Promise.all([
          getDatasets().catch(() => []),
          getModelTypes().catch(() => []),
        ])
        if (!cancelled) {
          setDatasets(datasetsData)
          setModelTypes(modelsData)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!isRunning) {
      setRunningStepHint(null)
      return
    }
    if (confirmationRequest) {
      setRunningStepHint("Waiting for your review…")
      return
    }
    const running = steps.find((s) => s.status === "running")
    if (running) {
      setRunningStepHint(STEP_LOADING_HINTS[running.id] ?? `Running ${running.name}…`)
      return
    }
    if (steps.length > 0 && steps.every((s) => s.status === "pending")) {
      setRunningStepHint("Planning your pipeline…")
      return
    }
    setRunningStepHint(null)
  }, [isRunning, steps, confirmationRequest])

  // Derive a brief subtitle for a completed step
  const computeStepSubtitle = useCallback((nodeName: string, summary?: Record<string, unknown>): string => {
    if (!summary) return ""
    switch (nodeName) {
      case "select_model":
        return summary.selected_model ? String(summary.selected_model) : ""
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
      case "label_split_definition":
        return summary.target_column
          ? `Target: ${summary.target_column}`
          : ""
      case "feature_selection_specification":
        return summary.num_features
          ? `${summary.num_features} features`
          : ""
      case "feature_engineering_executor": {
        const spec = summary.num_spec_features
        const created = Array.isArray(summary.features_created) ? summary.features_created.length : 0
        if (spec && created) return `${spec} → ${created} columns`
        if (created) return `${created} columns`
        return ""
      }
      case "feature_specification_and_engineering": {
        const n = summary.num_features
        const created = Array.isArray(summary.features_created) ? summary.features_created.length : 0
        const passed = summary.validation_passed
        const parts: string[] = []
        if (n) parts.push(`${n} specified`)
        if (created) parts.push(`${created} columns`)
        if (passed !== undefined) parts.push(passed ? "ok" : "issues")
        return parts.join(" · ")
      }
      case "feature_experiment_runner": {
        if (summary.skipped) return "skipped"
        const bv = summary.best_variant_name
        const tv = summary.total_variants
        const parts: string[] = []
        if (bv) parts.push(String(bv))
        if (tv != null) parts.push(`${tv} setups`)
        return parts.join(" · ")
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
      return
    }
    if (event.type === "token") {
      const phase = event.phase
      const isThinkingPhase = phase === "planner" || phase === "evaluator" || phase === "select_model"
      if (isThinkingPhase) {
        return
      }

      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (last && last.role === "agent" && last._streaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, content: last.content + (event.content || "") },
          ]
        }
        return [
          ...prev,
          { id: uid("msg"), role: "agent", content: event.content || "", timestamp: Date.now(), _streaming: true },
        ]
      })
      return
    }
    if (event.type === "tool_call" || event.type === "tool.start") {
      const toolName = event.tool || "unknown"
      const toolLabel =
        toolName === "analyze_data" ? "Running analysis…" :
        toolName === "train_model" ? "Training model (this may take several minutes)…" :
        toolName === "check_trained_models" ? "Checking trained models…" :
        toolName === "predict" ? "Making prediction…" :
        toolName === "get_model_details" ? "Loading model details…" :
        `Running ${toolName}…`
      addMessage("system", toolLabel)
      return
    }
    if (event.type === "tool_result" || event.type === "tool.end") {
      return
    }
    if (event.type === "predict.start") {
      addMessage("system", `Running predictions with **${event.model || "model"}**…`)
      return
    }
    if (event.type === "predict.complete") {
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
      setMessages((prev) =>
        prev.map((m) => (m._streaming ? { ...m, _streaming: undefined } : m)),
      )
      setIsRunning(false)
      setTimeout(() => saveCurrentMessages(), 100)
      return
    }

    if (event.type === "thinking" || event.type === "step.progress") {
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

      const headline =
        typeof event.headline === "string" && event.headline.trim()
          ? event.headline.trim()
          : `${STEP_DEFINITIONS.find((s) => s.id === nodeName)?.name || nodeName} complete`

      const detailMarkdown = buildStepDetailMarkdown(nodeName, event)
      const stepIdForMessage =
        nodeName === "cleaning_and_standardization" ? "cleaning" : nodeName

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

      setSteps((prev) => applyNodeSkippedToSteps(prev, nodeName))
      const skipSubtitle = subtitleFromSkippedEvent(event.summary)
      if (skipSubtitle) {
        setSteps((prev) =>
          prev.map((step) => (step.id === nodeName ? { ...step, subtitle: skipSubtitle } : step)),
        )
      }
    } else if (event.type === "completed" || (event.type === "stream.end" && event.pipeline_completed === true)) {
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
        `**Pipeline finished.** ${recap}\n\nOpen the report for feature importance, comparisons, and next steps.`,
        { showReportButton: true, stepId: "generate_report" },
      )
    } else if (event.type === "error") {
      setIsRunning(false)
      
      // Mark current running step as error
      setSteps((prev) => prev.map((step) => 
        step.status === "running" ? { ...step, status: "error" } : step
      ))
      
      addMessage("system", `Error: ${event.error || "Unknown error"}`)
    }
  }, [addMessage, computeStepSubtitle, saveCurrentMessages])

  const ensureExperimentId = useCallback(
    async (suggestedName?: string | null): Promise<string | null> => {
      if (experimentIdRef.current) return experimentIdRef.current
      try {
        const name = suggestedName?.trim() || undefined
        const exp = await createExperiment(name)
        setExperimentId(exp.id)
        experimentIdRef.current = exp.id
        onExperimentEnsuredRef.current?.(exp.id)
        return exp.id
      } catch {
        return null
      }
    },
    [],
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
      void (async () => {
        if (!experimentIdRef.current) {
          if (ids.length === 0) return
          try {
            const title = suggestExperimentTitleFromLinkedDatasets(ids)
            const exp = await createExperiment(title, ids)
            setExperimentId(exp.id)
            experimentIdRef.current = exp.id
            onExperimentEnsuredRef.current?.(exp.id)
          } catch {
            // keep local selection; persist can retry when user has an experiment
          }
          return
        }
        scheduleLinkedDatasetsPatch(ids)
      })()
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

    const runEid = await ensureExperimentId(suggestExperimentTitleFromUserMessage(goal))
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
        addMessage("system", "✓ Step accepted")
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
      opts?: { user_model_preference?: string },
    ) => {
      addMessage("user", content)

      if (isRunning) {
        addMessage("agent", "Please wait — a task is still running.")
        return
      }

      streamControllerRef.current?.abort()

      const linkedForThisSend = [...linkedDatasetsRef.current]

      void (async () => {
        const eid = await ensureExperimentId(suggestExperimentTitleFromUserMessage(content))
        if (!eid) {
          addMessage("system", "Could not create an experiment. Check the backend connection.")
          return
        }

        if (linkedForThisSend.length > 0) {
          setLinkedDatasets([])
          void updateExperiment(eid, { linked_datasets: [] })
          setSteps(createInitialSteps())
          emittedStepsRef.current = new Set()
          pipelineCompletionEmittedRef.current = false
          lastTrainingSummaryRef.current = null
          setProgress(0)
        }

        setIsRunning(true)

        const conversation = buildPlanningConversation(messagesRef.current, content)
        streamControllerRef.current = streamChat(
          {
            message: content,
            experiment_id: eid,
            linked_datasets: linkedForThisSend.length > 0 ? linkedForThisSend : null,
            model_preference: opts?.user_model_preference ?? null,
            conversation,
          },
          applyAgentStreamEvent,
          (error: Error) => {
            setIsRunning(false)
            addMessage("system", `Chat error: ${error.message}`)
          },
        )
      })()
    },
    [
      addMessage,
      isRunning,
      applyAgentStreamEvent,
      ensureExperimentId,
    ],
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
  }, [])

  const leaveLabSession = useCallback(() => {
    reset()
    setExperimentId(null)
    experimentIdRef.current = null
  }, [reset])

  const loadExperiment = useCallback(async (id: string) => {
    try {
      // Save current experiment's messages before switching
      await saveCurrentMessages()

      const exp = await getExperiment(id)
      streamControllerRef.current?.abort()

      setExperimentId(exp.id)

      // Restore chat history
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
            })),
        )
      } else {
        setMessages([])
      }

      // Restore training state if available
      if (exp.training_state) {
        setAgentState((prev) => ({ ...prev, ...(exp.training_state as Partial<TrainingAgentState>) }))
      } else {
        setAgentState(createInitialState())
      }

      setSteps(createInitialSteps())
      setIsRunning(false)
      setConfirmationRequest(null)
      setAcceptAllMode(false)
      emittedStepsRef.current = new Set()
      const rawLd = exp.linked_datasets
      setLinkedDatasets(Array.isArray(rawLd) ? rawLd.map(String) : [])
    } catch (err) {
      console.error("Failed to load experiment:", err)
    }
  }, [])

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
    linkedDatasets,
    updateLinkedDatasets,
    datasets,
    modelTypes,
    startAgent,
    sendMessage,
    handleConfirmation,
    reset,
    checkConnection,
    refreshDatasets,
    refreshModelTypes,
    setExperimentId,
    loadExperiment,
    saveCurrentMessages,
    leaveLabSession,
    runningStepHint,
  }
}
