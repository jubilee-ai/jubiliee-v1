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

/** Ephemeral merged "thinking" lines from graph custom stream (cleared on each new `started`). */
const GRAPH_THINKING_MSG_ID = "__graph_thinking__"

// Step definitions matching the agent's tool set (agent_simple.py _STEP_ORDER)
const STEP_DEFINITIONS = [
  { id: "data_collection", name: "Data Collection", description: "Load or collect the dataset" },
  { id: "select_model", name: "Model Selection", description: "Choose the ML model type for this task" },
  { id: "cleaning", name: "Cleaning", description: "Clean and standardize the data" },
  { id: "label_split_definition", name: "Label & Split", description: "Define target column and train/val/test splits" },
  { id: "feature_selection_specification", name: "Feature Selection", description: "Analyze data and specify features" },
  { id: "feature_engineering_executor", name: "Feature Engineering", description: "Execute feature transformations" },
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
  /** False until the first `/api/health` check on mount completes (success or failure). */
  isBackendReachabilityKnown: boolean
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
  /** Model type id linked for this experiment session (sent with each message / training run). */
  linkedModelId: string | null
  setLinkedModelId: (id: string | null) => void
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
}

export function useRealAgent(options?: UseRealAgentOptions): UseRealAgentReturn {
  const onExperimentEnsuredRef = useRef(options?.onExperimentEnsured)
  onExperimentEnsuredRef.current = options?.onExperimentEnsured
  const [agentState, setAgentState] = useState<TrainingAgentState>(createInitialState())
  const [steps, setSteps] = useState<StepInfo[]>(createInitialSteps())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isBackendConnected, setIsBackendConnected] = useState(false)
  const [isBackendReachabilityKnown, setIsBackendReachabilityKnown] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [modelTypes, setModelTypes] = useState<ModelType[]>([])


  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null)
  const [acceptAllMode, setAcceptAllMode] = useState(false)
  const [experimentId, setExperimentId] = useState<string | null>(null)
  const [linkedDatasets, setLinkedDatasets] = useState<string[]>([])
  const [linkedModelId, setLinkedModelId] = useState<string | null>(null)
  
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

  // Add a message to the chat
  const addMessage = useCallback((role: ChatMessage["role"], content: string) => {
    setMessages((prev) => [
      ...prev,
      { id: uid("msg"), role, content, timestamp: Date.now() },
    ])
  }, [])

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
      try {
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
      } catch {
        if (!cancelled) setIsBackendConnected(false)
      } finally {
        if (!cancelled) setIsBackendReachabilityKnown(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // Format detailed stream event for display
  const formatStreamDetails = useCallback((event: AgentStreamEvent): string => {
    const details = event.details as Record<string, unknown> | undefined
    const summary = event.summary as Record<string, unknown> | undefined
    
    console.log("[formatStreamDetails] node:", event.node, "details:", !!details, "summary:", !!summary)
    
    if (!details && !summary) return ""
    
    const lines: string[] = []
    
    // Use details.title and description if available
    if (details?.title) {
      lines.push(`**${String(details.title)}**`)
    }
    if (details?.description) {
      lines.push(String(details.description))
    }
    
    // Add step-specific formatted info
    const nodeName = event.node || ""
    
    if (nodeName === "select_model" && summary) {
      if (summary.selected_model) {
        lines.push(`\nModel: **${summary.selected_model}**`)
      }
      if (summary.explanation) {
        const explanation = String(summary.explanation)
        lines.push(`\n${explanation.slice(0, 300)}${explanation.length > 300 ? "..." : ""}`)
      }
    } else if (nodeName === "data_collection" && summary) {
      if (summary.dataset) lines.push(`Dataset: \`${summary.dataset}\``)
      if (summary.rows) lines.push(`Rows: ${summary.rows}`)
      if (summary.columns && Array.isArray(summary.columns)) {
        lines.push(`Columns (${summary.columns.length}): ${summary.columns.slice(0, 8).join(", ")}${summary.columns.length > 8 ? "..." : ""}`)
      }
    } else if (nodeName === "cleaning_and_standardization" && summary) {
      // Show cleaned dataset info
      if (summary.cleaned_dataset) lines.push(`Output: \`${summary.cleaned_dataset}\``)
      if (summary.rows && summary.rows !== "unknown") lines.push(`Rows: ${summary.rows}`)
      if (summary.columns && summary.columns !== "unknown") lines.push(`Columns: ${summary.columns}`)
      
      // Show transformations applied with full details
      if (summary.transformations && Array.isArray(summary.transformations) && summary.transformations.length > 0) {
        lines.push(`\n**Transformations Applied (${summary.transformations.length}):**\n`)
        summary.transformations.slice(0, 15).forEach((t: unknown) => {
          if (typeof t === "object" && t !== null) {
            const transform = t as Record<string, unknown>
            const toolName = String(transform.tool || transform.op || "transform").replace(/_tool$/, "")
            const args = transform.args as Record<string, unknown> | undefined
            const result = transform.result as string | undefined
            
            // Format tool name and columns
            let argsStr = ""
            if (args) {
              if (args.columns) {
                const cols = Array.isArray(args.columns) ? (args.columns as string[]).join(", ") : String(args.columns)
                argsStr = `**${cols}**`
              } else if (args.column) {
                argsStr = `**${args.column}**`
              }
              if (args.dataset_ref) {
                argsStr += argsStr ? ` from \`${args.dataset_ref}\`` : `\`${args.dataset_ref}\``
              }
              if (args.value !== undefined) argsStr += ` = ${args.value}`
              if (args.strategy) argsStr += ` (${args.strategy})`
            }
            
            // Show tool call
            lines.push(`\`${toolName}\` ${argsStr}`)
            
            // Show result on next line
            if (result) {
              lines.push(`  → ${result}`)
            }
            lines.push("") // blank line between transformations
          }
        })
        if (summary.transformations.length > 15) {
          lines.push(`*... and ${summary.transformations.length - 15} more transformations*`)
        }
      } else if (summary.num_transformations === 0) {
        lines.push(`\n*No transformations needed - data was already clean*`)
      }
      
      // Show reason at the end
      if (summary.reason) {
        lines.push(`**Summary:** ${summary.reason}`)
      }
    } else if (nodeName === "label_split_definition" && summary) {
      if (summary.target_column) lines.push(`Target column: **${summary.target_column}**`)
      if (summary.split_strategy) lines.push(`Split strategy: ${summary.split_strategy}`)
      if (summary.grain) lines.push(`Grain: ${summary.grain}`)
      if (summary.train_ref) lines.push(`\nTrain: \`${summary.train_ref}\``)
      if (summary.val_ref) lines.push(`Validation: \`${summary.val_ref}\``)
      if (summary.test_ref) lines.push(`Test: \`${summary.test_ref}\``)
    } else if (nodeName === "feature_selection_specification" && summary) {
      try {
        // Summary text
        if (summary.summary_text) {
          lines.push(`**Summary:** ${String(summary.summary_text)}`)
          lines.push("")
        }
        
        // Dataset overview
        const overview = summary.dataset_overview
        if (overview && typeof overview === "object") {
          const ov = overview as Record<string, unknown>
          if (ov.rows) {
            lines.push(`📊 **Dataset Overview**`)
            lines.push(`- Rows: ${String(ov.rows).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}`)
            lines.push(`- Columns: ${ov.columns || "N/A"}`)
            if (ov.numeric_columns != null || ov.categorical_columns != null) {
              lines.push(`- Numeric: ${ov.numeric_columns || 0}, Categorical: ${ov.categorical_columns || 0}`)
            }
            lines.push("")
          }
        }
        
        // Target analysis
        const target = summary.target_analysis
        if (target && typeof target === "object") {
          const t = target as Record<string, unknown>
          if (t.type) {
            lines.push(`🎯 **Target Column Analysis**`)
            if (t.type === "numeric") {
              lines.push(`- Type: Numeric`)
              if (t.mean != null) lines.push(`- Mean: ${Number(t.mean).toFixed(4)}`)
              if (t.std != null) lines.push(`- Std Dev: ${Number(t.std).toFixed(4)}`)
              if (t.min != null && t.max != null) lines.push(`- Range: [${Number(t.min).toFixed(2)}, ${Number(t.max).toFixed(2)}]`)
            } else if (t.type === "categorical") {
              lines.push(`- Type: Categorical`)
              if (t.unique_values) lines.push(`- Unique values: ${t.unique_values}`)
              if (t.top_value) lines.push(`- Most common: "${t.top_value}" (${t.top_freq} occurrences)`)
            }
            lines.push("")
          }
        }
        
        // Feature correlations with target
        const correlations = summary.feature_correlations
        if (correlations && Array.isArray(correlations) && correlations.length > 0) {
          lines.push(`📈 **Top Feature Correlations with Target**`)
          correlations.slice(0, 5).forEach((c: unknown, i: number) => {
            if (c && typeof c === "object") {
              const corr = c as {feature?: string; correlation?: number}
              const corrValue = Number(corr.correlation || 0)
              const bar = corrValue >= 0 ? "▓".repeat(Math.min(10, Math.round(Math.abs(corrValue) * 10))) : "░".repeat(Math.min(10, Math.round(Math.abs(corrValue) * 10)))
              lines.push(`${i + 1}. **${corr.feature || "unknown"}**: ${corrValue >= 0 ? '+' : ''}${corrValue.toFixed(4)} ${bar}`)
            }
          })
          lines.push("")
        }
        
        // High correlation pairs (multicollinearity)
        const highCorr = summary.high_correlation_pairs
        if (highCorr && Array.isArray(highCorr) && highCorr.length > 0) {
          lines.push(`⚠️ **High Correlation Pairs** (potential multicollinearity)`)
          highCorr.slice(0, 3).forEach((p: unknown) => {
            if (p && typeof p === "object") {
              const pair = p as {feature1?: string; feature2?: string; correlation?: number}
              lines.push(`- ${pair.feature1 || "?"} ↔ ${pair.feature2 || "?"}: ${Number(pair.correlation || 0).toFixed(4)}`)
            }
          })
          lines.push("")
        }
        
        // Leakage warnings
        const leakage = summary.leakage_warnings
        if (leakage && Array.isArray(leakage) && leakage.length > 0) {
          lines.push(`🚨 **Leakage Warnings**`)
          leakage.forEach((f: unknown) => {
            lines.push(`- ⚠️ ${String(f)}`)
          })
          lines.push("")
        }
        
        // Distribution stats
        const distStats = summary.distribution_stats
        if (distStats && Array.isArray(distStats) && distStats.length > 0) {
          const skewedCols = distStats.filter((d: unknown) => {
            if (d && typeof d === "object") {
              const stat = d as {skewness?: number}
              return stat.skewness != null && Math.abs(stat.skewness) > 1
            }
            return false
          })
          const outlierCols = distStats.filter((d: unknown) => {
            if (d && typeof d === "object") {
              const stat = d as {outlier_pct?: number}
              return stat.outlier_pct != null && stat.outlier_pct > 5
            }
            return false
          })
          if (skewedCols.length > 0 || outlierCols.length > 0) {
            lines.push(`📉 **Distribution Insights**`)
            if (skewedCols.length > 0) {
              lines.push(`- Highly skewed columns: ${skewedCols.map((d: unknown) => {
                const stat = d as {column?: string; skewness?: number}
                return `${stat.column || "?"} (${stat.skewness?.toFixed(2) || "?"})`
              }).join(", ")}`)
            }
            if (outlierCols.length > 0) {
              lines.push(`- Columns with outliers: ${outlierCols.map((d: unknown) => {
                const stat = d as {column?: string; outlier_pct?: number}
                return `${stat.column || "?"} (${stat.outlier_pct?.toFixed(1) || "?"}%)`
              }).join(", ")}`)
            }
            lines.push("")
          }
        }
        
        // Features specified - always show this
        if (summary.num_features) {
          lines.push(`✅ **Features Specified: ${summary.num_features}**`)
          if (summary.feature_names && Array.isArray(summary.feature_names)) {
            const names = summary.feature_names.map((n: unknown) => String(n))
            lines.push(`\`${names.slice(0, 8).join("\`, \`")}\`${names.length > 8 ? ` ... +${names.length - 8} more` : ""}`)
          }
        }
      } catch (err) {
        console.error("[formatStreamDetails] Error formatting feature_selection_specification:", err)
        lines.push(`Feature selection completed with ${summary.num_features || "?"} features`)
      }
    } else if (nodeName === "feature_engineering_executor" && summary) {
      if (summary.features_created && Array.isArray(summary.features_created)) {
        const numCreated = summary.features_created.length
        const numSpec = summary.num_spec_features
        lines.push(`Features created: ${numCreated}`)
        if (numSpec && numCreated !== numSpec) {
          lines.push(`*(${numSpec} feature specs → ${numCreated} columns after one-hot encoding)*`)
        }
        lines.push(`Names: ${summary.features_created.slice(0, 6).join(", ")}${summary.features_created.length > 6 ? "..." : ""}`)
      }
      if (summary.shapes && typeof summary.shapes === "object") {
        const shapes = summary.shapes as Record<string, unknown>
        const fmtShape = (s: unknown) => {
          if (Array.isArray(s) && s.length >= 2) return `${s[0]} × ${s[1]}`
          return String(s || "?")
        }
        if (shapes.train) lines.push(`Train shape: ${fmtShape(shapes.train)}`)
        if (shapes.val) lines.push(`Val shape: ${fmtShape(shapes.val)}`)
        if (shapes.test) lines.push(`Test shape: ${fmtShape(shapes.test)}`)
      }
      if (summary.validation_passed !== undefined) {
        lines.push(`Validation: ${summary.validation_passed ? "✓ Passed" : "⚠ Issues found"}`)
      }
    } else if (nodeName === "training_approval" && summary) {
      if (summary.model_type) lines.push(`Model: **${summary.model_type}**`)
      if (summary.task_type) lines.push(`Task: ${summary.task_type}`)
      const hp = summary.hyperparameters
      if (hp && typeof hp === "object") {
        const entries = Object.entries(hp as Record<string, unknown>)
        const keyParams = entries.slice(0, 4).map(([k, v]) => `\`${k}=${v}\``).join(", ")
        if (keyParams) {
          lines.push(`Key params: ${keyParams}${entries.length > 4 ? ` (+${entries.length - 4} more)` : ""}`)
        }
      }
      const strategy = summary.strategy_notes
      if (strategy) {
        const noteCount = Array.isArray(strategy) ? strategy.length : 1
        lines.push(`\nStrategy: ${noteCount} section${noteCount !== 1 ? "s" : ""} — view details for full plan`)
      }
    } else if (nodeName === "training" && summary) {
      if (summary.model_name) lines.push(`Model: **${summary.model_name}**`)
      if (summary.model_type) lines.push(`Type: ${summary.model_type}`)
      if (summary.num_iterations) lines.push(`Iterations: ${summary.num_iterations}`)
      
      // Classification metrics
      if (summary.test_accuracy != null) {
        lines.push(`\n**Classification Metrics:**`)
        lines.push(`Test Accuracy: ${(Number(summary.test_accuracy) * 100).toFixed(2)}%`)
        if (summary.test_roc_auc != null) lines.push(`Test ROC-AUC: ${Number(summary.test_roc_auc).toFixed(4)}`)
      }
      
      // Regression metrics
      if (summary.test_r2 != null || summary.val_r2 != null) {
        lines.push(`\n**Regression Metrics:**`)
        if (summary.val_r2 != null) lines.push(`Val R²: ${Number(summary.val_r2).toFixed(4)}`)
        if (summary.test_r2 != null) lines.push(`Test R²: ${Number(summary.test_r2).toFixed(4)}`)
        if (summary.test_rmse != null) lines.push(`Test RMSE: ${Number(summary.test_rmse).toFixed(2)}`)
        if (summary.test_mae != null) lines.push(`Test MAE: ${Number(summary.test_mae).toFixed(2)}`)
      }
      
      // Summary from training
      if (details?.summary) {
        const trainingSummary = String(details.summary)
        lines.push(`\n${trainingSummary.slice(0, 400)}${trainingSummary.length > 400 ? "..." : ""}`)
      }
    } else if (nodeName === "generate_report" && summary) {
      if (summary.report_path) lines.push(`Report: \`${summary.report_path}\``)
      if (summary.model_path) lines.push(`Model: \`${summary.model_path}\``)
    }
    
    return lines.join("\n")
  }, [])

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
        return `${rows} rows, ${colsStr} cols`
      }
      case "cleaning":
      case "cleaning_and_standardization":
        return summary.num_transformations != null
          ? `${summary.num_transformations} transformations`
          : ""
      case "label_split_definition":
        return summary.target_column
          ? `Target: ${summary.target_column}`
          : ""
      case "feature_selection_specification":
        return summary.num_features
          ? `${summary.num_features} features specified`
          : ""
      case "feature_engineering_executor": {
        const created = Array.isArray(summary.features_created) ? summary.features_created.length : 0
        const passed = summary.validation_passed
        return created ? `${created} features created, ${passed ? "passed" : "issues"}` : ""
      }
      case "training_approval": {
        const model = summary.model_type || ""
        const hp = summary.hyperparameters
        const hpCount = hp && typeof hp === "object" ? Object.keys(hp).length : 0
        return model ? `${model}, ${hpCount} params` : ""
      }
      case "training": {
        const parts: string[] = []
        if (summary.test_accuracy != null) parts.push(`Acc: ${(Number(summary.test_accuracy) * 100).toFixed(1)}%`)
        if (summary.test_roc_auc != null) parts.push(`AUC: ${Number(summary.test_roc_auc).toFixed(3)}`)
        if (summary.test_r2 != null) parts.push(`R²: ${Number(summary.test_r2).toFixed(4)}`)
        if (summary.test_rmse != null) parts.push(`RMSE: ${Number(summary.test_rmse).toFixed(0)}`)
        return parts.join(", ") || (summary.success ? "Completed" : "Failed")
      }
      case "generate_report":
        return "Saved"
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
        const content = event.content || ""
        if (!content) return
        setMessages((prev) => {
          const i = prev.findIndex((m) => m.id === GRAPH_THINKING_MSG_ID)
          if (i === -1) {
            return [
              ...prev,
              {
                id: GRAPH_THINKING_MSG_ID,
                role: "system",
                content: content,
                timestamp: Date.now(),
                _streaming: true,
              },
            ]
          }
          const cur = prev[i]
          const next = [...prev]
          next[i] = { ...cur, content: cur.content + content, timestamp: Date.now() }
          return next
        })
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
      const raw = (typeof event.message === "string" ? event.message.trim() : "")
        || (typeof event.phase === "string" ? event.phase.trim() : "")
      if (!raw) return
      const line = `⋯ ${raw}`
      setMessages((prev) => {
        const i = prev.findIndex((m) => m.id === GRAPH_THINKING_MSG_ID)
        if (i === -1) {
          return [
            ...prev,
            {
              id: GRAPH_THINKING_MSG_ID,
              role: "system",
              content: line,
              timestamp: Date.now(),
              _streaming: true,
            },
          ]
        }
        const cur = prev[i]
        const next = [...prev]
        next[i] = {
          ...cur,
          content: `${cur.content}\n${line}`,
          timestamp: Date.now(),
        }
        return next
      })
      return
    }

    if (event.type === "started" || (event.type === "stream.start" && event.pipeline_completed != null)) {
      setMessages((prev) => prev.filter((m) => m.id !== GRAPH_THINKING_MSG_ID))
      setProgress(0)
      addMessage("system", "Training stream started...")
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
        summary = event.summary
      } else if (nodeName === "training_approval" && event.summary && typeof event.summary === "object") {
        const s = event.summary as Record<string, unknown>
        const lines: string[] = []
        if (s.model_type) lines.push(`**Model:** ${s.model_type}`)
        if (s.task_type) lines.push(`**Task:** ${s.task_type}`)
        const hp = s.hyperparameters as Record<string, unknown> | undefined
        if (hp && typeof hp === "object") {
          const entries = Object.entries(hp)
          const keyParams = entries.slice(0, 5).map(([k, v]) => `\`${k}=${v}\``).join("  ·  ")
          lines.push("")
          lines.push(`**Hyperparameters** (${entries.length} total)`)
          lines.push(keyParams + (entries.length > 5 ? `  ·  *+${entries.length - 5} more*` : ""))
        }
        if (s.class_weight) lines.push(`\n**Class weight:** ${s.class_weight}`)
        const strategy = s.strategy_notes
        if (strategy) {
          const notes = Array.isArray(strategy) ? strategy : [strategy]
          lines.push("")
          lines.push(`**Strategy** — ${notes.length} section${notes.length !== 1 ? "s" : ""}`)
          lines.push("*View full details for the complete training plan*")
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
      addMessage("system", `Dataset loaded: ${event.dataset} → ${event.ref}`)
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

      setMessages(prev => prev.map(m =>
        m.id === GRAPH_THINKING_MSG_ID ? { ...m, _streaming: false } : m
      ))
      
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

      // Graph orchestration nodes: update checklist/state only; no chat line.
      if (nodeName === "planner" || nodeName === "dispatcher" || nodeName === "evaluator") {
        return
      }
      
      // Format and show detailed info in chat
      let formattedDetails = ""
      try {
        formattedDetails = formatStreamDetails(event)
      } catch (err) {
        console.error("[stream] Error formatting details:", err)
      }
      
      console.log("[stream] formattedDetails for", nodeName, "length:", formattedDetails?.length || 0)
      
      // For feature_selection_specification, always create a detailed message
      if (nodeName === "feature_selection_specification") {
        const summary = event.summary as Record<string, unknown> | undefined
        const details = event.details as Record<string, unknown> | undefined
        const messageLines: string[] = []
        
        messageLines.push("## ✅ Feature Selection Complete")
        messageLines.push("")
        
        if (details?.description) {
          messageLines.push(`> ${String(details.description)}`)
          messageLines.push("")
        }
        
        if (summary) {
          // Dataset overview - compact inline format
          const overview = summary.dataset_overview as Record<string, unknown> | undefined
          if (overview?.rows) {
            messageLines.push("### 📊 Dataset Overview")
            messageLines.push("")
            messageLines.push(`**${String(overview.rows).replace(/\B(?=(\d{3})+(?!\d))/g, ",")} rows** × **${overview.columns} columns** • ${overview.numeric_columns || 0} numeric • ${overview.categorical_columns || 0} categorical`)
            messageLines.push("")
          }
          
          // Numeric summaries - compact card style
          const numericSummaries = summary.numeric_summaries as Array<{column?: string; mean?: number; std?: number; min?: number; max?: number; skew?: number}> | undefined
          if (numericSummaries && Array.isArray(numericSummaries) && numericSummaries.length > 0) {
            messageLines.push("### 📈 Numeric Features")
            messageLines.push("")
            numericSummaries.slice(0, 6).forEach((s) => {
              if (s && typeof s === "object") {
                const skewWarning = s.skew && Math.abs(s.skew) > 1 ? " ⚠️" : ""
                messageLines.push(`**${s.column}**${skewWarning}`)
                messageLines.push(`Mean: \`${s.mean?.toLocaleString() ?? "N/A"}\` • Std: \`${s.std?.toLocaleString() ?? "N/A"}\` • Range: \`${s.min?.toLocaleString() ?? "?"}\` → \`${s.max?.toLocaleString() ?? "?"}\``)
                messageLines.push("")
              }
            })
            if (numericSummaries.length > 6) {
              messageLines.push(`*+ ${numericSummaries.length - 6} more numeric features*`)
              messageLines.push("")
            }
          }
          
          // Feature correlations - visual bar representation
          const correlations = summary.feature_correlations as Array<{feature?: string; correlation?: number}> | undefined
          if (correlations && Array.isArray(correlations) && correlations.length > 0) {
            messageLines.push("### 🎯 Target Correlations")
            messageLines.push("")
            correlations.slice(0, 6).forEach((c) => {
              if (c && typeof c === "object") {
                const corrValue = Number(c.correlation || 0)
                const absCorr = Math.abs(corrValue)
                const barLength = Math.round(absCorr * 20) // max 20 chars
                const bar = corrValue >= 0 ? "█".repeat(barLength) : "▓".repeat(barLength)
                const sign = corrValue >= 0 ? "+" : ""
                const color = corrValue >= 0 ? "🟢" : "🔴"
                messageLines.push(`${color} \`${c.feature?.padEnd(16) || "unknown".padEnd(16)}\` ${bar.padEnd(4)} **${sign}${corrValue.toFixed(3)}**`)
              }
            })
            messageLines.push("")
          }
          
          // Leakage warnings
          const leakage = summary.leakage_warnings as string[] | undefined
          if (leakage && Array.isArray(leakage) && leakage.length > 0) {
            messageLines.push("### 🚨 Leakage Warnings")
            messageLines.push("")
            leakage.forEach((f) => {
              messageLines.push(`> ⚠️ **${String(f)}** may cause data leakage`)
            })
            messageLines.push("")
          }
          
          // High correlation pairs
          const highCorr = summary.high_correlation_pairs as Array<{feature1?: string; feature2?: string; correlation?: number}> | undefined
          if (highCorr && Array.isArray(highCorr) && highCorr.length > 0) {
            messageLines.push("### ⚠️ Multicollinearity")
            messageLines.push("")
            highCorr.slice(0, 3).forEach((p) => {
              if (p && typeof p === "object") {
                messageLines.push(`\`${p.feature1 || "?"}\` ↔ \`${p.feature2 || "?"}\` = **${Number(p.correlation || 0).toFixed(3)}**`)
              }
            })
            messageLines.push("")
          }
          
          // Group summaries (categorical analysis) - compact horizontal
          const groupSummaries = summary.group_summaries as Array<{column?: string; n_groups?: number; groups?: Array<{value?: string; mean?: number}>}> | undefined
          if (groupSummaries && Array.isArray(groupSummaries) && groupSummaries.length > 0) {
            messageLines.push("### 📋 Categorical Feature Breakdown")
            messageLines.push("")
            groupSummaries.slice(0, 4).forEach((g) => {
              if (g && typeof g === "object" && g.groups && Array.isArray(g.groups)) {
                const groupStr = g.groups.slice(0, 4).map((grp) => {
                  if (grp && typeof grp === "object") {
                    return `${grp.value}: **${((grp.mean || 0) * 100).toFixed(0)}%**`
                  }
                  return ""
                }).filter(Boolean).join(" • ")
                messageLines.push(`**${g.column}** → ${groupStr}`)
              }
            })
            messageLines.push("")
          }
          
          // Features specified - clean list
          if (summary.num_features) {
            messageLines.push("---")
            messageLines.push("")
            messageLines.push(`### ✅ ${summary.num_features} Features Selected`)
            messageLines.push("")
            if (summary.feature_names && Array.isArray(summary.feature_names)) {
              const names = summary.feature_names.map((n: unknown) => String(n))
              messageLines.push(`\`${names.slice(0, 10).join("\` • \`")}\`${names.length > 10 ? ` *+${names.length - 10} more*` : ""}`)
            }
          }
          
          messageLines.push("")
          messageLines.push("---")
          messageLines.push("*📊 View the **Analysis** tab in the report for histograms, Lorenz curves, and more.*")
        }
        
        const finalMessage = messageLines.join("\n")
        console.log("[stream] feature_selection message:", finalMessage.substring(0, 500))
        addMessage("agent", finalMessage)
      } else if (formattedDetails && formattedDetails.trim()) {
        addMessage("agent", formattedDetails)
      } else {
        const headline = event.headline
        if (headline && typeof headline === "string") {
          addMessage("agent", `**${STEP_DEFINITIONS.find(s => s.id === nodeName)?.name || nodeName}** — ${headline}`)
        } else {
          const stepDef = STEP_DEFINITIONS.find((s) => s.id === nodeName)
          addMessage("agent", `✓ ${stepDef?.name || nodeName} complete`)
        }
      }
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
      
      addMessage("agent", "**Training completed successfully!**\n\nClick 'View Report' to see detailed results including metrics, feature importance, and recommendations.")
    } else if (event.type === "error") {
      setIsRunning(false)
      
      // Mark current running step as error
      setSteps((prev) => prev.map((step) => 
        step.status === "running" ? { ...step, status: "error" } : step
      ))
      
      addMessage("system", `Error: ${event.error || "Unknown error"}`)
    }
  }, [addMessage, formatStreamDetails, computeStepSubtitle, saveCurrentMessages])

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

    const initialState = createInitialState()
    initialState.goal = goal
    initialState.linked_datasets = linkedDatasets || null
    initialState.user_model_preference = modelPreference || linkedModelId || null
    setAgentState(initialState)

    const hitlLabel = hitl === false ? " (no human review)" : ""
    addMessage("agent", `Starting training with goal: "${goal}"\n\nStreaming progress updates in real-time${hitlLabel}...`)

    streamControllerRef.current = streamChat(
      {
        message: goal,
        linked_datasets: linkedDatasets ?? null,
        model_preference: modelPreference ?? linkedModelId ?? null,
        experiment_id: runEid,
      },
      applyAgentStreamEvent,
      (error: Error) => {
        setIsRunning(false)
        addMessage("system", `Stream error: ${error.message}`)
      },
    )
  }, [addMessage, checkConnection, applyAgentStreamEvent, linkedModelId, ensureExperimentId])

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

      void (async () => {
        const eid = await ensureExperimentId(suggestExperimentTitleFromUserMessage(content))
        if (!eid) {
          addMessage("system", "Could not create an experiment. Check the backend connection.")
          return
        }

        const ds = linkedDatasetsRef.current
        if (ds.length > 0) {
          setSteps(createInitialSteps())
          emittedStepsRef.current = new Set()
          setProgress(0)
        }

        setIsRunning(true)

        streamControllerRef.current = streamChat(
          {
            message: content,
            experiment_id: eid,
            linked_datasets: ds.length > 0 ? ds : null,
            model_preference: opts?.user_model_preference ?? linkedModelId ?? null,
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
      linkedModelId,
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
    setLinkedDatasets([])
    setLinkedModelId(null)
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
          exp.chat_history.map((m: Record<string, unknown>) => ({
            id: (m.id as string) || uid("msg"),
            role: (m.role as ChatMessage["role"]) || "agent",
            content: (m.content as string) || "",
            timestamp: (m.timestamp as number) || Date.now(),
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
      const ts = exp.training_state as Partial<TrainingAgentState> | null | undefined
      const pref = ts?.user_model_preference
      setLinkedModelId(typeof pref === "string" && pref ? pref : null)
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
    isBackendReachabilityKnown,
    currentJobId,
    progress,
    confirmationRequest,
    experimentId,
    linkedDatasets,
    updateLinkedDatasets,
    linkedModelId,
    setLinkedModelId,
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
  }
}
