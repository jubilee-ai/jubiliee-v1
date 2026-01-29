/**
 * Hook for interacting with the real Training Agent via the FastAPI backend
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
  startTraining,
  getTrainingStatus,
  getDatasets,
  getModelTypes,
  streamTraining,
  streamResumeTraining,
  type Dataset,
  type ModelType,
  type StreamEvent,
} from "@/lib/api"
import { uid } from "@/lib/utils"

// Step definitions matching the agent graph
const STEP_DEFINITIONS = [
  { id: "select_model", name: "Model Selection", description: "Selecting the optimal model based on the goal" },
  { id: "data_collection", name: "Data Collection", description: "Gathering and linking datasets" },
  { id: "cleaning", name: "Cleaning & Standardization", description: "Cleaning and preparing data for training" },
  { id: "label_split_definition", name: "Label & Split Definition", description: "Defining target column and split strategy" },
  { id: "feature_selection_specification", name: "Feature Selection", description: "Selecting and specifying features to use" },
  { id: "feature_engineering_executor", name: "Feature Engineering", description: "Executing feature transformations" },
  { id: "training", name: "Training", description: "Training the model and evaluating metrics" },
  { id: "generate_report", name: "Generate Report", description: "Creating final report with results" },
]

function createInitialSteps(): StepInfo[] {
  return STEP_DEFINITIONS.map((def) => ({
    id: def.id,
    name: def.name,
    description: def.description,
    status: "pending",
  }))
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
  
  // Data
  datasets: Dataset[]
  modelTypes: ModelType[]
  
  // Actions
  startAgent: (goal: string, datasets?: string[], modelPreference?: string) => Promise<void>
  sendMessage: (content: string) => void
  handleConfirmation: (action: ConfirmationAction, comment?: string) => void
  reset: () => void
  checkConnection: () => Promise<boolean>
}

export function useRealAgent(): UseRealAgentReturn {
  const [agentState, setAgentState] = useState<TrainingAgentState>(createInitialState())
  const [steps, setSteps] = useState<StepInfo[]>(createInitialSteps())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isBackendConnected, setIsBackendConnected] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [modelTypes, setModelTypes] = useState<ModelType[]>([])
  const [threadId, setThreadId] = useState<string | null>(null)
  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null)
  const [acceptAllMode, setAcceptAllMode] = useState(false)
  
  // Use a ref to track accept-all mode to avoid stale closure issues in callbacks
  const acceptAllModeRef = useRef(acceptAllMode)
  acceptAllModeRef.current = acceptAllMode
  
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const streamControllerRef = useRef<AbortController | null>(null)
  const useStreaming = true // Enable streaming by default

  // Add a message to the chat
  const addMessage = useCallback((role: ChatMessage["role"], content: string) => {
    setMessages((prev) => [
      ...prev,
      { id: uid("msg"), role, content, timestamp: Date.now() },
    ])
  }, [])

  // Update steps based on current step from backend
  const updateStepsFromProgress = useCallback((currentStep: string | null, status: "running" | "completed") => {
    const stepOrder = STEP_DEFINITIONS.map((s) => s.id)
    const currentIndex = currentStep ? stepOrder.indexOf(currentStep) : -1
    
    setSteps((prev) =>
      prev.map((step, index) => {
        if (status === "completed") {
          return { ...step, status: "completed" }
        }
        if (index < currentIndex) {
          return { ...step, status: "completed" }
        }
        if (index === currentIndex) {
          return { ...step, status: "running" }
        }
        return { ...step, status: "pending" }
      })
    )
  }, [])

  // Check backend connection
  const checkConnection = useCallback(async () => {
    try {
      const connected = await checkHealth()
      setIsBackendConnected(connected)
      
      if (connected) {
        // Load datasets and models
        const [datasetsData, modelsData] = await Promise.all([
          getDatasets().catch(() => []),
          getModelTypes().catch(() => []),
        ])
        setDatasets(datasetsData)
        setModelTypes(modelsData)
      }
      
      return connected
    } catch {
      setIsBackendConnected(false)
      return false
    }
  }, [])

  // Check connection on mount
  useEffect(() => {
    checkConnection()
  }, [checkConnection])

  // Poll for job status
  const pollJob = useCallback(async (jobId: string) => {
    try {
      const status = await getTrainingStatus(jobId)
      setProgress(status.progress)
      
      if (status.current_step) {
        updateStepsFromProgress(status.current_step, status.status === "completed" ? "completed" : "running")
      }
      
      if (status.status === "completed" && status.state) {
        // Update agent state from result
        const resultState = status.state as unknown as TrainingAgentState
        setAgentState(resultState)
        setIsRunning(false)
        setCurrentJobId(null)
        
        // Mark all steps completed
        setSteps((prev) => prev.map((step) => ({ ...step, status: "completed" })))
        
        // Add completion message
        const metrics = resultState.training_metrics
        if (metrics?.success) {
          // Check if we have classification or regression metrics
          const hasClassificationMetrics = metrics.test_accuracy != null || metrics.test_roc_auc != null
          
          // Get regression metrics from iterations if not at top level
          const lastIteration = metrics.iterations?.[metrics.iterations.length - 1]
          const testR2 = metrics.test_r2 ?? lastIteration?.test_r2
          const testRmse = metrics.test_rmse ?? lastIteration?.test_rmse
          const testMae = metrics.test_mae ?? lastIteration?.test_mae
          const hasRegressionMetrics = testR2 != null || testRmse != null
          
          if (hasClassificationMetrics) {
            addMessage("agent", `Training completed successfully!\n\nModel: ${metrics.model_name}\nTest Accuracy: ${((metrics.test_accuracy as number) * 100).toFixed(1)}%\nTest ROC-AUC: ${(metrics.test_roc_auc as number)?.toFixed(3) || "N/A"}`)
          } else if (hasRegressionMetrics) {
            // Regression task with metrics
            const metricsStr = [
              testR2 != null ? `Test R²: ${testR2.toFixed(4)}` : null,
              testRmse != null ? `RMSE: ${testRmse.toFixed(2)}` : null,
              testMae != null ? `MAE: ${testMae.toFixed(2)}` : null,
            ].filter(Boolean).join("\n")
            
            addMessage("agent", `Training completed successfully!\n\nModel: ${metrics.model_name || resultState.model_weights_path}\nModel Type: ${metrics.model_type || resultState.selected_model}\n\n${metricsStr}\n\n${metrics.summary ? `Summary: ${metrics.summary.slice(0, 200)}...` : ""}`)
          } else {
            // Other task
            addMessage("agent", `Training completed successfully!\n\nModel: ${metrics.model_name || resultState.model_weights_path}\nModel Type: ${metrics.model_type || resultState.selected_model}\nIterations: ${metrics.num_iterations || 1}\n\nClick "View Report" for detailed results.`)
          }
        } else if (resultState.report_path) {
          addMessage("agent", `Training completed!\n\nModel: ${resultState.model_weights_path || "Unknown"}\nReport saved to: ${resultState.report_path}\n\nClick "View Report" for details.`)
        } else {
          addMessage("agent", "Training completed. Check the report for details.")
        }
        
        // Clear polling
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      } else if (status.status === "error") {
        setIsRunning(false)
        setCurrentJobId(null)
        addMessage("system", `Error: ${status.error || "Unknown error occurred"}`)
        
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }
    } catch (err) {
      console.error("Polling error:", err)
    }
  }, [addMessage, updateStepsFromProgress])

  // Format detailed stream event for display
  const formatStreamDetails = useCallback((event: StreamEvent): string => {
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
        lines.push(`Features created: ${summary.features_created.length}`)
        lines.push(`Names: ${summary.features_created.slice(0, 6).join(", ")}${summary.features_created.length > 6 ? "..." : ""}`)
      }
      if (summary.shapes && typeof summary.shapes === "object") {
        const shapes = summary.shapes as Record<string, number[]>
        if (shapes.train) lines.push(`Train shape: ${shapes.train[0]} × ${shapes.train[1]}`)
        if (shapes.val) lines.push(`Val shape: ${shapes.val[0]} × ${shapes.val[1]}`)
        if (shapes.test) lines.push(`Test shape: ${shapes.test[0]} × ${shapes.test[1]}`)
      }
      if (summary.validation_passed !== undefined) {
        lines.push(`Validation: ${summary.validation_passed ? "✓ Passed" : "⚠ Issues found"}`)
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

  // Handle streaming event
  const handleStreamEvent = useCallback((event: StreamEvent) => {
    console.log("[stream]", event)
    
    // Store thread_id from any event
    if (event.thread_id) {
      setThreadId(event.thread_id)
    }
    
    if (event.type === "started") {
      setProgress(0)
      addMessage("system", "Training stream started...")
      
      // Mark the first step as running
      setSteps((prev) =>
        prev.map((step, index) => {
          if (index === 0) {
            return { ...step, status: "running", startTime: Date.now() }
          }
          return step
        })
      )
    } else if (event.type === "interrupt") {
      // Human-in-the-loop interrupt - pause for user approval
      console.log("[stream] Interrupt received:", event)
      
      const nodeName = event.node || "unknown"
      const summary = typeof event.summary === "string" 
        ? event.summary 
        : JSON.stringify(event.summary || {}, null, 2)
      const stateSnapshot = (event.state_snapshot || {}) as Record<string, unknown>
      
      // Update step to awaiting confirmation with details
      setSteps((prev) =>
        prev.map((step) => {
          if (step.id === nodeName) {
            return { 
              ...step, 
              status: "awaiting_confirmation", 
              endTime: Date.now(),
              details: summary,
            }
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
        const currentThreadId = event.thread_id
        if (currentThreadId) {
          // Mark step as completed and continue
          setSteps((prev) =>
            prev.map((step) =>
              step.id === nodeName ? { ...step, status: "completed" } : step
            )
          )
          // Auto-resume with approval
          setTimeout(() => {
            streamControllerRef.current = streamResumeTraining(
              { thread_id: currentThreadId, approved: true },
              handleStreamEvent,
              (error) => {
                setIsRunning(false)
                addMessage("system", `Stream error: ${error.message}`)
              }
            )
          }, 100)
        }
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
      
    } else if (event.type === "dataset_loaded") {
      addMessage("system", `Dataset loaded: ${event.dataset} → ${event.ref}`)
    } else if (event.type === "node_complete") {
      const nodeName = event.node || "unknown"
      const nodeProgress = event.progress || 0
      
      setProgress(nodeProgress)
      
      // Update steps based on node:
      // - Previous steps: completed
      // - Current step: completed
      // - Next step: running (with spinner)
      // - Future steps: pending
      const stepOrder = STEP_DEFINITIONS.map((s) => s.id)
      const nodeIndex = stepOrder.indexOf(nodeName)
      
      setSteps((prev) =>
        prev.map((step, index) => {
          if (index < nodeIndex) {
            // Previous steps are completed
            return { ...step, status: "completed" }
          } else if (index === nodeIndex) {
            // Current step just completed
            return { ...step, status: "completed", endTime: Date.now() }
          } else if (index === nodeIndex + 1) {
            // Next step is now running (show spinner)
            return { ...step, status: "running", startTime: Date.now() }
          }
          // Future steps remain pending
          return step
        })
      )
      
      // Update agent state from event
      if (event.state) {
        setAgentState((prev) => ({
          ...prev,
          ...(event.state as Partial<TrainingAgentState>),
        }))
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
        // Fallback to simple message
        const stepDef = STEP_DEFINITIONS.find((s) => s.id === nodeName)
        addMessage("agent", `✓ ${stepDef?.name || nodeName} complete`)
      }
    } else if (event.type === "completed") {
      setProgress(100)
      setIsRunning(false)
      
      // Mark all steps completed
      setSteps((prev) => prev.map((step) => ({ ...step, status: "completed", endTime: step.endTime || Date.now() })))
      
      addMessage("agent", "**Training completed successfully!**\n\nClick 'View Report' to see detailed results including metrics, feature importance, and recommendations.")
    } else if (event.type === "error") {
      setIsRunning(false)
      
      // Mark current running step as error
      setSteps((prev) => prev.map((step) => 
        step.status === "running" ? { ...step, status: "error" } : step
      ))
      
      addMessage("system", `Error: ${event.error || "Unknown error"}`)
    }
  }, [addMessage, formatStreamDetails])

  // Start training with streaming
  const startAgentStreaming = useCallback(async (goal: string, linkedDatasets?: string[], modelPreference?: string) => {
    // Check connection first
    const connected = await checkConnection()
    if (!connected) {
      addMessage("system", "Backend not connected. Please start the FastAPI server with: uvicorn app:app --reload")
      return
    }
    
    setIsRunning(true)
    setProgress(0)
    setSteps(createInitialSteps())
    
    // Update initial state
    const initialState = createInitialState()
    initialState.goal = goal
    initialState.linked_datasets = linkedDatasets || null
    initialState.user_model_preference = modelPreference || null
    setAgentState(initialState)
    
    addMessage("agent", `Starting training with goal: "${goal}"\n\nStreaming progress updates in real-time...`)
    
    // Start streaming
    streamControllerRef.current = streamTraining(
      {
        goal,
        linked_datasets: linkedDatasets,
        user_model_preference: modelPreference,
      },
      handleStreamEvent,
      (error) => {
        setIsRunning(false)
        addMessage("system", `Stream error: ${error.message}`)
      }
    )
  }, [addMessage, checkConnection, handleStreamEvent])

  // Handle confirmation actions (human-in-the-loop)
  const handleConfirmation = useCallback((action: ConfirmationAction, comment?: string) => {
    if (!confirmationRequest || !threadId) {
      console.warn("[handleConfirmation] No confirmation request or thread_id")
      return
    }
    
    const currentStep = confirmationRequest.step
    const stepDef = STEP_DEFINITIONS.find((s) => s.id === currentStep)
    
    switch (action) {
      case "accept":
        addMessage("system", "✓ Step accepted")
        
        // Mark step as completed
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "completed" } : step
          )
        )
        
        // Clear confirmation and resume
        setConfirmationRequest(null)
        setIsRunning(true)
        
        // Resume streaming with approval
        streamControllerRef.current = streamResumeTraining(
          { thread_id: threadId, approved: true },
          handleStreamEvent,
          (error) => {
            setIsRunning(false)
            addMessage("system", `Stream error: ${error.message}`)
          }
        )
        break
        
      case "accept_all":
        addMessage("system", "Auto-accepting remaining steps")
        setAcceptAllMode(true)
        acceptAllModeRef.current = true // Set ref immediately to avoid stale closure
        
        // Mark step as completed
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "completed" } : step
          )
        )
        
        // Clear confirmation and resume
        setConfirmationRequest(null)
        setIsRunning(true)
        
        // Resume streaming with approval
        streamControllerRef.current = streamResumeTraining(
          { thread_id: threadId, approved: true },
          handleStreamEvent,
          (error) => {
            setIsRunning(false)
            addMessage("system", `Stream error: ${error.message}`)
          }
        )
        break
        
      case "redo":
        addMessage("system", `↻ Requested redo${comment ? `: ${comment}` : ""}`)
        addMessage("agent", `Got it. I'll redo **${stepDef?.name || currentStep}**${comment ? ` with your feedback: "${comment}"` : ""}.`)
        
        // Mark step as running (will be redone)
        setSteps((prev) =>
          prev.map((step) =>
            step.id === currentStep ? { ...step, status: "running", startTime: Date.now() } : step
          )
        )
        
        // Clear confirmation and resume with feedback
        setConfirmationRequest(null)
        setIsRunning(true)
        
        // Resume streaming with rejection and feedback
        streamControllerRef.current = streamResumeTraining(
          { thread_id: threadId, approved: false, feedback: comment || "Please redo this step." },
          handleStreamEvent,
          (error) => {
            setIsRunning(false)
            addMessage("system", `Stream error: ${error.message}`)
          }
        )
        break
    }
  }, [confirmationRequest, threadId, addMessage, handleStreamEvent])

  // Start training with polling (fallback)
  const startAgentPolling = useCallback(async (goal: string, linkedDatasets?: string[], modelPreference?: string) => {
    // Check connection first
    const connected = await checkConnection()
    if (!connected) {
      addMessage("system", "Backend not connected. Please start the FastAPI server with: uvicorn app:app --reload")
      return
    }
    
    setIsRunning(true)
    setProgress(0)
    setSteps(createInitialSteps())
    
    // Update initial state
    const initialState = createInitialState()
    initialState.goal = goal
    initialState.linked_datasets = linkedDatasets || null
    initialState.user_model_preference = modelPreference || null
    setAgentState(initialState)
    
    addMessage("agent", `Starting training with goal: "${goal}"\n\nThis will run the full pipeline. Progress will update as steps complete.`)
    
    try {
      // Start training job
      const response = await startTraining({
        goal,
        linked_datasets: linkedDatasets,
        user_model_preference: modelPreference,
      })
      
      setCurrentJobId(response.job_id)
      addMessage("system", `Training job started (ID: ${response.job_id.slice(0, 8)}...)`)
      
      // Start polling for status
      pollingRef.current = setInterval(() => {
        pollJob(response.job_id)
      }, 1000)
      
    } catch (err) {
      setIsRunning(false)
      const errorMsg = err instanceof Error ? err.message : "Unknown error"
      addMessage("system", `Failed to start training: ${errorMsg}`)
    }
  }, [addMessage, checkConnection, pollJob])

  // Start training (uses streaming by default)
  const startAgent = useCallback(async (goal: string, linkedDatasets?: string[], modelPreference?: string) => {
    if (useStreaming) {
      await startAgentStreaming(goal, linkedDatasets, modelPreference)
    } else {
      await startAgentPolling(goal, linkedDatasets, modelPreference)
    }
  }, [useStreaming, startAgentStreaming, startAgentPolling])

  // Send a chat message
  const sendMessage = useCallback((content: string) => {
    addMessage("user", content)
    
    const lowerContent = content.toLowerCase()
    
    if (lowerContent.includes("train") || lowerContent.includes("start")) {
      if (!isRunning) {
        const goal = content.replace(/train|start|model|please/gi, "").trim() || "Build a model"
        startAgent(goal)
      } else {
        addMessage("agent", "Training is already in progress. Please wait for it to complete.")
      }
    } else if (lowerContent.includes("status")) {
      if (currentJobId) {
        addMessage("agent", `Training in progress... ${progress}% complete`)
      } else {
        addMessage("agent", "No training job running. Type 'train [goal]' to start.")
      }
    } else if (lowerContent.includes("help")) {
      addMessage("agent", `Available commands:
- Type your training goal to start (e.g., "Train a loan default prediction model")
- "status" - Check current training status
- "reset" - Start over

Note: The real agent runs the full pipeline at once. Progress updates show which step is running.`)
    } else if (lowerContent.includes("reset")) {
      reset()
      addMessage("agent", "Reset complete. Tell me your training goal to start again.")
    } else {
      addMessage("agent", `I understand you said: "${content}". If you want to start training, just describe your goal. Type "help" for available commands.`)
    }
  }, [addMessage, isRunning, currentJobId, progress, startAgent])

  // Reset everything
  const reset = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
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
    setThreadId(null)
    setConfirmationRequest(null)
    setAcceptAllMode(false)
    acceptAllModeRef.current = false
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
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
    datasets,
    modelTypes,
    startAgent,
    sendMessage,
    handleConfirmation,
    reset,
    checkConnection,
  }
}
