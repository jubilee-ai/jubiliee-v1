/**
 * Hook for interacting with the real Training Agent via the FastAPI backend
 */

import { useState, useCallback, useRef, useEffect } from "react"
import type {
  TrainingAgentState,
  StepInfo,
  ChatMessage,
} from "@/types/agent"
import {
  checkHealth,
  startTraining,
  getTrainingStatus,
  getDatasets,
  getModelTypes,
  streamTraining,
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
  { id: "human_confirmation", name: "Human Confirmation", description: "Review and confirm before training" },
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
  
  // Data
  datasets: Dataset[]
  modelTypes: ModelType[]
  
  // Actions
  startAgent: (goal: string, datasets?: string[], modelPreference?: string) => Promise<void>
  sendMessage: (content: string) => void
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
    
    if (!details && !summary) return ""
    
    const lines: string[] = []
    
    // Use details.title and description if available
    if (details?.title) {
      lines.push(`**${details.title}**`)
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
      if (summary.num_features) lines.push(`Features specified: ${summary.num_features}`)
      if (summary.feature_names && Array.isArray(summary.feature_names)) {
        lines.push(`Features: ${summary.feature_names.slice(0, 6).join(", ")}${summary.feature_names.length > 6 ? "..." : ""}`)
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
    } else if (nodeName === "human_confirmation" && summary) {
      lines.push(summary.confirmed ? "✓ Confirmed - proceeding to training" : "Awaiting confirmation...")
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
      const formattedDetails = formatStreamDetails(event)
      if (formattedDetails) {
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
    datasets,
    modelTypes,
    startAgent,
    sendMessage,
    reset,
    checkConnection,
  }
}
