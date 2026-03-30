import { useState, useCallback, useRef } from "react"
import type {
  TrainingAgentState,
  StepInfo,
  ChatMessage,
  ConfirmationAction,
  ConfirmationRequest,
} from "@/types/agent"
import {
  createInitialState,
  createInitialSteps,
  executeStep,
  getNextStep,
  getConfirmationRequest,
  STEP_DEFINITIONS,
} from "@/lib/mockAgent"
import { uid } from "@/lib/utils"

export interface UseAgentStateReturn {
  // State
  agentState: TrainingAgentState
  steps: StepInfo[]
  messages: ChatMessage[]
  isRunning: boolean
  currentStepId: string | null
  confirmationRequest: ConfirmationRequest | null
  acceptAllMode: boolean
  
  // Actions
  startAgent: (goal: string, datasets?: string[], modelPreference?: string) => void
  handleConfirmation: (action: ConfirmationAction, comment?: string) => void
  sendMessage: (content: string) => void
  reset: () => void
}

export function useAgentState(): UseAgentStateReturn {
  const [agentState, setAgentState] = useState<TrainingAgentState>(createInitialState())
  const [steps, setSteps] = useState<StepInfo[]>(createInitialSteps())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [currentStepId, setCurrentStepId] = useState<string | null>(null)
  const [confirmationRequest, setConfirmationRequest] = useState<ConfirmationRequest | null>(null)
  const [acceptAllMode, setAcceptAllMode] = useState(false)
  
  const stateRef = useRef(agentState)
  stateRef.current = agentState
  
  // Use a ref to track accept-all mode to avoid stale closure issues
  const acceptAllModeRef = useRef(acceptAllMode)
  acceptAllModeRef.current = acceptAllMode

  // Add a message to the chat
  const addMessage = useCallback((role: ChatMessage["role"], content: string, links?: ChatMessage["links"]) => {
    setMessages((prev) => [
      ...prev,
      {
        id: uid("msg"),
        role,
        content,
        timestamp: Date.now(),
        links,
      },
    ])
  }, [])

  // Update a step's status and metrics
  const updateStep = useCallback((stepId: string, updates: Partial<StepInfo>) => {
    setSteps((prev) =>
      prev.map((step) =>
        step.id === stepId ? { ...step, ...updates } : step
      )
    )
  }, [])

  // Execute a single step
  const runStep = useCallback(async (stepId: string): Promise<boolean> => {
    const currentState = stateRef.current
    
    // Mark step as running
    setCurrentStepId(stepId)
    updateStep(stepId, { status: "running", startTime: Date.now() })
    
    const stepDef = STEP_DEFINITIONS.find((s) => s.id === stepId)
    addMessage("system", `Starting: ${stepDef?.name || stepId}...`)
    
    try {
      // Execute the step
      const result = await executeStep(stepId, currentState)
      
      // Update agent state
      const { stepMetrics, stepDetails, confirmationRequired, ...stateUpdates } = result
      setAgentState((prev) => ({ ...prev, ...stateUpdates }))
      stateRef.current = { ...stateRef.current, ...stateUpdates }
      
      // Update step with results
      updateStep(stepId, {
        status: confirmationRequired ? "awaiting_confirmation" : "completed",
        metrics: stepMetrics,
        details: stepDetails,
        endTime: Date.now(),
      })
      
      // Add agent message about completion
      addMessage("agent", `Completed ${stepDef?.name || stepId}.\n\n${stepDetails || ""}`)
      
      // If confirmation required and not in accept-all mode, wait for user
      if (confirmationRequired && !acceptAllModeRef.current) {
        setConfirmationRequest(getConfirmationRequest(stepId, stateRef.current, stepMetrics, stepDetails))
        setIsRunning(false)
        return false // Stop execution, waiting for confirmation
      }
      
      // Mark as completed if we're continuing
      if (confirmationRequired) {
        updateStep(stepId, { status: "completed" })
      }
      
      return true // Continue to next step
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error"
      updateStep(stepId, { status: "error", details: errorMessage, endTime: Date.now() })
      addMessage("system", `Error in ${stepDef?.name || stepId}: ${errorMessage}`)
      setAgentState((prev) => ({ ...prev, error: errorMessage }))
      setIsRunning(false)
      return false
    }
  }, [updateStep, addMessage])

  // Run all steps from current position
  const runFromStep = useCallback(async (startStepId: string) => {
    setIsRunning(true)
    let currentStep: string | null = startStepId
    
    while (currentStep) {
      const shouldContinue = await runStep(currentStep)
      if (!shouldContinue) {
        return // Paused for confirmation or error
      }
      currentStep = getNextStep(currentStep)
    }
    
    // All steps completed
    setIsRunning(false)
    setCurrentStepId(null)
    addMessage("agent", "Training pipeline completed! Check the final report for detailed results.")
  }, [runStep, addMessage])

  // Start the agent with initial parameters
  const startAgent = useCallback((goal: string, datasets?: string[], modelPreference?: string) => {
    // Reset state
    const initialState = createInitialState()
    initialState.goal = goal
    initialState.linked_datasets = datasets || null
    initialState.user_model_preference = modelPreference || null
    initialState.current_step = "select_model"
    
    setAgentState(initialState)
    stateRef.current = initialState
    setSteps(createInitialSteps())
    setConfirmationRequest(null)
    setAcceptAllMode(false)
    
    // Add welcome message
    addMessage("agent", `Starting training agent with goal: "${goal}"\n\nI'll guide you through each step and ask for your confirmation before proceeding.`)
    
    // Start running
    runFromStep("select_model")
  }, [addMessage, runFromStep])

  // Handle confirmation actions
  const handleConfirmation = useCallback((action: ConfirmationAction, comment?: string) => {
    if (!confirmationRequest) return
    
    const currentStep = confirmationRequest.step
    
    switch (action) {
      case "accept":
        addMessage("system", "Step accepted")
        updateStep(currentStep, { status: "completed" })
        setConfirmationRequest(null)
        const nextStep = getNextStep(currentStep)
        if (nextStep) {
          runFromStep(nextStep)
        } else {
          setIsRunning(false)
        }
        break
        
      case "accept_all":
        addMessage("system", "Auto-accepting remaining steps")
        setAcceptAllMode(true)
        acceptAllModeRef.current = true // Set ref immediately to avoid stale closure
        updateStep(currentStep, { status: "completed" })
        setConfirmationRequest(null)
        const nextStepAll = getNextStep(currentStep)
        if (nextStepAll) {
          runFromStep(nextStepAll)
        } else {
          setIsRunning(false)
        }
        break
        
      case "redo":
        addMessage("system", `↻ Requested redo"${comment ? `: ${comment}` : ""}"`)
        addMessage("agent", `Got it. I'll redo this step${comment ? ` with your feedback: "${comment}"` : ""}.`)
        updateStep(currentStep, { status: "pending" })
        setConfirmationRequest(null)
        // Re-run the same step
        runFromStep(currentStep)
        break
    }
  }, [confirmationRequest, addMessage, updateStep, runFromStep])

  // Send a chat message
  const sendMessage = useCallback((content: string) => {
    addMessage("user", content)
    
    // Simple parsing for quick actions
    const lowerContent = content.toLowerCase()
    
    if (lowerContent.includes("train") || lowerContent.includes("start")) {
      // Extract goal from message
      const goal = content.replace(/train|start|model|please/gi, "").trim() || "Build a model"
      if (!isRunning && !confirmationRequest) {
        startAgent(goal)
      } else {
        addMessage("agent", "I'm currently running or waiting for your confirmation. Please respond to the current step first.")
      }
    } else if (lowerContent.includes("accept")) {
      if (confirmationRequest) {
        handleConfirmation(lowerContent.includes("all") ? "accept_all" : "accept")
      }
    } else if (lowerContent.includes("redo")) {
      if (confirmationRequest) {
        handleConfirmation("redo", content)
      }
    } else if (lowerContent.includes("help")) {
      addMessage("agent", `Available commands:
- Type your training goal to start (e.g., "Train a loan default prediction model")
- "accept" - Accept the current step
- "accept all" - Accept all remaining steps
- "redo [feedback]" - Redo the current step with your feedback
- "reset" - Start over`)
    } else if (lowerContent.includes("reset")) {
      reset()
      addMessage("agent", "Reset complete. Tell me your training goal to start again.")
    } else {
      addMessage("agent", "I'll take that into consideration. You can say 'accept', 'redo', or ask me about anything related to the current step.")
    }
  }, [addMessage, isRunning, confirmationRequest, startAgent, handleConfirmation])

  // Reset everything
  const reset = useCallback(() => {
    setAgentState(createInitialState())
    setSteps(createInitialSteps())
    setMessages([])
    setIsRunning(false)
    setCurrentStepId(null)
    setConfirmationRequest(null)
    setAcceptAllMode(false)
  }, [])

  return {
    agentState,
    steps,
    messages,
    isRunning,
    currentStepId,
    confirmationRequest,
    acceptAllMode,
    startAgent,
    handleConfirmation,
    sendMessage,
    reset,
  }
}
