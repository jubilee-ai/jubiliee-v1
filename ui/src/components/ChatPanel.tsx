import React, { useState, useRef, useEffect, useCallback, useMemo, useImperativeHandle, forwardRef } from "react"
import type {
  ChatMessage,
  ConfirmationRequest,
  ConfirmationAction,
  Dataset,
  TrainingAgentState,
  StepInfo,
  TaskPlanSummary,
} from "@/types/agent"
import type { Dataset as ApiDataset } from "@/lib/api"
import { AVAILABLE_DATASETS } from "@/lib/mockAgent"
import { StepDetailModal } from "@/components/StepDetailModal"
import { Settings2, Sparkles, BarChart3, Zap, FileText, Moon, Loader2 } from "lucide-react"

import {
  MessageBubble,
  EmptyState,
  LoadingIndicator,
  ConfirmationPanel,
  PastConfirmation,
  ChatInput,
  detectStepFromMessage,
  STEP_KEYWORDS,
} from "./chat"
import { PredictionResultCard } from "./chat/PredictionResultCard"
import type { ResolvedConfirmation } from "./chat"
import { BackgroundTaskDialog } from "./chat/BackgroundTaskDialog"
import type { BackgroundTaskPayload } from "./chat/BackgroundTaskDialog"
import { TaskStatusView } from "@/components/TaskStatusView"
import { TRACE_EXCLUDE_IDS } from "@/lib/trainingSteps"

const STEP_TO_PHASE: Record<string, string> = {
  data_collection: "Setup",
  cleaning: "Preparation",
  label_split_definition: "Preparation",
  feature_selection_specification: "Features",
  feature_specification_and_engineering: "Features",
  feature_engineering_executor: "Features",
  feature_experiment_runner: "Features",
  training_approval: "Training",
  training: "Training",
  generate_report: "Output",
}

interface ChatPanelProps {
  messages: ChatMessage[]
  confirmationRequest: ConfirmationRequest | null
  isRunning: boolean
  /** Assign-task flow: orchestrator + clarifications only until user starts a run from the plan card */
  backgroundIntakeActive?: boolean
  onSendMessage: (
    content: string,
    opts?: { user_model_preference?: string; linked_datasets_override?: string[] },
  ) => void
  onConfirmation: (action: ConfirmationAction, comment?: string) => void
  linkedDatasets: string[]
  onLinkedDatasetsChange: (ids: string[]) => void
  datasets?: ApiDataset[]
  highlightedMessageId?: string | null
  onClearHighlight?: () => void
  onViewReport?: () => void
  /** Opens the Models tab and highlights this experiment’s trained model (next to View Report in chat). */
  onViewModelInRegistry?: () => void
  agentState?: TrainingAgentState
  steps?: StepInfo[]
  experimentId?: string | null
  /** Current pipeline activity label (shown beside the loading wave) */
  runningStepHint?: string | null
  /** Hide chat composer (background task mode). */
  hideComposer?: boolean
  /** Bridge UI while hands-off run is being registered after plan approval. */
  startingHandsOffTask?: boolean
  onApproveTrainingPlan?: (messageId: string, plan: TaskPlanSummary, refs: string[]) => void
  /** Opens from empty state + composer; submits a planning message geared toward background run */
  onSubmitBackgroundTask?: (payload: BackgroundTaskPayload) => void
}

export interface ChatPanelRef {
  scrollToMessage: (messageId: string) => void
  findMessageByStepName: (stepName: string) => string | null
}

export const ChatPanel = forwardRef<ChatPanelRef, ChatPanelProps>(function ChatPanel({
  messages,
  confirmationRequest,
  isRunning,
  backgroundIntakeActive = false,
  onSendMessage,
  onConfirmation,
  linkedDatasets,
  onLinkedDatasetsChange,
  datasets: propDatasets,
  highlightedMessageId,
  onClearHighlight,
  onViewReport,
  onViewModelInRegistry,
  agentState,
  steps,
  experimentId,
  runningStepHint,
  hideComposer,
  startingHandsOffTask = false,
  onApproveTrainingPlan,
  onSubmitBackgroundTask,
}, ref) {
  const availableDatasets = propDatasets && propDatasets.length > 0 
    ? propDatasets 
    : AVAILABLE_DATASETS

  const [draft, setDraft] = useState("")
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null)
  const [backgroundDialogOpen, setBackgroundDialogOpen] = useState(false)
  
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const [pastConfirmations, setPastConfirmations] = useState<ResolvedConfirmation[]>([])
  const confirmationShownAfterMessageIdRef = useRef<string | null>(null)
  const prevConfirmationStepRef = useRef<string | null>(null)

  useEffect(() => {
    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    })
  }, [messages.length])

  useEffect(() => {
    if (confirmationRequest && confirmationRequest.step !== prevConfirmationStepRef.current) {
      const lastMessage = messages[messages.length - 1]
      confirmationShownAfterMessageIdRef.current = lastMessage?.id || null
      prevConfirmationStepRef.current = confirmationRequest.step
    }
    if (!confirmationRequest) {
      prevConfirmationStepRef.current = null
    }
  }, [confirmationRequest, messages])

  useEffect(() => {
    if (messages.length === 0) {
      setPastConfirmations([])
      confirmationShownAfterMessageIdRef.current = null
      prevConfirmationStepRef.current = null
    }
  }, [messages.length])

  const confirmationsMap = useMemo(() => {
    const map = new Map<string, ResolvedConfirmation[]>()
    for (const conf of pastConfirmations) {
      if (conf.afterMessageId) {
        const existing = map.get(conf.afterMessageId) || []
        existing.push(conf)
        map.set(conf.afterMessageId, existing)
      }
    }
    return map
  }, [pastConfirmations])

  const handleBackgroundDialogSubmit = useCallback(
    (payload: BackgroundTaskPayload) => {
      onSubmitBackgroundTask?.(payload)
      setBackgroundDialogOpen(false)
    },
    [onSubmitBackgroundTask],
  )

  const handleConfirmationWithTracking = useCallback((action: ConfirmationAction, comment?: string) => {
    if (confirmationRequest) {
      const skipPastCard =
        confirmationRequest.step === "data_collection" &&
        (action === "accept" || action === "accept_all")
      if (!skipPastCard) {
        setPastConfirmations((prev) => [
          ...prev,
          {
            ...confirmationRequest,
            resolvedAction: action,
            afterMessageId: confirmationShownAfterMessageIdRef.current || undefined,
            redoComment: action === "redo" ? comment : undefined,
          },
        ])
      }
    }
    onConfirmation(action, comment)
  }, [onConfirmation, confirmationRequest])

  useImperativeHandle(ref, () => ({
    scrollToMessage: (messageId: string) => {
      const el = messageRefs.current.get(messageId)
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" })
      }
    },
    findMessageByStepName: (stepName: string) => {
      const keywords = STEP_KEYWORDS[stepName] || [stepName]
      
      for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i]
        if (msg.role === "agent") {
          const content = msg.content.toLowerCase()
          if (keywords.some(kw => content.includes(kw.toLowerCase()))) {
            return msg.id
          }
        }
      }
      return null
    }
  }), [messages])

  const handleSend = useCallback(() => {
    const text = draft.trim()
    if (!text) return

    onSendMessage(text)
    setDraft("")
  }, [draft, onSendMessage])

  const handleDatasetSelect = useCallback((dataset: Dataset) => {
    const key = dataset.name || dataset.file || dataset.id
    if (!key) {
      console.warn("Dataset has no usable identifier, skipping")
      return
    }
    if (!linkedDatasets.includes(key)) {
      onLinkedDatasetsChange([...linkedDatasets, key])
    }
  }, [linkedDatasets, onLinkedDatasetsChange])

  const handleDatasetRemove = useCallback((datasetFile: string) => {
    onLinkedDatasetsChange(linkedDatasets.filter((d) => d !== datasetFile))
  }, [linkedDatasets, onLinkedDatasetsChange])

  const handleContainerClick = useCallback(() => {
    if (highlightedMessageId && onClearHighlight) {
      onClearHighlight()
    }
  }, [highlightedMessageId, onClearHighlight])

  const setMessageRef = useCallback((id: string) => (el: HTMLDivElement | null) => {
    if (el) messageRefs.current.set(id, el)
    else messageRefs.current.delete(id)
  }, [])

  const resolveAgentStep = useCallback((msg: ChatMessage) => {
    if (msg.role !== "agent") return null
    const id = msg.stepId ?? detectStepFromMessage(msg.content)
    if (id && TRACE_EXCLUDE_IDS.has(id)) return null
    return id
  }, [])

  const openStepDetails = useCallback((stepId: string) => {
    if (TRACE_EXCLUDE_IDS.has(stepId)) return
    setSelectedStepId(stepId)
  }, [])

  const phaseMarkers = useMemo(() => {
    const markers = new Map<string, string>()
    let lastPhase: string | null = null
    for (const msg of messages) {
      if (msg.role === "agent") {
        const step = msg.stepId ?? detectStepFromMessage(msg.content)
        const phase =
          step && !TRACE_EXCLUDE_IDS.has(step) ? STEP_TO_PHASE[step] : null
        if (phase && phase !== lastPhase) {
          markers.set(msg.id, phase)
          lastPhase = phase
        }
      }
    }
    return markers
  }, [messages])

  const isTaskLab = Boolean(agentState?.lab_mode === "task" && experimentId)

  if (startingHandsOffTask && !isTaskLab) {
    return (
      <div
        className="flex flex-1 min-h-0 flex-col overflow-hidden dot-grid"
        onClick={handleContainerClick}
      >
        <div className="flex-1 min-h-0 overflow-y-auto flex flex-col">
          <div className="max-w-4xl mx-auto px-6 sm:px-10 py-10 flex-1 flex flex-col items-center justify-center gap-4 min-h-[min(420px,70vh)]">
            <span className="text-overline font-bold text-muted-foreground/60 tracking-[0.2em] uppercase">
              Background task
            </span>
            <Loader2 className="h-9 w-9 animate-spin text-primary" aria-hidden />
            <p className="text-sm font-medium text-foreground text-center">Starting your run…</p>
            <p className="text-xs text-muted-foreground text-center max-w-sm leading-relaxed">
              Connecting to your experiment and loading the task view. This usually takes a moment.
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (isTaskLab && agentState) {
    return (
      <div
        className="flex flex-1 min-h-0 flex-col overflow-hidden dot-grid"
        onClick={handleContainerClick}
      >
        <TaskStatusView
          agentState={agentState}
          onViewReport={onViewReport ?? (() => {})}
          datasets={availableDatasets}
        />
      </div>
    )
  }

  return (
    <div
      className="flex flex-1 min-h-0 flex-col overflow-hidden dot-grid"
      onClick={handleContainerClick}
    >
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain"
      >
        <div className="space-y-5 max-w-4xl mx-auto px-6 py-8">
          {backgroundIntakeActive && (
            <div className="rounded-xl border border-primary/20 bg-primary/[0.06] px-4 py-3 text-left shadow-sm">
              <div className="flex items-start gap-3">
                <div className="rounded-lg bg-primary/15 p-2 shrink-0">
                  <Moon className="h-4 w-4 text-primary" />
                </div>
                <div className="min-w-0 space-y-1">
                  <p className="text-sm font-semibold text-foreground">Background task</p>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Jubilee may ask a brief follow-up here. Otherwise you only need to{" "}
                    <span className="text-foreground/90 font-medium">Run on my behalf</span> on the plan card — the full
                    pipeline then runs away from this chat.
                  </p>
                </div>
              </div>
            </div>
          )}
          {messages.length === 0 ? (
            <EmptyState
              onAssignTask={onSubmitBackgroundTask ? () => setBackgroundDialogOpen(true) : undefined}
            />
          ) : null}

          {messages.map((msg) => {
            const detectedStep = resolveAgentStep(msg)
            const stepInfo = detectedStep && steps ? steps.find(s => s.id === detectedStep) : null
            const isClickable = !!detectedStep && !!agentState && !!steps && stepInfo?.status === "completed"
            const confsAfterThis = confirmationsMap.get(msg.id) || []
            const phaseMarker = phaseMarkers.get(msg.id)
            
            return (
              <React.Fragment key={msg.id}>
                {phaseMarker && <PhaseMarker phase={phaseMarker} />}
                {msg.prediction ? (
                  <PredictionResultCard
                    result={msg.prediction}
                    onEvaluate={(model) => onSendMessage(`Evaluate model ${model}`)}
                  />
                ) : (
                  <MessageBubble
                    message={msg}
                    isHighlighted={highlightedMessageId === msg.id}
                    onViewReport={onViewReport}
                    onViewModelInRegistry={onViewModelInRegistry}
                    stepId={detectedStep}
                    isClickable={isClickable}
                    onStepClick={isClickable ? () => openStepDetails(detectedStep) : undefined}
                    isRunning={isRunning}
                    onApproveTrainingPlan={onApproveTrainingPlan}
                    datasets={availableDatasets}
                    agentState={agentState}
                    ref={setMessageRef(msg.id)}
                  />
                )}
                
                {confsAfterThis.map((conf, index) => (
                  <PastConfirmation
                    key={`past-conf-${msg.id}-${index}`}
                    confirmation={conf}
                    agentState={agentState}
                    steps={steps}
                    onViewDetails={openStepDetails}
                  />
                ))}
              </React.Fragment>
            )
          })}

          {isRunning && !confirmationRequest && (
            <LoadingIndicator label={runningStepHint} />
          )}

          {confirmationRequest && (
            <ConfirmationPanel
              confirmationRequest={confirmationRequest}
              onConfirmation={handleConfirmationWithTracking}
              agentState={agentState}
              steps={steps}
              onViewDetails={openStepDetails}
            />
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {!hideComposer && (
        <ChatInput
          draft={draft}
          onDraftChange={setDraft}
          onSend={handleSend}
          isDisabled={isRunning && !confirmationRequest}
          placeholder={
            backgroundIntakeActive
              ? "Answer only if Jubilee asks something…"
              : messages.length === 0
                ? "Message Jubilee…"
                : "Send a message…"
          }
          selectedDatasets={linkedDatasets}
          onDatasetSelect={handleDatasetSelect}
          onDatasetRemove={handleDatasetRemove}
          availableDatasets={availableDatasets}
          experimentId={experimentId}
          onAssignBackgroundTask={
            onSubmitBackgroundTask ? () => setBackgroundDialogOpen(true) : undefined
          }
        />
      )}

      {onSubmitBackgroundTask && (
        <BackgroundTaskDialog
          open={backgroundDialogOpen}
          onOpenChange={setBackgroundDialogOpen}
          datasets={availableDatasets}
          isSubmitting={isRunning}
          onSubmit={handleBackgroundDialogSubmit}
        />
      )}

      {selectedStepId && agentState && steps && (
        <StepDetailModal
          stepId={selectedStepId}
          agentState={agentState}
          steps={steps}
          onClose={() => setSelectedStepId(null)}
        />
      )}
    </div>
  )
})

const PHASE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  Setup: Settings2,
  Preparation: Sparkles,
  Features: BarChart3,
  Training: Zap,
  Output: FileText,
}

function PhaseMarker({ phase }: { phase: string }) {
  const Icon = PHASE_ICONS[phase]
  return (
    <div className="flex items-center gap-3 py-2 animate-phase-in">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border/40 to-transparent" />
      <div className="flex items-center gap-1.5 text-overline uppercase tracking-[0.15em] text-muted-foreground/50 font-semibold select-none">
        {Icon && <Icon className="h-3 w-3 text-primary/40 animate-spin-in" />}
        <span>{phase}</span>
      </div>
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border/40 to-transparent" />
    </div>
  )
}
