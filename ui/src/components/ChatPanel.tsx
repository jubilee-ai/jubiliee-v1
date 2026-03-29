import React, { useState, useRef, useEffect, useCallback, useMemo, useImperativeHandle, forwardRef } from "react"
import type { ChatMessage, ConfirmationRequest, ConfirmationAction, Dataset, TrainingAgentState, StepInfo } from "@/types/agent"
import type { Dataset as ApiDataset, ModelType } from "@/lib/api"
import { AVAILABLE_DATASETS, AVAILABLE_MODELS } from "@/lib/mockAgent"
import { StepDetailModal } from "@/components/StepDetailModal"
import { Settings2, Sparkles, BarChart3, Zap, FileText } from "lucide-react"

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

const STEP_TO_PHASE: Record<string, string> = {
  select_model: "Setup",
  data_collection: "Setup",
  cleaning: "Preparation",
  label_split_definition: "Preparation",
  feature_selection_specification: "Features",
  feature_specification_and_engineering: "Features",
  feature_engineering_executor: "Features",
  training_approval: "Training",
  training: "Training",
  generate_report: "Output",
}

interface ChatPanelProps {
  messages: ChatMessage[]
  confirmationRequest: ConfirmationRequest | null
  isRunning: boolean
  onSendMessage: (content: string, opts?: { user_model_preference?: string }) => void
  onConfirmation: (action: ConfirmationAction, comment?: string) => void
  linkedDatasets: string[]
  onLinkedDatasetsChange: (ids: string[]) => void
  linkedModelId: string | null
  onLinkedModelChange: (id: string | null) => void
  datasets?: ApiDataset[]
  modelTypes?: ModelType[]
  highlightedMessageId?: string | null
  onClearHighlight?: () => void
  onViewReport?: () => void
  agentState?: TrainingAgentState
  steps?: StepInfo[]
  hasExperimentChecklist?: boolean
  experimentId?: string | null
  /** Current pipeline activity label (shown beside the loading wave) */
  runningStepHint?: string | null
}

export interface ChatPanelRef {
  scrollToMessage: (messageId: string) => void
  findMessageByStepName: (stepName: string) => string | null
}

export const ChatPanel = forwardRef<ChatPanelRef, ChatPanelProps>(function ChatPanel({
  messages,
  confirmationRequest,
  isRunning,
  onSendMessage,
  onConfirmation,
  linkedDatasets,
  onLinkedDatasetsChange,
  linkedModelId,
  onLinkedModelChange,
  datasets: propDatasets,
  modelTypes: propModelTypes,
  highlightedMessageId,
  onClearHighlight,
  onViewReport,
  agentState,
  steps,
  hasExperimentChecklist,
  experimentId,
  runningStepHint,
}, ref) {
  const availableDatasets = propDatasets && propDatasets.length > 0 
    ? propDatasets 
    : AVAILABLE_DATASETS
  const availableModels = propModelTypes && propModelTypes.length > 0
    ? propModelTypes
    : AVAILABLE_MODELS

  const [draft, setDraft] = useState("")
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null)
  const [useHitl, setUseHitl] = useState(true)
  
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

  const getConfirmationsAfterMessage = useCallback(() => {
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

  const handleConfirmationWithTracking = useCallback((action: ConfirmationAction, comment?: string) => {
    if (confirmationRequest) {
      setPastConfirmations(prev => [...prev, { 
        ...confirmationRequest, 
        resolvedAction: action,
        afterMessageId: confirmationShownAfterMessageIdRef.current || undefined,
        redoComment: action === "redo" ? comment : undefined
      }])
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

  const confirmationsMap = getConfirmationsAfterMessage()

  const resolveAgentStep = useCallback((msg: ChatMessage) => {
    if (msg.role !== "agent") return null
    return msg.stepId ?? detectStepFromMessage(msg.content)
  }, [])

  const phaseMarkers = useMemo(() => {
    const markers = new Map<string, string>()
    let lastPhase: string | null = null
    for (const msg of messages) {
      if (msg.role === "agent") {
        const step = msg.stepId ?? detectStepFromMessage(msg.content)
        const phase = step ? STEP_TO_PHASE[step] : null
        if (phase && phase !== lastPhase) {
          markers.set(msg.id, phase)
          lastPhase = phase
        }
      }
    }
    return markers
  }, [messages])

  return (
    <div
      className="flex flex-1 min-h-0 flex-col overflow-hidden dot-grid"
      onClick={handleContainerClick}
    >
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain"
      >
        <div className={`space-y-5 max-w-3xl mx-auto px-6 py-8 ${hasExperimentChecklist ? "pt-14" : ""}`}>
          {messages.length === 0 && (
            <EmptyState onSelectSuggestion={setDraft} />
          )}

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
                    stepId={detectedStep}
                    isClickable={isClickable}
                    onStepClick={isClickable ? () => setSelectedStepId(detectedStep) : undefined}
                    ref={(el) => {
                      if (el) messageRefs.current.set(msg.id, el)
                      else messageRefs.current.delete(msg.id)
                    }}
                  />
                )}
                
                {confsAfterThis.map((conf, index) => (
                  <PastConfirmation
                    key={`past-conf-${msg.id}-${index}`}
                    confirmation={conf}
                    agentState={agentState}
                    steps={steps}
                    onViewDetails={setSelectedStepId}
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
              onViewDetails={setSelectedStepId}
            />
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <ChatInput
        draft={draft}
        onDraftChange={setDraft}
        onSend={handleSend}
        isDisabled={isRunning && !confirmationRequest}
        placeholder={messages.length === 0 ? "Ask a question or attach a dataset to train..." : "Send a message..."}
        selectedDatasets={linkedDatasets}
        onDatasetSelect={handleDatasetSelect}
        onDatasetRemove={handleDatasetRemove}
        selectedModel={linkedModelId}
        onModelSelect={(id) => onLinkedModelChange(id)}
        onModelRemove={() => onLinkedModelChange(null)}
        availableDatasets={availableDatasets}
        availableModels={availableModels}
        useHitl={useHitl}
        onToggleHitl={() => setUseHitl(v => !v)}
        experimentId={experimentId}
      />

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
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.15em] text-muted-foreground/50 font-semibold select-none">
        {Icon && <Icon className="h-3 w-3 text-primary/40 animate-spin-in" />}
        <span>{phase}</span>
      </div>
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border/40 to-transparent" />
    </div>
  )
}
