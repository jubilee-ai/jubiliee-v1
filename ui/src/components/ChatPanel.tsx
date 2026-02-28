import React, { useState, useRef, useEffect, useCallback, useImperativeHandle, forwardRef } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { ChatMessage, ConfirmationRequest, ConfirmationAction, Dataset, TrainingAgentState, StepInfo } from "@/types/agent"
import type { Dataset as ApiDataset, ModelType } from "@/lib/api"
import { AVAILABLE_DATASETS, AVAILABLE_MODELS } from "@/lib/mockAgent"
import { StepDetailModal } from "@/components/StepDetailModal"

// Import subcomponents
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
import type { ResolvedConfirmation } from "./chat"

interface ChatPanelProps {
  messages: ChatMessage[]
  confirmationRequest: ConfirmationRequest | null
  isRunning: boolean
  onSendMessage: (content: string) => void
  onConfirmation: (action: ConfirmationAction, comment?: string) => void
  onStartAgent: (goal: string, datasets?: string[], modelPreference?: string, simple?: boolean, hitl?: boolean) => void
  datasets?: ApiDataset[]
  modelTypes?: ModelType[]
  highlightedMessageId?: string | null
  onClearHighlight?: () => void
  onViewReport?: () => void
  agentState?: TrainingAgentState
  steps?: StepInfo[]
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
  onStartAgent,
  datasets: propDatasets,
  modelTypes: propModelTypes,
  highlightedMessageId,
  onClearHighlight,
  onViewReport,
  agentState,
  steps,
}, ref) {
  // Use provided datasets/models or fall back to defaults
  const availableDatasets = propDatasets && propDatasets.length > 0 
    ? propDatasets 
    : AVAILABLE_DATASETS
  const availableModels = propModelTypes && propModelTypes.length > 0
    ? propModelTypes
    : AVAILABLE_MODELS

  // Local state
  const [draft, setDraft] = useState("")
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null)
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [useSimple, setUseSimple] = useState(false)
  const [useHitl, setUseHitl] = useState(true)
  
  // Refs
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // Track past confirmation requests so they persist in the chat
  const [pastConfirmations, setPastConfirmations] = useState<ResolvedConfirmation[]>([])
  const confirmationShownAfterMessageIdRef = useRef<string | null>(null)
  const prevConfirmationStepRef = useRef<string | null>(null)

  // Auto-scroll to new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages.length])

  // Track which message the current confirmation appeared after
  useEffect(() => {
    if (confirmationRequest && confirmationRequest.step !== prevConfirmationStepRef.current) {
      // A new confirmation just appeared - record the last message ID
      const lastMessage = messages[messages.length - 1]
      confirmationShownAfterMessageIdRef.current = lastMessage?.id || null
      prevConfirmationStepRef.current = confirmationRequest.step
    }
    if (!confirmationRequest) {
      prevConfirmationStepRef.current = null
    }
  }, [confirmationRequest, messages])

  // Clear past confirmations when messages are reset
  useEffect(() => {
    if (messages.length === 0) {
      setPastConfirmations([])
      confirmationShownAfterMessageIdRef.current = null
      prevConfirmationStepRef.current = null
    }
  }, [messages.length])

  // Create a map of messageId -> past confirmations that should appear after it
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

  // Handle confirmation with tracking (saves confirmation before it disappears)
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

  // Expose methods to parent via ref
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

  // Handle sending messages
  const handleSend = useCallback(() => {
    const text = draft.trim()
    if (!text) return

    // Check if this is starting a new training
    if (messages.length === 0 || text.toLowerCase().includes("train")) {
      onStartAgent(text, selectedDatasets.length > 0 ? selectedDatasets : undefined, selectedModel || undefined, useSimple, useHitl)
      setSelectedDatasets([])
      setSelectedModel(null)
    } else {
      onSendMessage(text)
    }
    setDraft("")
  }, [draft, messages.length, onStartAgent, onSendMessage, selectedDatasets, selectedModel, useSimple, useHitl])

  // Handle dataset selection
  const handleDatasetSelect = useCallback((dataset: Dataset) => {
    if (!selectedDatasets.includes(dataset.file)) {
      setSelectedDatasets(prev => [...prev, dataset.file])
    }
  }, [selectedDatasets])

  const handleDatasetRemove = useCallback((datasetFile: string) => {
    setSelectedDatasets(prev => prev.filter(d => d !== datasetFile))
  }, [])

  // Clear highlight on container click
  const handleContainerClick = useCallback(() => {
    if (highlightedMessageId && onClearHighlight) {
      onClearHighlight()
    }
  }, [highlightedMessageId, onClearHighlight])

  const confirmationsMap = getConfirmationsAfterMessage()

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background" onClick={handleContainerClick}>
      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-5 max-w-3xl mx-auto px-6 py-8">
          {/* Empty state */}
          {messages.length === 0 && (
            <EmptyState onSelectSuggestion={setDraft} />
          )}

          {/* Message list with past confirmations */}
          {messages.map((msg) => {
            const detectedStep = msg.role === "agent" ? detectStepFromMessage(msg.content) : null
            const stepInfo = detectedStep && steps ? steps.find(s => s.id === detectedStep) : null
            const isClickable = !!detectedStep && !!agentState && !!steps && stepInfo?.status === "completed"
            const confsAfterThis = confirmationsMap.get(msg.id) || []
            
            return (
              <React.Fragment key={msg.id}>
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
                
                {/* Render past confirmations that appeared after this message */}
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

          {/* Loading indicator */}
          {isRunning && !confirmationRequest && (
            <LoadingIndicator />
          )}

          {/* Current confirmation panel */}
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
      </ScrollArea>

      {/* Input area */}
      <ChatInput
        draft={draft}
        onDraftChange={setDraft}
        onSend={handleSend}
        isDisabled={isRunning && !confirmationRequest}
        placeholder={messages.length === 0 ? "Describe what you want to train..." : "Send a message..."}
        selectedDatasets={selectedDatasets}
        onDatasetSelect={handleDatasetSelect}
        onDatasetRemove={handleDatasetRemove}
        selectedModel={selectedModel}
        onModelSelect={setSelectedModel}
        onModelRemove={() => setSelectedModel(null)}
        availableDatasets={availableDatasets}
        availableModels={availableModels}
        useSimple={useSimple}
        onToggleSimple={() => setUseSimple(v => !v)}
        useHitl={useHitl}
        onToggleHitl={() => setUseHitl(v => !v)}
      />

      {/* Step Detail Modal */}
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
