import { useState, useRef, useEffect, useCallback, useImperativeHandle, forwardRef } from "react"
import ReactMarkdown from "react-markdown"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import type { ChatMessage, ConfirmationRequest, ConfirmationAction, Dataset, TrainingAgentState, StepInfo } from "@/types/agent"
import { cn } from "@/lib/utils"
import { AVAILABLE_DATASETS, AVAILABLE_MODELS } from "@/lib/mockAgent"
import type { Dataset as ApiDataset, ModelType } from "@/lib/api"
import { StepDetailModal } from "@/components/StepDetailModal"
import {
  Send,
  User,
  Bot,
  Database,
  Cpu,
  Check,
  RotateCcw,
  FastForward,
  X,
  FileText,
  ChevronRight,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

interface ChatPanelProps {
  messages: ChatMessage[]
  confirmationRequest: ConfirmationRequest | null
  isRunning: boolean
  onSendMessage: (content: string) => void
  onConfirmation: (action: ConfirmationAction, comment?: string) => void
  onStartAgent: (goal: string, datasets?: string[], modelPreference?: string) => void
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

// Step detection - ordered from most specific to least specific
// Each entry has keywords that should be unique to that step
const STEP_PATTERNS: Array<{ stepId: string; patterns: RegExp[] }> = [
  // Report - check first, very specific phrases
  {
    stepId: "generate_report",
    patterns: [
      /training pipeline complete/i,
      /report generated/i,
      /generate_report/i,
      /pipeline complete.*view report/i,
    ]
  },
  // Training - specific training result patterns
  {
    stepId: "training",
    patterns: [
      /training complete/i,
      /model trained/i,
      /test accuracy[:\s]/i,
      /test r[²2][:\s]/i,
      /roc-auc[:\s]/i,
      /training iteration/i,
      /model performance/i,
      /val_accuracy/i,
      /test_accuracy/i,
    ]
  },
  // Feature Engineering - specific to execution
  {
    stepId: "feature_engineering_executor",
    patterns: [
      /feature engineering complete/i,
      /features created/i,
      /features engineered/i,
      /feature engineering executor/i,
      /transformed dataset/i,
    ]
  },
  // Feature Selection - analysis phase
  {
    stepId: "feature_selection_specification",
    patterns: [
      /feature selection complete/i,
      /feature selection specification/i,
      /features specified/i,
      /feature analysis/i,
      /analyzing features/i,
      /correlation analysis/i,
    ]
  },
  // Label/Split Definition
  {
    stepId: "label_split_definition",
    patterns: [
      /label definition/i,
      /label.+split/i,
      /split definition/i,
      /target column[:\s]/i,
      /train\/val\/test/i,
      /split strategy/i,
      /70\/15\/15/i,
    ]
  },
  // Cleaning
  {
    stepId: "cleaning",
    patterns: [
      /cleaning complete/i,
      /data cleaning/i,
      /cleaning.+standardization/i,
      /transformations applied/i,
      /cleaned dataset/i,
      /missing values (filled|imputed|handled)/i,
    ]
  },
  // Data Collection
  {
    stepId: "data_collection",
    patterns: [
      /data collection complete/i,
      /dataset loaded/i,
      /loading dataset/i,
      /loaded.*rows/i,
      /\d+\s+rows.*\d+\s+columns/i,
    ]
  },
  // Model Selection - last because "model" is common
  {
    stepId: "select_model",
    patterns: [
      /model selection complete/i,
      /selected.*as the optimal model/i,
      /selected model[:\s]/i,
      /model:\s*(logistic_regression|random_forest|xgboost|gradient_boost)/i,
      /choosing.*model/i,
    ]
  },
]

function detectStepFromMessage(content: string): string | null {
  // Don't make messages clickable if they already have "View Report" CTA
  // These are final completion messages that have the View Report button
  if (/view report/i.test(content) || /click.*report/i.test(content)) {
    return null
  }
  
  for (const { stepId, patterns } of STEP_PATTERNS) {
    if (patterns.some(pattern => pattern.test(content))) {
      return stepId
    }
  }
  return null
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
  const availableDatasets = propDatasets && propDatasets.length > 0 
    ? propDatasets 
    : AVAILABLE_DATASETS
  const availableModels = propModelTypes && propModelTypes.length > 0
    ? propModelTypes
    : AVAILABLE_MODELS
    
  const [draft, setDraft] = useState("")
  const [redoComment, setRedoComment] = useState("")
  const [showDatasetPicker, setShowDatasetPicker] = useState(false)
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null)
  const messageRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    scrollToMessage: (messageId: string) => {
      const el = messageRefs.current.get(messageId)
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" })
      }
    },
    findMessageByStepName: (stepName: string) => {
      // Find a message that mentions this step
      const stepKeywords: Record<string, string[]> = {
        "select_model": ["Model Selection", "selected model", "xgboost", "random_forest", "logistic_regression"],
        "data_collection": ["Data Collection", "Dataset loaded", "dataset"],
        "cleaning": ["Cleaning", "cleaned", "transformations"],
        "label_split_definition": ["Label", "Split", "target column"],
        "feature_selection_specification": ["Feature Selection", "features specified"],
        "feature_engineering_executor": ["Feature Engineering", "features created"],
        "human_confirmation": ["Confirmation", "confirmed"],
        "training": ["Training", "trained", "R²", "accuracy", "RMSE"],
        "generate_report": ["Report", "complete"],
      }
      const keywords = stepKeywords[stepName] || [stepName]
      
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
  const [showModelPicker, setShowModelPicker] = useState(false)
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages.length])

  const handleSend = () => {
    const text = draft.trim()
    if (!text) return

    // Check if this is starting a new training
    if (messages.length === 0 || text.toLowerCase().includes("train")) {
      onStartAgent(text, selectedDatasets.length > 0 ? selectedDatasets : undefined, selectedModel || undefined)
      // Clear selections after starting
      setSelectedDatasets([])
      setSelectedModel(null)
    } else {
      onSendMessage(text)
    }
    setDraft("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleAtMention = (type: "dataset" | "model") => {
    if (type === "dataset") {
      setShowDatasetPicker(true)
    } else {
      setShowModelPicker(true)
    }
  }

  const selectDataset = (dataset: Dataset) => {
    if (!selectedDatasets.includes(dataset.file)) {
      setSelectedDatasets([...selectedDatasets, dataset.file])
    }
    setShowDatasetPicker(false)
  }

  const selectModel = (modelId: string) => {
    setSelectedModel(modelId)
    setShowModelPicker(false)
  }

  // Clear highlight on click
  const handleContainerClick = useCallback(() => {
    if (highlightedMessageId && onClearHighlight) {
      onClearHighlight()
    }
  }, [highlightedMessageId, onClearHighlight])

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background" onClick={handleContainerClick}>
      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="space-y-5 max-w-3xl mx-auto px-6 py-8">
          {messages.length === 0 && (
            <div className="text-center py-16">
              <h2 className="text-xl font-medium tracking-tight mb-3">What would you like to train?</h2>
              <p className="text-muted-foreground mb-8">
                Describe your goal and I'll handle the rest.
              </p>
              <div className="flex flex-wrap gap-3 justify-center">
                <SuggestionChip
                  label="Predict loan defaults"
                  onClick={() => setDraft("Train a model to predict loan defaults")}
                />
                <SuggestionChip
                  label="Predict insurance costs"
                  onClick={() => setDraft("Train a model to predict insurance costs")}
                />
              </div>
            </div>
          )}

          {messages.map((msg) => {
            const detectedStep = msg.role === "agent" ? detectStepFromMessage(msg.content) : null
            const stepInfo = detectedStep && steps ? steps.find(s => s.id === detectedStep) : null
            const isClickable = !!detectedStep && !!agentState && !!steps && stepInfo?.status === "completed"
            
            return (
              <MessageBubble 
                key={msg.id} 
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
            )
          })}

          {/* Confirmation Panel */}
          {confirmationRequest && (
            <div className="rounded-lg border bg-muted/30 p-4">
              <h3 className="font-semibold flex items-center gap-2 mb-2">
                <Badge variant="warning">Awaiting Confirmation</Badge>
                {confirmationRequest.stepName}
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                {confirmationRequest.summary}
              </p>

              <div className="space-y-3">
                {/* Redo with comment */}
                <div className="flex gap-2">
                  <Textarea
                    value={redoComment}
                    onChange={(e) => setRedoComment(e.target.value)}
                    placeholder="Add feedback for redo (optional)..."
                    className="min-h-[60px] text-sm"
                  />
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      onConfirmation("redo", redoComment || undefined)
                      setRedoComment("")
                    }}
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Redo
                  </Button>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={() => onConfirmation("accept")}
                  >
                    <Check className="h-4 w-4 mr-2" />
                    Accept
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => onConfirmation("accept_all")}
                  >
                    <FastForward className="h-4 w-4 mr-2" />
                    Accept All
                  </Button>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Input - always visible at bottom */}
      <div className="flex-shrink-0 px-6 py-4 max-w-3xl mx-auto w-full">
        <div className="rounded-2xl border border-border bg-card shadow-sm">
          {/* Selected items */}
          {(selectedDatasets.length > 0 || selectedModel) && (
            <div className="flex items-center gap-2 px-4 pt-3 flex-wrap">
              {selectedDatasets.map((ds) => (
                <Badge key={ds} variant="secondary" className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2">
                  <Database className="h-3 w-3 text-muted-foreground" />
                  {ds.split("/").pop()}
                  <button
                    onClick={() => setSelectedDatasets(selectedDatasets.filter((d) => d !== ds))}
                    className="ml-0.5 hover:text-foreground text-muted-foreground"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
              {selectedModel && (
                <Badge variant="secondary" className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2">
                  <Cpu className="h-3 w-3 text-muted-foreground" />
                  {selectedModel}
                  <button
                    onClick={() => setSelectedModel(null)}
                    className="ml-0.5 hover:text-foreground text-muted-foreground"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              )}
            </div>
          )}

          {/* Textarea */}
          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={messages.length === 0 ? "Describe what you want to train..." : "Send a message..."}
            className="min-h-[56px] max-h-[200px] resize-none border-0 shadow-none focus-visible:ring-0 px-4 py-3 text-[15px] placeholder:text-muted-foreground/60"
            disabled={isRunning && !confirmationRequest}
          />

          {/* Actions bar */}
          <div className="flex items-center justify-between px-3 pb-3">
            <div className="flex items-center gap-1">
              <Dialog open={showDatasetPicker} onOpenChange={setShowDatasetPicker}>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-8 px-3 text-muted-foreground hover:text-foreground" onClick={() => handleAtMention("dataset")}>
                    <Database className="h-4 w-4" />
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="text-lg font-medium">Select Dataset</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-1 max-h-[350px] overflow-y-auto -mx-2">
                    {availableDatasets.map((ds) => (
                      <button
                        key={ds.file}
                        className="w-full text-left px-4 py-3 rounded-xl hover:bg-muted/60 transition-colors"
                        onClick={() => selectDataset(ds as Dataset)}
                      >
                        <div className="font-medium">{ds.name}</div>
                        <div className="text-sm text-muted-foreground mt-0.5">{ds.rows?.toLocaleString()} rows</div>
                      </button>
                    ))}
                  </div>
                </DialogContent>
              </Dialog>

              <Dialog open={showModelPicker} onOpenChange={setShowModelPicker}>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-8 px-3 text-muted-foreground hover:text-foreground" onClick={() => handleAtMention("model")}>
                    <Cpu className="h-4 w-4" />
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="text-lg font-medium">Select Model</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-1 -mx-2">
                    {availableModels.map((model) => (
                      <button
                        key={model.id}
                        className="w-full text-left px-4 py-3 rounded-xl hover:bg-muted/60 transition-colors"
                        onClick={() => selectModel(model.id)}
                      >
                        <div className="font-medium">{model.name}</div>
                        <div className="text-sm text-muted-foreground mt-0.5">{model.description}</div>
                      </button>
                    ))}
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            <Button
              size="sm"
              className="h-8 px-3 rounded-lg"
              onClick={handleSend}
              disabled={(isRunning && !confirmationRequest) || !draft.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

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

const MAX_CONTENT_LENGTH = 400 // Characters before truncation

interface MessageBubbleProps {
  message: ChatMessage
  isHighlighted?: boolean
  onViewReport?: () => void
  stepId?: string | null
  isClickable?: boolean
  onStepClick?: () => void
}

const MessageBubble = forwardRef<HTMLDivElement, MessageBubbleProps>(
  function MessageBubble({ message, isHighlighted, onViewReport, stepId, isClickable, onStepClick }, ref) {
    const [isExpanded, setIsExpanded] = useState(false)
    const isUser = message.role === "user"
    const isSystem = message.role === "system"
    
    // Check if this message mentions "View Report" - show a clickable button
    const hasViewReport = !isUser && !isSystem && 
      (message.content.toLowerCase().includes("view report") || 
       message.content.toLowerCase().includes("training completed"))

    const shouldTruncate = message.content.length > MAX_CONTENT_LENGTH
    const displayContent = shouldTruncate && !isExpanded
      ? message.content.slice(0, MAX_CONTENT_LENGTH) + "..."
      : message.content

    if (isSystem) {
      return (
        <div ref={ref} className="text-xs text-muted-foreground/70 text-center py-2">
          {message.content}
        </div>
      )
    }

    const handleBubbleClick = () => {
      if (isClickable && onStepClick) {
        onStepClick()
      }
    }

    return (
      <div 
        ref={ref}
        className={cn(
          "flex gap-3 transition-all duration-300",
          isUser && "flex-row-reverse",
          isHighlighted && "scale-[1.02]"
        )}
      >
        <div
          className={cn(
            "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center",
            isUser ? "bg-foreground text-background" : "bg-muted"
          )}
        >
          {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5 text-muted-foreground" />}
        </div>
        <div
          onClick={handleBubbleClick}
          className={cn(
            "flex-1 max-w-[92%] rounded-2xl px-4 py-3 transition-all duration-300",
            isUser ? "bg-foreground text-background" : "bg-muted/50",
            isHighlighted && "ring-2 ring-foreground/20 shadow-lg",
            isClickable && "cursor-pointer hover:bg-muted/70 hover:shadow-md group"
          )}
        >
          <div className="text-[15px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-headings:font-medium prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-code:bg-black/5 prose-code:dark:bg-white/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-[13px] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-strong:font-semibold">
            <ReactMarkdown>{displayContent}</ReactMarkdown>
          </div>
          
          {/* Show more/less button */}
          {shouldTruncate && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                setIsExpanded(!isExpanded)
              }}
              className="mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors font-medium"
            >
              {isExpanded ? "Show less" : "Show more"}
            </button>
          )}
          
          {/* View Details hint for clickable messages */}
          {isClickable && (
            <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground group-hover:text-foreground transition-colors">
              <span>Click for details</span>
              <ChevronRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
            </div>
          )}
          
          {/* View Report button */}
          {hasViewReport && onViewReport && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onViewReport()
              }}
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 transition-colors"
            >
              <FileText className="h-3.5 w-3.5" />
              View Report
            </button>
          )}
        </div>
      </div>
    )
  }
)

function SuggestionChip({
  label,
  onClick,
}: {
  label: string
  onClick: () => void
}) {
  return (
    <button
      className="px-5 py-2.5 rounded-full border border-border bg-background hover:bg-muted/50 text-sm font-medium transition-all hover:border-muted-foreground/20"
      onClick={onClick}
    >
      {label}
    </button>
  )
}
