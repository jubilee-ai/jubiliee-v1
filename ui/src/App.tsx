import { useState, useEffect, useRef, useCallback } from "react"
import { useRealAgent } from "@/hooks/useRealAgent"
import { ProgressPanel } from "@/components/ProgressPanel"
import { ChatPanel, ChatPanelRef } from "@/components/ChatPanel"
import { FinalReport } from "@/components/FinalReport"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Sparkles, RotateCcw, FileText } from "lucide-react"
import type { StepInfo, TrainingAgentState, ConfirmationAction } from "@/types/agent"

export default function App() {
  const [showReport, setShowReport] = useState(false)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const chatPanelRef = useRef<ChatPanelRef>(null)
  
  // Real agent hook
  const realAgent = useRealAgent()
  
  const agent = {
    agentState: realAgent.agentState,
    steps: realAgent.steps,
    messages: realAgent.messages,
    isRunning: realAgent.isRunning,
    currentStepId: realAgent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")?.id || null,
    confirmationRequest: realAgent.confirmationRequest,
    startAgent: realAgent.startAgent as (goal: string, datasets?: string[], modelPreference?: string, simple?: boolean, hitl?: boolean) => Promise<void>,
    handleConfirmation: realAgent.handleConfirmation,
    sendMessage: realAgent.sendMessage,
    reset: realAgent.reset,
  }
  
  const isComplete = agent.agentState.training_metrics?.success

  // Check backend connection periodically
  useEffect(() => {
    realAgent.checkConnection()
    const interval = setInterval(() => realAgent.checkConnection(), 10000)
    return () => clearInterval(interval)
  }, [realAgent.checkConnection])

  // Handle step click from progress panel
  const handleStepClick = useCallback((stepId: string) => {
    if (chatPanelRef.current) {
      const messageId = chatPanelRef.current.findMessageByStepName(stepId)
      if (messageId) {
        setHighlightedMessageId(messageId)
        chatPanelRef.current.scrollToMessage(messageId)
      }
    }
  }, [])

  const handleClearHighlight = useCallback(() => {
    setHighlightedMessageId(null)
  }, [])

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Minimal Header */}
        <header className="flex-shrink-0 h-14 px-5 flex items-center justify-between border-b border-border/50">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-foreground flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-background" />
            </div>
            <span className="font-semibold tracking-tight">Jubilee</span>
          </div>

          <div className="flex items-center gap-2">
            {isComplete && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowReport(true)}
                className="h-8"
              >
                <FileText className="h-4 w-4 mr-1.5" />
                View Report
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={agent.reset}
              className="h-8 w-8 p-0"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </header>

        {/* Backend offline banner */}
        {!realAgent.isBackendConnected && (
          <div className="flex-shrink-0 px-5 py-2.5 text-sm text-muted-foreground bg-muted/40 border-b border-border/50">
            Run <code className="bg-background px-2 py-0.5 rounded-md text-xs font-mono">uvicorn app:app --reload</code> to start
          </div>
        )}

        {/* Main Content */}
        <main className="flex-1 min-h-0 flex overflow-hidden">
          <div className="w-full flex h-full">
            {/* Left Panel - Progress */}
            <div className="w-[340px] flex-shrink-0 hidden lg:block h-full overflow-auto border-r border-border/50">
              <ProgressPanel
                steps={agent.steps}
                currentStepId={agent.currentStepId}
                agentState={agent.agentState}
                isRunning={agent.isRunning}
                onStepClick={handleStepClick}
              />
            </div>

            {/* Right Panel - Chat */}
            <div className="flex-1 min-w-0 h-full">
              <ChatPanel
                ref={chatPanelRef}
                messages={agent.messages}
                confirmationRequest={agent.confirmationRequest}
                isRunning={agent.isRunning}
                onSendMessage={agent.sendMessage}
                onConfirmation={agent.handleConfirmation as (action: ConfirmationAction, comment?: string) => void}
                onStartAgent={agent.startAgent}
                datasets={realAgent.datasets}
                modelTypes={realAgent.modelTypes}
                highlightedMessageId={highlightedMessageId}
                onClearHighlight={handleClearHighlight}
                onViewReport={() => setShowReport(true)}
                agentState={agent.agentState}
                steps={agent.steps}
              />
            </div>
          </div>
        </main>

        {/* Mobile Progress Toggle (for smaller screens) */}
        <div className="lg:hidden fixed bottom-4 left-4 z-50">
          <MobileProgressButton
            steps={agent.steps}
            currentStepId={agent.currentStepId}
            agentState={agent.agentState}
            isRunning={agent.isRunning}
          />
        </div>

        {/* Final Report Modal */}
        {showReport && (
          <FinalReport
            agentState={agent.agentState}
            steps={agent.steps}
            onClose={() => setShowReport(false)}
          />
        )}
      </div>
    </TooltipProvider>
  )
}

// Mobile progress button component
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"

function MobileProgressButton({
  steps,
  currentStepId,
  agentState,
  isRunning,
}: {
  steps: StepInfo[]
  currentStepId: string | null
  agentState: TrainingAgentState
  isRunning: boolean
}) {
  const completedSteps = steps.filter((s) => s.status === "completed").length

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="default" size="lg" className="rounded-full shadow-lg">
          <Sparkles className="h-5 w-5 mr-2" />
          {completedSteps}/{steps.length} Steps
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[90vw] max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Pipeline Progress</DialogTitle>
        </DialogHeader>
        <div className="flex-1 overflow-auto">
          <ProgressPanel
            steps={steps}
            currentStepId={currentStepId}
            agentState={agentState}
            isRunning={isRunning}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}
