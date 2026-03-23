import { useState, useEffect, useRef, useCallback } from "react"
import { useRealAgent } from "@/hooks/useRealAgent"
import { ProgressPanel } from "@/components/ProgressPanel"
import { ChatPanel, ChatPanelRef } from "@/components/ChatPanel"
import { ExperimentSidebar } from "@/components/ExperimentSidebar"
import { FinalReport } from "@/components/FinalReport"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Sparkles, RotateCcw, FileText, PanelLeftClose, PanelLeft } from "lucide-react"
import type { StepInfo, TrainingAgentState, ConfirmationAction } from "@/types/agent"
import { createExperiment, saveExperimentMessages } from "@/lib/api"

export default function App() {
  const [showReport, setShowReport] = useState(false)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const chatPanelRef = useRef<ChatPanelRef>(null)

  const realAgent = useRealAgent()

  const agent = {
    agentState: realAgent.agentState,
    steps: realAgent.steps,
    messages: realAgent.messages,
    isRunning: realAgent.isRunning,
    currentStepId: realAgent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")?.id || null,
    confirmationRequest: realAgent.confirmationRequest,
    startAgent: realAgent.startAgent as (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => Promise<void>,
    handleConfirmation: realAgent.handleConfirmation,
    sendMessage: realAgent.sendMessage,
    reset: realAgent.reset,
  }

  const isComplete = agent.agentState.training_metrics?.success

  useEffect(() => {
    realAgent.checkConnection()
    const interval = setInterval(() => realAgent.checkConnection(), 10000)
    return () => clearInterval(interval)
  }, [realAgent.checkConnection])

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

  const [switchingTo, setSwitchingTo] = useState<string | null>(null)

  const handleSelectExperiment = useCallback(async (id: string) => {
    setSwitchingTo(id)
    try {
      await realAgent.loadExperiment(id)
    } finally {
      setSwitchingTo(null)
    }
  }, [realAgent.loadExperiment])

  const handleNewExperiment = useCallback(async () => {
    try {
      // Save current experiment's messages before creating a new one
      if (realAgent.experimentId && realAgent.messages.length > 0) {
        await saveExperimentMessages(
          realAgent.experimentId,
          realAgent.messages.map((m) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
          })),
        )
      }
      const exp = await createExperiment()
      realAgent.reset()
      realAgent.setExperimentId(exp.id)
    } catch {
      realAgent.reset()
    }
  }, [realAgent])

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex-shrink-0 h-14 px-4 flex items-center justify-between border-b border-border/25 bg-background/80 backdrop-blur-sm z-10">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="h-8 w-8 p-0 text-muted-foreground"
            >
              {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
            </Button>
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-primary flex items-center justify-center">
                <Sparkles className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="font-semibold tracking-tight text-[15px]">Jubilee</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isComplete && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowReport(true)}
                className="h-8 text-xs"
              >
                <FileText className="h-3.5 w-3.5 mr-1.5" />
                Report
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={agent.reset}
              className="h-8 w-8 p-0 text-muted-foreground"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </header>

        {/* Backend offline banner */}
        {!realAgent.isBackendConnected && (
          <div className="flex-shrink-0 px-5 py-2 text-sm text-muted-foreground bg-muted/40 border-b border-border/25">
            Run <code className="bg-background px-2 py-0.5 rounded-md text-xs font-mono">docker compose up --build</code> to start
          </div>
        )}

        {/* Main Content */}
        <main className="flex-1 min-h-0 flex overflow-hidden">
          {/* Experiment Sidebar */}
          {sidebarOpen && (
            <div className="w-[260px] flex-shrink-0 border-r border-border/25 hidden md:block">
              <ExperimentSidebar
                activeExperimentId={realAgent.experimentId}
                onSelectExperiment={handleSelectExperiment}
                onNewExperiment={handleNewExperiment}
                isBackendConnected={realAgent.isBackendConnected}
                switchingTo={switchingTo}
              />
            </div>
          )}

          <div className="flex-1 min-w-0 flex h-full">
            {/* Pipeline Progress Panel */}
            <div className="w-[280px] flex-shrink-0 hidden lg:block h-full overflow-auto border-r border-border/25">
              <ProgressPanel
                steps={agent.steps}
                currentStepId={agent.currentStepId}
                agentState={agent.agentState}
                isRunning={agent.isRunning}
                onStepClick={handleStepClick}
              />
            </div>

            {/* Chat Panel */}
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

        {/* Mobile Progress Toggle */}
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
