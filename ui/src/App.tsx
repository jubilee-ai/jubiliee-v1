import { useState, useEffect, useRef, useCallback } from "react"
import { useRealAgent } from "@/hooks/useRealAgent"
import { ProgressPanel } from "@/components/ProgressPanel"
import { ChatPanel, ChatPanelRef } from "@/components/ChatPanel"
import { AppSidebar, type AppTab } from "@/components/AppSidebar"
import { FinalReport } from "@/components/FinalReport"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { RotateCcw, FileText, Search, Bell } from "lucide-react"
import type { StepInfo, TrainingAgentState, ConfirmationAction } from "@/types/agent"
import { createExperiment, saveExperimentMessages } from "@/lib/api"
import { cn } from "@/lib/utils"

export default function App() {
  const [showReport, setShowReport] = useState(false)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<AppTab>("experiment_lab")
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

  const completedSteps = agent.steps.filter((s) => s.status === "completed").length
  const totalSteps = agent.steps.length
  const currentStep = agent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")
  const hasActivity = agent.isRunning || completedSteps > 0

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-xl shadow-[0_1px_0_hsl(var(--border)/0.3)] flex items-center justify-between px-6 h-14">
          <div className="font-headline text-xl font-bold tracking-tight text-foreground">Jubilee</div>

          <div className="flex items-center gap-1.5">
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
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground">
              <Search className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground">
              <Bell className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={agent.reset}
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center ml-1">
              <span className="text-[11px] font-semibold text-primary">U</span>
            </div>
          </div>
        </nav>

        {/* Backend offline banner */}
        {!realAgent.isBackendConnected && (
          <div className="fixed top-14 left-0 right-0 z-40 px-5 py-1.5 text-xs text-muted-foreground bg-muted text-center">
            Run <code className="bg-card px-1.5 py-0.5 rounded text-[11px] font-mono">docker compose up --build</code> to start
          </div>
        )}

        {/* Main Layout — min-h-0 so inner chat can scroll instead of growing the page */}
        <div className="flex flex-1 min-h-0 overflow-hidden pt-14">
          {/* Fixed Sidebar */}
          <aside className="hidden md:flex w-60 fixed left-0 top-14 bottom-0 flex-col z-40">
            <AppSidebar
              activeTab={activeTab}
              onTabChange={setActiveTab}
              activeExperimentId={realAgent.experimentId}
              onSelectExperiment={handleSelectExperiment}
              onNewExperiment={handleNewExperiment}
              isBackendConnected={realAgent.isBackendConnected}
              switchingTo={switchingTo}
            />
          </aside>

          {/* Main Content */}
          <main className="flex-1 min-h-0 flex flex-col overflow-hidden md:ml-60">
            {activeTab === "experiment_lab" ? (
              <div className="flex flex-1 min-h-0 flex-col overflow-hidden relative">
                {/* Floating experiment checklist */}
                {hasActivity && (
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10">
                    <ExperimentChecklistIndicator
                      steps={agent.steps}
                      currentStep={currentStep ?? null}
                      completedSteps={completedSteps}
                      totalSteps={totalSteps}
                      agentState={agent.agentState}
                      isRunning={agent.isRunning}
                      onStepClick={handleStepClick}
                    />
                  </div>
                )}

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
                  hasExperimentChecklist={hasActivity}
                />
              </div>
            ) : activeTab === "datasets" ? (
              <div className="flex-1 overflow-auto">
                <div className="max-w-5xl mx-auto px-8 py-10">
                  <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">Data Management</span>
                  <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">Data Assets</h1>
                  <p className="mt-3 text-muted-foreground text-sm leading-relaxed max-w-lg">Manage and explore your datasets for training and evaluation.</p>
                  <div className="mt-10 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                    Coming soon.
                  </div>
                </div>
              </div>
            ) : activeTab === "models" ? (
              <div className="flex-1 overflow-auto">
                <div className="max-w-5xl mx-auto px-8 py-10">
                  <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">Model Management</span>
                  <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">Model Registry</h1>
                  <p className="mt-3 text-muted-foreground text-sm leading-relaxed max-w-lg">Monitor deployments, track performance metrics, and manage your model lifecycle.</p>
                  <div className="mt-10 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                    Coming soon.
                  </div>
                </div>
              </div>
            ) : activeTab === "settings" ? (
              <div className="flex-1 overflow-auto">
                <div className="max-w-5xl mx-auto px-8 py-10">
                  <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">Configuration</span>
                  <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">Settings & API</h1>
                  <p className="mt-3 text-muted-foreground text-sm leading-relaxed max-w-lg">Account management, API keys, and MCP access configuration.</p>
                  <div className="mt-10 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                    Coming soon.
                  </div>
                </div>
              </div>
            ) : null}
          </main>
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

function ExperimentChecklistIndicator({
  steps,
  currentStep,
  completedSteps,
  totalSteps,
  agentState,
  isRunning,
  onStepClick,
}: {
  steps: StepInfo[]
  currentStep: StepInfo | null
  completedSteps: number
  totalSteps: number
  agentState: TrainingAgentState
  isRunning: boolean
  onStepClick: (stepId: string) => void
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button className="inline-flex items-center gap-2.5 h-8 px-3.5 rounded-full bg-card/95 backdrop-blur-md shadow-[0_2px_8px_rgba(0,0,0,0.06)] border border-border/40 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] hover:border-border/60 transition-all cursor-pointer group">
          <div className="flex items-center gap-[3px]">
            {steps.map((s, i) => (
              <div
                key={i}
                className={cn(
                  "w-[5px] h-[5px] rounded-full transition-colors",
                  s.status === "completed" && "bg-[hsl(var(--step-complete))]",
                  (s.status === "running" || s.status === "awaiting_confirmation") && "bg-primary animate-pulse",
                  (s.status === "pending" || s.status === "skipped") && "bg-muted-foreground/20",
                  s.status === "error" && "bg-destructive",
                  s.status === "stale" && "bg-amber-400",
                )}
              />
            ))}
          </div>
          <span className="text-[11px] text-muted-foreground font-medium tabular-nums">
            {completedSteps}/{totalSteps}
          </span>
          {currentStep && (
            <>
              <div className="w-px h-3 bg-border/40" />
              <span className="text-[11px] text-muted-foreground truncate max-w-[140px] group-hover:text-foreground transition-colors">
                {currentStep.name}
              </span>
            </>
          )}
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-md max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="font-headline">Experiment checklist</DialogTitle>
        </DialogHeader>
        <div className="flex-1 overflow-auto -mx-6 px-6">
          <ProgressPanel
            steps={steps}
            currentStepId={currentStep?.id ?? null}
            agentState={agentState}
            isRunning={isRunning}
            onStepClick={onStepClick}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}
