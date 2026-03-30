import { useState, useEffect, useRef, useCallback, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react"
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
import { RotateCcw, FileText, Search, Bell, ArrowLeft } from "lucide-react"
import type { StepInfo, TrainingAgentState, ConfirmationAction } from "@/types/agent"
import { createExperiment } from "@/lib/api"
import { cn } from "@/lib/utils"
import { DatasetsPage } from "@/components/DatasetsPage"
import { ModelsPage } from "@/components/ModelsPage"
import { Show, SignIn, UserButton, useAuth } from "@clerk/react"

export default function App() {
  const { isLoaded, isSignedIn } = useAuth()

  if (!isLoaded) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    )
  }

  if (!isSignedIn) {
    return <SignInGate />
  }

  return <AuthenticatedApp />
}

function SignInGate() {
  return (
    <div className="fixed inset-0 z-[100] overflow-hidden bg-background">
      <div className="jubilee-dot-grid absolute inset-0 opacity-[0.72]" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(18,86,210,0.14),_transparent_38%),radial-gradient(circle_at_bottom_right,_hsl(var(--primary)/0.08),_transparent_30%)]" />

      <div className="relative flex min-h-screen items-center justify-center px-4 py-10">
        <div className="mx-auto w-full max-w-[440px]">
          <div className="overflow-hidden rounded-[28px] border border-border/60 bg-card/95 shadow-[0_24px_80px_-12px_rgba(18,86,210,0.18),0_8px_40px_rgba(10,26,54,0.06)] backdrop-blur-xl">
            <div className="px-6 pt-8 pb-2 text-center">
              <p className="text-[11px] font-medium uppercase tracking-[0.24em] text-muted-foreground">
                Sign in / Sign up
              </p>
              <div className="mt-3 flex items-center justify-center gap-2.5">
                <img src="/jubilee-logo.svg" alt="" className="h-9 w-9 shrink-0" width={36} height={36} />
                <h1 className="text-[1.95rem] font-semibold tracking-tight text-foreground">
                  Jubilee
                </h1>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Access your training workspace.
              </p>
            </div>

            <div className="jubilee-signin-gate px-6 pb-6">
            <SignIn
              routing="hash"
              withSignUp
              appearance={{
                theme: "simple",
                variables: {
                  colorPrimary: "hsl(222 36% 81%)",
                  colorPrimaryForeground: "hsl(222 58% 24%)",
                  colorForeground: "hsl(222 69% 13%)",
                  colorMutedForeground: "hsl(222 32% 47%)",
                  colorBackground: "hsl(0 0% 100%)",
                  colorInput: "hsl(223 100% 96%)",
                  colorInputForeground: "hsl(222 69% 13%)",
                  colorNeutral: "hsl(221 100% 92%)",
                  colorBorder: "hsl(221 100% 92%)",
                  colorRing: "hsl(218 62% 53%)",
                  colorDanger: "hsl(349 52% 44%)",
                  colorSuccess: "hsl(217 88% 57%)",
                  colorWarning: "hsl(42 96% 58%)",
                  colorShadow: "rgba(18, 86, 210, 0.12)",
                  colorModalBackdrop: "rgba(10, 26, 54, 0.35)",
                  fontFamily: "Inter, system-ui, -apple-system, sans-serif",
                  fontFamilyButtons: "Inter, system-ui, -apple-system, sans-serif",
                  borderRadius: "0.9rem",
                },
                options: {
                  socialButtonsPlacement: "top",
                  socialButtonsVariant: "blockButton",
                },
                elements: {
                  rootBox: "w-full",
                  main: "w-full px-0 pb-0 pt-1",
                  cardBox: "w-full shadow-none",
                  card: "w-full border-0 bg-transparent shadow-none",
                  header: "hidden",
                  headerTitle: "hidden",
                  headerSubtitle: "hidden",
                  socialButtonsBlockButton:
                    "h-11 rounded-xl border border-border bg-muted/70 text-foreground shadow-none hover:bg-accent hover:text-foreground",
                  socialButtonsBlockButtonText: "font-medium",
                  dividerLine: "bg-border",
                  dividerText: "text-muted-foreground text-[10px] uppercase tracking-[0.22em]",
                  formFieldLabel: "text-foreground/90 text-sm font-medium",
                  formFieldInput:
                    "h-11 rounded-xl border border-input bg-muted/60 px-4 text-foreground shadow-none placeholder:text-muted-foreground focus:border-primary focus:bg-card",
                  formFieldInputShowPasswordButton:
                    "text-muted-foreground hover:text-foreground",
                  formButtonPrimary:
                    "h-11 rounded-xl border-0 bg-primary-subtle text-primary-subtle-foreground shadow-none hover:bg-primary-subtle/88",
                  footerActionText: "text-muted-foreground",
                  footerActionLink: "text-primary hover:text-primary/90 font-medium",
                  identityPreviewText: "text-foreground",
                  identityPreviewEditButton: "text-primary hover:text-primary/90",
                  formResendCodeLink: "text-primary hover:text-primary/90",
                  otpCodeFieldInput:
                    "rounded-xl border border-input bg-muted/60 text-foreground shadow-none",
                  alertText: "text-sm",
                  footer:
                    "mt-6 w-full border-t border-border/50 bg-[linear-gradient(180deg,transparent,rgba(18,86,210,0.04))] px-0 pb-0 pt-5",
                },
              }}
            />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function AuthenticatedApp() {
  const SIDEBAR_MIN_WIDTH = 224
  const SIDEBAR_MAX_WIDTH = 420
  const [sidebarWidth, setSidebarWidth] = useState(240)
  const [isResizingSidebar, setIsResizingSidebar] = useState(false)
  const [showReport, setShowReport] = useState(false)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<AppTab>("experiment_lab")
  const [experimentsListNonce, setExperimentsListNonce] = useState(0)
  const chatPanelRef = useRef<ChatPanelRef>(null)

  const bumpExperimentsList = useCallback(() => {
    setExperimentsListNonce((n) => n + 1)
  }, [])

  const realAgent = useRealAgent({
    onExperimentEnsured: () => bumpExperimentsList(),
  })

  const agent = {
    agentState: realAgent.agentState,
    steps: realAgent.steps,
    messages: realAgent.messages,
    isRunning: realAgent.isRunning,
    currentStepId: realAgent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")?.id || null,
    confirmationRequest: realAgent.confirmationRequest,
    startAgent: realAgent.startAgent as (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => Promise<void>, // legacy / programmatic
    handleConfirmation: realAgent.handleConfirmation,
    sendMessage: realAgent.sendMessage,
    reset: realAgent.reset,
  }

  const isComplete = agent.agentState.training_metrics?.success

  // Lightweight backend reachability only (catalog loads once in useRealAgent on mount;
  // refetch datasets/models when user opens those tabs).
  useEffect(() => {
    const ping = () => {
      void realAgent.checkConnection()
    }
    const onVisible = () => {
      if (document.visibilityState === "visible") ping()
    }
    document.addEventListener("visibilitychange", onVisible)
    const interval = setInterval(ping, 60_000)
    return () => {
      document.removeEventListener("visibilitychange", onVisible)
      clearInterval(interval)
    }
  }, [realAgent.checkConnection])

  useEffect(() => {
    if (activeTab === "datasets") void realAgent.refreshDatasets()
    if (activeTab === "models") void realAgent.refreshModelTypes()
  }, [activeTab, realAgent.refreshDatasets, realAgent.refreshModelTypes])

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

  const openBlankExperiment = useCallback(async () => {
    try {
      const exp = await createExperiment()
      bumpExperimentsList()
      await realAgent.loadExperiment(exp.id)
    } catch {
      realAgent.leaveLabSession()
    }
  }, [realAgent, bumpExperimentsList])

  const handleSidebarResizeStart = useCallback((event: ReactMouseEvent<HTMLButtonElement>) => {
    event.preventDefault()
    setIsResizingSidebar(true)
  }, [])

  useEffect(() => {
    if (!isResizingSidebar) return

    const onMouseMove = (event: MouseEvent) => {
      const next = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, event.clientX))
      setSidebarWidth(next)
    }

    const onMouseUp = () => {
      setIsResizingSidebar(false)
    }

    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    window.addEventListener("mousemove", onMouseMove)
    window.addEventListener("mouseup", onMouseUp)

    return () => {
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseup", onMouseUp)
    }
  }, [isResizingSidebar, SIDEBAR_MAX_WIDTH, SIDEBAR_MIN_WIDTH])

  const completedSteps = agent.steps.filter((s) => s.status === "completed").length
  const totalSteps = agent.steps.length
  const currentStep = agent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")
  const trainingPhaseStepIds = ["training_approval", "training", "generate_report"] as const
  const isInTrainingPhase = agent.steps.some(
    (s) =>
      trainingPhaseStepIds.includes(s.id as (typeof trainingPhaseStepIds)[number]) &&
      s.status !== "pending",
  )
  const hasExperimentChecklist = !!realAgent.experimentId && isInTrainingPhase
  const layoutStyle = { "--sidebar-width": `${sidebarWidth}px` } as CSSProperties

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 w-full z-50 border-b border-border/60 bg-background/90 backdrop-blur-xl shadow-[0_1px_0_hsl(var(--border)/0.35),0_12px_40px_-12px_rgba(18,86,210,0.07)] flex items-center justify-between px-6 h-14">
          <div className="flex items-center gap-2.5 font-headline text-xl font-bold tracking-tight text-foreground">
            <img src="/jubilee-logo.svg" alt="" className="h-8 w-8 shrink-0" width={32} height={32} />
            Jubilee
          </div>

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
            <div className="ml-1 flex h-8 w-8 shrink-0 items-center justify-center">
              <Show when="signed-in">
                <UserButton
                  appearance={{
                    elements: {
                      userButtonAvatarBox: "h-8 w-8",
                      userButtonTrigger:
                        "rounded-full focus:shadow-none focus:ring-0 [&:focus-visible]:ring-0 [&:focus-visible]:ring-offset-0",
                    },
                  }}
                />
              </Show>
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
        <div className="flex flex-1 min-h-0 overflow-hidden pt-14" style={layoutStyle}>
          {/* Fixed Sidebar */}
          <aside className="hidden md:flex w-[var(--sidebar-width)] fixed left-0 top-14 bottom-0 flex-col z-40">
            <AppSidebar
              activeTab={activeTab}
              onTabChange={setActiveTab}
              activeExperimentId={realAgent.experimentId}
              onSelectExperiment={handleSelectExperiment}
              onNewExperiment={() => void openBlankExperiment()}
              isBackendConnected={realAgent.isBackendConnected}
              switchingTo={switchingTo}
              experimentsListNonce={experimentsListNonce}
              onExperimentsChanged={bumpExperimentsList}
              onActiveExperimentDeleted={() => {
                realAgent.leaveLabSession()
                bumpExperimentsList()
              }}
            />
            <div className="absolute inset-y-0 -right-2 z-50 hidden md:flex w-4 items-center justify-center">
              <button
                type="button"
                aria-label="Resize sidebar"
                className={cn(
                  "group flex h-full w-full cursor-col-resize items-center justify-center",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0"
                )}
                onMouseDown={handleSidebarResizeStart}
              >
                <span
                  className={cn(
                    "h-10 w-1 rounded-full bg-border/70 transition-colors",
                    "group-hover:bg-primary/70",
                    isResizingSidebar && "bg-primary"
                  )}
                />
              </button>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 min-h-0 flex flex-col overflow-hidden md:ml-[var(--sidebar-width)]">
            {activeTab === "experiment_lab" ? (
              <div className="flex flex-1 min-h-0 flex-col overflow-hidden relative">
                {realAgent.experimentId ? (
                  <div className="flex shrink-0 items-center border-b border-border/40 bg-background/95 px-2 py-1.5 z-20">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="gap-1.5 h-8 text-muted-foreground hover:text-foreground"
                      onClick={() => realAgent.leaveLabSession()}
                    >
                      <ArrowLeft className="h-4 w-4" />
                      Back
                    </Button>
                  </div>
                ) : null}
                <div className="relative flex flex-1 min-h-0 flex-col overflow-hidden">
                  {hasExperimentChecklist && (
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
                    linkedDatasets={realAgent.linkedDatasets}
                    onLinkedDatasetsChange={realAgent.updateLinkedDatasets}
                    datasets={realAgent.datasets}
                    highlightedMessageId={highlightedMessageId}
                    onClearHighlight={handleClearHighlight}
                    onViewReport={() => setShowReport(true)}
                    agentState={agent.agentState}
                    steps={agent.steps}
                    hasExperimentChecklist={hasExperimentChecklist}
                    experimentId={realAgent.experimentId}
                    runningStepHint={realAgent.runningStepHint}
                  />
                </div>
              </div>
            ) : activeTab === "datasets" ? (
              <DatasetsPage datasets={realAgent.datasets} />
            ) : activeTab === "models" ? (
              <ModelsPage modelTypes={realAgent.modelTypes} />
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
