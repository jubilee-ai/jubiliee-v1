import { useState, useEffect, useCallback, useMemo, type CSSProperties, type MouseEvent as ReactMouseEvent } from "react"
import { useRealAgent } from "@/hooks/useRealAgent"
import { ChatPanel } from "@/components/ChatPanel"
import { AppSidebar, type AppTab } from "@/components/AppSidebar"
import { RegistryFinalReport } from "@/components/TrainingReportJsonDialog"
import { FinalReport } from "@/components/FinalReport"
import { trainingReportModelLabel } from "@/lib/trainingReport"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { RotateCcw, Search, Bell, ArrowLeft } from "lucide-react"
import type { ConfirmationAction, TaskPlanSummary } from "@/types/agent"
import { filterVisiblePipelineSteps } from "@/lib/trainingSteps"
import { cn } from "@/lib/utils"
import { useExperimentDetailQuery } from "@/lib/queries"
import { DatasetsPage } from "@/components/DatasetsPage"
import { ModelsPage } from "@/components/ModelsPage"
import { InsuranceBrokerDemoPage } from "@/components/InsuranceBrokerDemoPage"
import { ModelRiskProfilePage } from "@/components/model-risk/ModelRiskProfilePage"
import { Show, SignIn, UserButton, useAuth } from "@clerk/react"

function appDebug(event: string, payload?: Record<string, unknown>) {
  const ts = new Date().toISOString()
  if (payload) {
    console.log(`[app:experiment-switch][${ts}] ${event}`, payload)
    return
  }
  console.log(`[app:experiment-switch][${ts}] ${event}`)
}

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
              <p className="text-caption font-medium uppercase tracking-[0.24em] text-muted-foreground">
                Sign in / Sign up
              </p>
              <div className="mt-3 flex items-center justify-center gap-2.5">
                <img src="/jubilee-logo.svg" alt="" className="h-9 w-9 shrink-0" width={36} height={36} />
                <h1 className="font-headline text-3xl font-semibold tracking-tight text-foreground">
                  Jubilee
                </h1>
              </div>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
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
                  dividerText: "text-muted-foreground text-overline uppercase tracking-[0.22em]",
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
  /** Single report dialog for chat + model registry (same component, same behavior). */
  const [reportView, setReportView] = useState<
    null | { kind: "chat" } | { kind: "registry"; modelName: string }
  >(null)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<AppTab>("broker_demo")
  /** When set, TanStack Query loads this experiment from the API and applies it to the lab hook (sidebar selection). */
  const [experimentDetailQueryId, setExperimentDetailQueryId] = useState<string | null>(null)
  /** When set, Models tab scrolls to and briefly highlights this trained model row. */
  const [modelsScrollToModelName, setModelsScrollToModelName] = useState<string | null>(null)
  /** Model Risk profile drill-in from Models registry (same tab, no router). */
  const [selectedModelRiskName, setSelectedModelRiskName] = useState<string | null>(null)
  const realAgent = useRealAgent()

  const experimentDetailQuery = useExperimentDetailQuery(experimentDetailQueryId)

  /** Full step state stays in the hook for SSE; chat hides feature steps for unsupervised. */
  const visiblePipelineSteps = useMemo(
    () => filterVisiblePipelineSteps(realAgent.steps, realAgent.agentState),
    [realAgent.steps, realAgent.agentState],
  )

  const agent = {
    agentState: realAgent.agentState,
    steps: visiblePipelineSteps,
    messages: realAgent.messages,
    isRunning: realAgent.isRunning,
    currentStepId:
      visiblePipelineSteps.find(
        (s) => s.status === "running" || s.status === "awaiting_confirmation",
      )?.id || null,
    confirmationRequest: realAgent.confirmationRequest,
    startAgent: realAgent.startAgent as (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => Promise<void>, // legacy / programmatic
    handleConfirmation: realAgent.handleConfirmation,
    sendMessage: realAgent.sendMessage,
    reset: realAgent.reset,
  }

  /** Registry link by experiment id, else model name from agent state (metrics / weights path). */
  const trainedModelNameForCurrentExperiment = useMemo(() => {
    if (!realAgent.experimentId) return null
    const linked = realAgent.trainedModels.find(
      (m) => m.experiment_id === realAgent.experimentId,
    )
    if (linked) return linked.model_name
    return trainingReportModelLabel(agent.agentState)
  }, [realAgent.experimentId, realAgent.trainedModels, agent.agentState])

  const handleOpenTrainedModelForExperiment = useCallback(() => {
    const name = trainedModelNameForCurrentExperiment
    if (!name) return
    void realAgent.refreshTrainedModels()
    setModelsScrollToModelName(name)
    setActiveTab("models")
  }, [trainedModelNameForCurrentExperiment, realAgent.refreshTrainedModels])

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
    if (activeTab === "models") {
      void realAgent.refreshModelTypes()
      void realAgent.refreshTrainedModels()
    }
  }, [activeTab, realAgent.refreshDatasets, realAgent.refreshModelTypes, realAgent.refreshTrainedModels])

  useEffect(() => {
    if (activeTab !== "models") setSelectedModelRiskName(null)
  }, [activeTab])

  const taskRunning =
    realAgent.agentState.lab_mode === "task" &&
    realAgent.agentState.task_status === "running" &&
    !!realAgent.experimentId

  useEffect(() => {
    if (!taskRunning) return
    const t = window.setInterval(() => {
      void realAgent.refreshExperimentTraining()
    }, 5000)
    return () => window.clearInterval(t)
  }, [taskRunning, realAgent.refreshExperimentTraining])

  useEffect(() => {
    const d = experimentDetailQuery.data
    const qid = experimentDetailQueryId
    appDebug("detail-query-effect", {
      selectedQueryId: qid,
      queryDataId: d?.id ?? null,
      dataUpdatedAt: experimentDetailQuery.dataUpdatedAt,
      isSuccess: experimentDetailQuery.isSuccess,
      isPending: experimentDetailQuery.isPending,
      isFetching: experimentDetailQuery.isFetching,
      activeExperimentId: realAgent.experimentId,
      isRunning: realAgent.isRunning,
      queryError: experimentDetailQuery.error ? String(experimentDetailQuery.error) : null,
    })
    if (!d?.id || qid == null) return
    if (d.id !== qid) return
    if (!experimentDetailQuery.isSuccess) return
    // While actively streaming in this same experiment, do not replace local state from query cache updates.
    if (realAgent.experimentId === qid && realAgent.isRunning) {
      appDebug("apply-skip-running-stream", { experimentId: qid })
      return
    }
    appDebug("apply-experiment-detail", {
      experimentId: d.id,
      chatHistoryCount: Array.isArray(d.chat_history) ? d.chat_history.length : null,
      hasTrainingState: !!d.training_state,
    })
    realAgent.applyExperimentDetail(d)
  }, [
    experimentDetailQuery.data,
    experimentDetailQuery.dataUpdatedAt,
    experimentDetailQuery.isSuccess,
    experimentDetailQueryId,
    realAgent.experimentId,
    realAgent.isRunning,
    realAgent.applyExperimentDetail,
  ])

  const handleClearHighlight = useCallback(() => {
    setHighlightedMessageId(null)
  }, [])

  /** Row spinner: only while the selected experiment is not yet applied to the lab hook.
   * Do not key off `isFetching` — background refetches keep `isFetching` true and would leave the spinner stuck after load. */
  const switchingTo =
    experimentDetailQueryId &&
    realAgent.experimentId !== experimentDetailQueryId &&
    !experimentDetailQuery.isError
      ? experimentDetailQueryId
      : null

  const handleSelectExperiment = useCallback(
    (id: string) => {
      appDebug("sidebar-select-start", {
        selectedId: id,
        currentActiveExperimentId: realAgent.experimentId,
        currentQuerySelectionId: experimentDetailQueryId,
        currentMessagesCount: realAgent.messages.length,
      })

      // Deduplicate repeated clicks while already targeting this experiment.
      if (experimentDetailQueryId === id) {
        appDebug("sidebar-select-skip-already-targeted", { selectedId: id })
        return
      }

      // Switch immediately; persist previous conversation in background so a slow save
      // cannot block experiment selection/UI updates.
      setExperimentDetailQueryId(id)
      appDebug("sidebar-select-set-query-id", { selectedId: id })
      void realAgent.saveCurrentMessages()
      appDebug("sidebar-select-triggered-background-save", { selectedId: id })
    },
    [experimentDetailQueryId, realAgent],
  )

  const handleOpenExperimentFromModels = useCallback(
    (experimentId: string) => {
      setActiveTab("experiment_lab")
      handleSelectExperiment(experimentId)
    },
    [handleSelectExperiment],
  )

  const handleStartBlankChat = useCallback(() => {
    appDebug("start-blank-chat", {
      previousExperimentId: realAgent.experimentId,
      previousMessagesCount: realAgent.messages.length,
    })
    void realAgent.saveCurrentMessages()
    setExperimentDetailQueryId(null)
    realAgent.leaveLabSession()
  }, [realAgent])

  const handleLeaveLabSession = useCallback(() => {
    appDebug("leave-lab-session", {
      previousExperimentId: realAgent.experimentId,
      previousMessagesCount: realAgent.messages.length,
    })
    setExperimentDetailQueryId(null)
    realAgent.leaveLabSession()
  }, [realAgent])

  const handleResetWorkspace = useCallback(() => {
    appDebug("reset-workspace", {
      previousExperimentId: realAgent.experimentId,
      previousMessagesCount: realAgent.messages.length,
    })
    setExperimentDetailQueryId(null)
    realAgent.reset()
  }, [realAgent])

  useEffect(() => {
    appDebug("selection-state-changed", {
      querySelectedExperimentId: experimentDetailQueryId,
      realAgentExperimentId: realAgent.experimentId,
      messagesCount: realAgent.messages.length,
      isRunning: realAgent.isRunning,
      switchingTo,
    })
  }, [
    experimentDetailQueryId,
    realAgent.experimentId,
    realAgent.messages.length,
    realAgent.isRunning,
    switchingTo,
  ])

  /** From chat: keep the interactive training pipeline in this session (SSE graph). */
  const handleApproveTrainingPlanGuided = useCallback(
    (messageId: string, plan: TaskPlanSummary, refs: string[]) => {
      realAgent.markTaskPlanResolved(messageId)
      realAgent.startGuidedTrainingFromPlan(plan, refs)
    },
    [realAgent],
  )

  /** Optional: async run without tying the UI to live graph steps (lab_mode task). */
  const handleApproveTrainingPlanBackground = useCallback(
    async (messageId: string, plan: TaskPlanSummary, refs: string[]) => {
      realAgent.markTaskPlanResolved(messageId)
      await realAgent.startHandsOffTrainingFromPlan(plan, refs)
    },
    [realAgent],
  )

  const handleSubmitBackgroundTask = useCallback(
    ({
      goal,
      preferences,
      linkedKeys,
    }: {
      goal: string
      preferences: string
      linkedKeys: string[]
    }) => {
      const g = goal.trim()
      if (!g) return
      const tail = [
        preferences.trim() ? `Prefs: ${preferences.trim()}` : "",
        linkedKeys.length > 0 ? `Refs: ${linkedKeys.join(", ")}` : "",
      ]
        .filter(Boolean)
        .join(" ")
      const msg = `[Background task] ${g} ${tail}`.trim()
      realAgent.sendMessage(msg, {
        displayTopic: g,
        user_model_preference: preferences.trim() || undefined,
        begin_background_intake: true,
        force_orchestrator: true,
        persist_linked_datasets: linkedKeys.length > 0 ? linkedKeys : undefined,
      })
    },
    [realAgent],
  )

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

  const layoutStyle = { "--sidebar-width": `${sidebarWidth}px` } as CSSProperties
  const hideAppSidebar = activeTab === "broker_demo"

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 w-full z-50 border-b border-border/60 bg-background/90 backdrop-blur-xl shadow-[0_1px_0_hsl(var(--border)/0.35),0_12px_40px_-12px_rgba(18,86,210,0.07)] flex items-center justify-between px-6 h-14">
          <div className="flex min-w-0 items-center gap-2 sm:gap-4">
            <div className="flex items-center gap-2.5 font-headline text-xl font-bold tracking-tight text-foreground">
              <img src="/jubilee-logo.svg" alt="" className="h-8 w-8 shrink-0" width={32} height={32} />
              Jubilee
            </div>
            {hideAppSidebar ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 text-muted-foreground hover:text-foreground"
                onClick={() => setActiveTab("experiment_lab")}
              >
                Workspace
              </Button>
            ) : null}
          </div>

          <div className="flex items-center gap-1.5">
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground">
              <Search className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground">
              <Bell className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleResetWorkspace}
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

        {/* Main Layout — min-h-0 so inner chat can scroll instead of growing the page */}
        <div className="flex flex-1 min-h-0 overflow-hidden pt-14" style={layoutStyle}>
          {/* Fixed Sidebar — hidden on broker demo (full-width placement UI) */}
          {!hideAppSidebar ? (
            <aside className="hidden md:flex w-[var(--sidebar-width)] fixed left-0 top-14 bottom-0 flex-col z-40">
              <AppSidebar
                activeTab={activeTab}
                onTabChange={setActiveTab}
                activeExperimentId={realAgent.experimentId}
                onSelectExperiment={handleSelectExperiment}
                onStartBlankChat={handleStartBlankChat}
                isBackendConnected={realAgent.isBackendConnected}
                switchingTo={switchingTo}
                onActiveExperimentDeleted={() => {
                  setExperimentDetailQueryId(null)
                  realAgent.leaveLabSession()
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
          ) : null}

          {/* Main Content */}
          <main
            className={cn(
              "flex-1 min-h-0 flex flex-col overflow-hidden",
              !hideAppSidebar && "md:ml-[var(--sidebar-width)]",
            )}
          >
            {/* All tabs stay mounted; inactive ones are hidden via CSS to preserve state and avoid refetches */}
            <div className={cn("flex flex-1 min-h-0 flex-col overflow-hidden relative", activeTab !== "experiment_lab" && "hidden")}>
                {realAgent.experimentId ? (
                  <div className="flex shrink-0 items-center border-b border-border/40 bg-background/95 px-2 py-1.5 z-20">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="gap-1.5 h-8 text-muted-foreground hover:text-foreground"
                      onClick={() => handleLeaveLabSession()}
                    >
                      <ArrowLeft className="h-4 w-4" />
                      Back
                    </Button>
                  </div>
                ) : null}
                <div className="relative flex flex-1 min-h-0 flex-col overflow-hidden">
                  <ChatPanel
                    key={realAgent.experimentId ?? experimentDetailQueryId ?? "draft"}
                    messages={agent.messages}
                    confirmationRequest={agent.confirmationRequest}
                    isRunning={agent.isRunning}
                    backgroundIntakeActive={realAgent.backgroundIntakeActive}
                    onSendMessage={agent.sendMessage}
                    onConfirmation={agent.handleConfirmation as (action: ConfirmationAction, comment?: string) => void}
                    linkedDatasets={realAgent.linkedDatasets}
                    onLinkedDatasetsChange={realAgent.updateLinkedDatasets}
                    datasets={realAgent.datasets}
                    highlightedMessageId={highlightedMessageId}
                    onClearHighlight={handleClearHighlight}
                    onViewReport={() => setReportView({ kind: "chat" })}
                    onViewModelInRegistry={
                      trainedModelNameForCurrentExperiment
                        ? handleOpenTrainedModelForExperiment
                        : undefined
                    }
                    agentState={agent.agentState}
                    steps={agent.steps}
                    experimentId={realAgent.experimentId}
                    runningStepHint={realAgent.runningStepHint}
                    hideComposer={agent.agentState.lab_mode === "task"}
                    startingHandsOffTask={realAgent.startingHandsOffTask}
                    onApproveTrainingPlanGuided={handleApproveTrainingPlanGuided}
                    onApproveTrainingPlanBackground={handleApproveTrainingPlanBackground}
                    onSubmitBackgroundTask={handleSubmitBackgroundTask}
                  />
                </div>
            </div>
            <div className={cn("flex-1 overflow-auto", activeTab !== "broker_demo" && "hidden")}>
              <InsuranceBrokerDemoPage />
            </div>
            <div className={cn("flex-1 overflow-auto", activeTab !== "datasets" && "hidden")}>
              <DatasetsPage
                enabled={activeTab === "datasets" && realAgent.isBackendConnected}
                onDatasetsChanged={() => {
                  void realAgent.refreshDatasets()
                }}
              />
            </div>
            <div className={cn("flex-1 overflow-auto", activeTab !== "models" && "hidden")}>
              {selectedModelRiskName ? (
                <ModelRiskProfilePage
                  modelName={selectedModelRiskName}
                  onBack={() => setSelectedModelRiskName(null)}
                />
              ) : (
                <ModelsPage
                  trainedModels={realAgent.trainedModels}
                  loading={realAgent.trainedModelsLoading}
                  onOpenExperiment={handleOpenExperimentFromModels}
                  onViewReport={(modelName) => setReportView({ kind: "registry", modelName })}
                  scrollToModelName={modelsScrollToModelName}
                  onScrollToModelConsumed={() => setModelsScrollToModelName(null)}
                  riskInventoryEnabled={activeTab === "models"}
                  onOpenModelRisk={(name) => setSelectedModelRiskName(name)}
                />
              )}
            </div>
            {/* Settings tab (commented out)
            <div className={cn("flex-1 overflow-auto", activeTab !== "settings" && "hidden")}>
              <div className="max-w-5xl mx-auto px-8 py-10">
                  <span className="text-overline font-bold text-muted-foreground tracking-widest uppercase">Configuration</span>
                  <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">Settings & API</h1>
                  <p className="mt-3 text-muted-foreground text-sm leading-relaxed max-w-lg">Account management, API keys, and MCP access configuration.</p>
                  <div className="mt-10 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                    Coming soon.
                  </div>
              </div>
            </div>
            */}
          </main>
        </div>

        {reportView?.kind === "chat" && (
          <FinalReport
            agentState={agent.agentState}
            steps={agent.steps}
            onClose={() => setReportView(null)}
            datasets={realAgent.datasets}
          />
        )}
        {reportView?.kind === "registry" && (
          <RegistryFinalReport
            modelName={reportView.modelName}
            onClose={() => setReportView(null)}
          />
        )}
      </div>
    </TooltipProvider>
  )
}
