import { useState, useEffect, useRef, useCallback } from "react"
import { useRealAgent } from "@/hooks/useRealAgent"
import { useProjects } from "@/hooks/useProjects"
import { useDeployments } from "@/hooks/useDeployments"
import { ProjectSidebar } from "@/components/ProjectSidebar"
import { ChatPanel, ChatPanelRef } from "@/components/ChatPanel"
import { FinalReport } from "@/components/FinalReport"
import { Button } from "@/components/ui/button"
import { TooltipProvider } from "@/components/ui/tooltip"
import { Sparkles, RotateCcw, FileText, FolderOpen } from "lucide-react"
import type { ConfirmationAction } from "@/types/agent"
import type { Deployment } from "@/types/deployment"

export default function App() {
  const [showReport, setShowReport] = useState(false)
  const [highlightedMessageId, setHighlightedMessageId] = useState<string | null>(null)
  const chatPanelRef = useRef<ChatPanelRef>(null)
  
  const realAgent = useRealAgent()
  const projectManager = useProjects()
  const deploymentManager = useDeployments(projectManager.projects, realAgent.isBackendConnected)

  const agent = {
    agentState: realAgent.agentState,
    steps: realAgent.steps,
    messages: realAgent.messages,
    isRunning: realAgent.isRunning,
    currentStepId: realAgent.steps.find(s => s.status === "running" || s.status === "awaiting_confirmation")?.id || null,
    confirmationRequest: realAgent.confirmationRequest,
    handleConfirmation: realAgent.handleConfirmation,
    sendMessage: realAgent.sendMessage,
    reset: realAgent.reset,
  }

  const isComplete = agent.agentState.training_metrics?.success

  // Sync agent state into the active project whenever it changes
  useEffect(() => {
    if (projectManager.activeProjectId && (agent.agentState.goal || agent.isRunning)) {
      projectManager.updateProjectFromState(agent.agentState, agent.isRunning)
    }
  }, [
    agent.agentState.goal,
    agent.agentState.selected_model,
    agent.agentState.training_metrics,
    agent.agentState.error,
    agent.agentState.report_path,
    agent.isRunning,
  ])

  useEffect(() => {
    realAgent.checkConnection()
    const interval = setInterval(() => realAgent.checkConnection(), 10000)
    return () => clearInterval(interval)
  }, [realAgent.checkConnection])

  useEffect(() => {
    if (agent.agentState.training_metrics?.model_name || agent.agentState.report_path) {
      deploymentManager.refresh()
    }
  }, [agent.agentState.training_metrics?.model_name, agent.agentState.report_path, deploymentManager.refresh])

  const handleClearHighlight = useCallback(() => {
    setHighlightedMessageId(null)
  }, [])

  const handleStartAgent = useCallback(async (goal: string, datasets?: string[], modelPreference?: string, hitl?: boolean) => {
    if (projectManager.activeProjectId) {
      projectManager.updateProjectSession(projectManager.activeProjectId, realAgent.getSessionSnapshot())
    }
    projectManager.createProject(goal, datasets)
    await realAgent.startAgent(goal, datasets, modelPreference, hitl)
  }, [
    projectManager.activeProjectId,
    projectManager.updateProjectSession,
    realAgent.getSessionSnapshot,
    projectManager.createProject,
    realAgent.startAgent,
  ])

  const handleNewProject = useCallback(() => {
    if (projectManager.activeProjectId) {
      projectManager.updateProjectSession(projectManager.activeProjectId, realAgent.getSessionSnapshot())
    }
    projectManager.createEmptyProject()
    agent.reset()
  }, [
    projectManager.activeProjectId,
    projectManager.updateProjectSession,
    realAgent.getSessionSnapshot,
    projectManager.createEmptyProject,
    agent.reset,
  ])

  const handleSelectProject = useCallback((id: string) => {
    if (id === projectManager.activeProjectId) return

    if (projectManager.activeProjectId) {
      projectManager.updateProjectSession(projectManager.activeProjectId, realAgent.getSessionSnapshot())
    }

    const nextProject = projectManager.getProjectById(id)
    projectManager.selectProject(id)

    if (nextProject?.session) {
      realAgent.loadSessionSnapshot(nextProject.session)
    } else {
      realAgent.reset()
    }
  }, [
    projectManager.activeProjectId,
    projectManager.updateProjectSession,
    realAgent.getSessionSnapshot,
    projectManager.getProjectById,
    projectManager.selectProject,
    realAgent.loadSessionSnapshot,
    realAgent.reset,
  ])

  useEffect(() => {
    if (!projectManager.activeProjectId) return
    projectManager.updateProjectSession(
      projectManager.activeProjectId,
      realAgent.getSessionSnapshot()
    )
  }, [
    projectManager.activeProjectId,
    projectManager.updateProjectSession,
    realAgent.getSessionSnapshot,
    realAgent.agentState,
    realAgent.steps,
    realAgent.messages,
    realAgent.progress,
  ])

  return (
    <TooltipProvider>
      <div className="h-screen bg-background flex flex-col overflow-hidden">
        {/* Header */}
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
            {/* Left Panel - Projects */}
            <div className="w-[280px] flex-shrink-0 hidden lg:block h-full overflow-auto border-r border-border/50">
              <ProjectSidebar
                projects={projectManager.projects}
                deployments={deploymentManager.deployments}
                activeProjectId={projectManager.activeProjectId}
                onSelectProject={handleSelectProject}
                onDeleteProject={projectManager.deleteProject}
                onNewProject={handleNewProject}
                onRefreshDeployments={deploymentManager.refresh}
                isDeploymentsLoading={deploymentManager.isLoading}
                deploymentsError={deploymentManager.error}
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
                onStartAgent={handleStartAgent}
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

        {/* Mobile Projects Toggle */}
        <div className="lg:hidden fixed bottom-4 left-4 z-50">
          <MobileProjectsButton
            projects={projectManager.projects}
            deployments={deploymentManager.deployments}
            activeProjectId={projectManager.activeProjectId}
            onSelectProject={handleSelectProject}
            onDeleteProject={projectManager.deleteProject}
            onNewProject={handleNewProject}
            onRefreshDeployments={deploymentManager.refresh}
            isDeploymentsLoading={deploymentManager.isLoading}
            deploymentsError={deploymentManager.error}
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

// Mobile projects button component
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import type { Project } from "@/types/project"

function MobileProjectsButton({
  projects,
  deployments,
  activeProjectId,
  onSelectProject,
  onDeleteProject,
  onNewProject,
  onRefreshDeployments,
  isDeploymentsLoading,
  deploymentsError,
}: {
  projects: Project[]
  deployments: Deployment[]
  activeProjectId: string | null
  onSelectProject: (id: string) => void
  onDeleteProject: (id: string) => void
  onNewProject: () => void
  onRefreshDeployments: () => void
  isDeploymentsLoading: boolean
  deploymentsError: string | null
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="default" size="lg" className="rounded-full shadow-lg">
          <FolderOpen className="h-5 w-5 mr-2" />
          {projects.length} Projects
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[90vw] max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Projects</DialogTitle>
        </DialogHeader>
        <div className="flex-1 overflow-auto">
          <ProjectSidebar
            projects={projects}
            deployments={deployments}
            activeProjectId={activeProjectId}
            onSelectProject={onSelectProject}
            onDeleteProject={onDeleteProject}
            onNewProject={onNewProject}
            onRefreshDeployments={onRefreshDeployments}
            isDeploymentsLoading={isDeploymentsLoading}
            deploymentsError={deploymentsError}
          />
        </div>
      </DialogContent>
    </Dialog>
  )
}
