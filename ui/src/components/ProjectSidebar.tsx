import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { Project } from "@/types/project"
import type { Deployment } from "@/types/deployment"
import {
  Plus,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Circle,
  Trash2,
  FolderOpen,
  Rocket,
  RefreshCw,
} from "lucide-react"

interface ProjectSidebarProps {
  projects: Project[]
  deployments: Deployment[]
  activeProjectId: string | null
  onSelectProject: (id: string) => void
  onDeleteProject: (id: string) => void
  onNewProject: () => void
  onRefreshDeployments: () => void
  isDeploymentsLoading?: boolean
  deploymentsError?: string | null
}

function statusIcon(status: Project["status"]) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
    case "running":
      return <Loader2 className="h-3.5 w-3.5 text-blue-500 animate-spin flex-shrink-0" />
    case "error":
      return <AlertCircle className="h-3.5 w-3.5 text-red-500 flex-shrink-0" />
    default:
      return <Circle className="h-3.5 w-3.5 text-muted-foreground/40 flex-shrink-0" />
  }
}

function statusLabel(status: Project["status"]) {
  switch (status) {
    case "completed": return "Completed"
    case "running": return "Running"
    case "error": return "Error"
    default: return "Idle"
  }
}

function formatMetricLine(project: Project): string | null {
  const m = project.metrics
  if (!m) return null

  const parts: string[] = []
  if (m.modelType || m.modelName) {
    parts.push(m.modelName || m.modelType || "")
  }
  if (m.testAccuracy != null) parts.push(`Acc ${(m.testAccuracy * 100).toFixed(1)}%`)
  if (m.testRocAuc != null) parts.push(`AUC ${m.testRocAuc.toFixed(3)}`)
  if (m.testR2 != null) parts.push(`R² ${m.testR2.toFixed(4)}`)
  if (m.testRmse != null) parts.push(`RMSE ${m.testRmse.toFixed(0)}`)
  return parts.length > 0 ? parts.join(" · ") : null
}

function formatDeploymentMetricsLine(deployment: Deployment): string | null {
  const parts = [
    deployment.metrics.testAccuracy != null
      ? `Acc ${(deployment.metrics.testAccuracy * 100).toFixed(1)}%`
      : null,
    deployment.metrics.testRocAuc != null
      ? `AUC ${deployment.metrics.testRocAuc.toFixed(3)}`
      : null,
    deployment.metrics.testR2 != null
      ? `R² ${deployment.metrics.testR2.toFixed(4)}`
      : null,
    deployment.metrics.testRmse != null
      ? `RMSE ${deployment.metrics.testRmse.toFixed(0)}`
      : null,
  ].filter(Boolean)

  return parts.length > 0 ? parts.join(" · ") : null
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return "just now"
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function ProjectSidebar({
  projects,
  deployments,
  activeProjectId,
  onSelectProject,
  onDeleteProject,
  onNewProject,
  onRefreshDeployments,
  isDeploymentsLoading = false,
  deploymentsError = null,
}: ProjectSidebarProps) {
  return (
    <div className="h-full flex flex-col bg-muted/30">
      <Tabs defaultValue="projects" className="h-full flex flex-col">
        <div className="p-3 pb-2">
          <TabsList className="w-full h-9 grid grid-cols-2">
            <TabsTrigger value="projects" className="text-xs">
              Projects
            </TabsTrigger>
            <TabsTrigger value="deployments" className="text-xs">
              Deployments
            </TabsTrigger>
          </TabsList>
        </div>

        <Separator />

        <TabsContent value="projects" className="flex-1 mt-0 data-[state=inactive]:hidden">
          <div className="h-full flex flex-col">
            <div className="p-4 pb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">Projects</span>
                <span className="text-xs text-muted-foreground">
                  {projects.length} total
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={onNewProject}
                className="w-full mt-2 h-8 text-xs"
              >
                <Plus className="h-3.5 w-3.5 mr-1.5" />
                New Project
              </Button>
            </div>

            <Separator />

            <ScrollArea className="flex-1">
              <div className="p-2">
                {projects.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                    <FolderOpen className="h-10 w-10 text-muted-foreground/30 mb-3" />
                    <p className="text-sm text-muted-foreground/60 leading-relaxed">
                      No projects yet. Start a new training pipeline to create one.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    {projects.map((project) => {
                      const isActive = project.id === activeProjectId
                      const metricLine = formatMetricLine(project)

                      return (
                        <div
                          key={project.id}
                          onClick={() => onSelectProject(project.id)}
                          className={`
                            group relative rounded-lg px-3 py-2.5 cursor-pointer transition-colors
                            ${isActive
                              ? "bg-foreground/[0.06] ring-1 ring-foreground/10"
                              : "hover:bg-muted/60"
                            }
                          `}
                        >
                          {/* Top row: status + goal */}
                          <div className="flex items-start gap-2">
                            <div className="mt-0.5">
                              {statusIcon(project.status)}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-[13px] font-medium leading-snug line-clamp-2">
                                {project.goal || "Untitled Project"}
                              </p>
                            </div>
                          </div>

                          {/* Meta row */}
                          <div className="mt-1.5 pl-[22px] flex items-center gap-2 text-[11px] text-muted-foreground">
                            <span>{statusLabel(project.status)}</span>
                            <span className="text-muted-foreground/30">·</span>
                            <span>{timeAgo(project.updatedAt)}</span>
                          </div>

                          {/* Metrics row */}
                          {metricLine && (
                            <div className="mt-1 pl-[22px] text-[11px] text-muted-foreground/80 truncate">
                              {metricLine}
                            </div>
                          )}

                          {/* Datasets */}
                          {project.datasets.length > 0 && (
                            <div className="mt-1 pl-[22px] text-[10px] text-muted-foreground/50 truncate">
                              {project.datasets.join(", ")}
                            </div>
                          )}

                          {/* Delete button */}
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  onDeleteProject(project.id)
                                }}
                                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-destructive/10"
                              >
                                <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent side="left">Delete project</TooltipContent>
                          </Tooltip>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        </TabsContent>

        <TabsContent value="deployments" className="flex-1 mt-0 data-[state=inactive]:hidden">
          <div className="h-full flex flex-col">
            <div className="p-4 pb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">Deployments</span>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {deployments.length} total
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onRefreshDeployments}
                    className="h-7 w-7 p-0"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isDeploymentsLoading ? "animate-spin" : ""}`} />
                  </Button>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground">
                Observability for deployed models.
              </p>
            </div>

            <Separator />

            <ScrollArea className="flex-1">
              <div className="p-2 space-y-2">
                {deploymentsError && (
                  <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-2 text-[11px] text-destructive">
                    {deploymentsError}
                  </div>
                )}
                {deployments.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                    <Rocket className="h-10 w-10 text-muted-foreground/30 mb-3" />
                    <p className="text-sm text-muted-foreground/60 leading-relaxed">
                      No deployed models yet.
                    </p>
                  </div>
                ) : (
                  deployments.map((deployment) => (
                    <div
                      key={deployment.id}
                      className="rounded-lg px-3 py-2.5 border border-border/60 bg-background/40"
                    >
                      <div className="text-[13px] font-medium leading-snug truncate">
                        {deployment.modelName}
                      </div>
                      {deployment.modelType && (
                        <div className="text-[11px] text-muted-foreground mt-0.5 truncate">
                          {deployment.modelType}
                        </div>
                      )}
                      {deployment.description && (
                        <p className="text-[11px] text-muted-foreground/90 mt-1 line-clamp-2">
                          {deployment.description}
                        </p>
                      )}
                      {deployment.datasetUsed.length > 0 && (
                        <div className="mt-1.5 text-[11px] text-muted-foreground">
                          Dataset: {deployment.datasetUsed.join(", ")}
                        </div>
                      )}
                      {formatDeploymentMetricsLine(deployment) && (
                        <div className="mt-1 text-[11px] text-muted-foreground/90">
                          {formatDeploymentMetricsLine(deployment)}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
