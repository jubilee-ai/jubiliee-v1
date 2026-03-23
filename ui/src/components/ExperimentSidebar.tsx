import { useState, useEffect, useCallback, useRef } from "react"
import { Plus, MessageSquare, Loader2, CheckCircle2, AlertCircle, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  listExperiments,
  createExperiment,
  deleteExperiment,
  type ExperimentSummary,
} from "@/lib/api"
import { cn } from "@/lib/utils"

interface ExperimentSidebarProps {
  activeExperimentId: string | null
  onSelectExperiment: (id: string) => void
  onNewExperiment: () => void
  isBackendConnected: boolean
  switchingTo?: string | null
}

function statusIcon(status: string) {
  switch (status) {
    case "running":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-[hsl(var(--step-active))]" />
    case "completed":
      return <CheckCircle2 className="h-3.5 w-3.5 text-[hsl(var(--step-complete))]" />
    case "failed":
      return <AlertCircle className="h-3.5 w-3.5 text-[hsl(var(--step-error))]" />
    default:
      return <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
  }
}

function relativeTime(dateStr: string | null | undefined): string {
  if (!dateStr) return ""
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function ExperimentSidebar({
  activeExperimentId,
  onSelectExperiment,
  onNewExperiment,
  isBackendConnected,
  switchingTo,
}: ExperimentSidebarProps) {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const cacheRef = useRef<ExperimentSummary[]>([])

  const refresh = useCallback(async () => {
    if (!isBackendConnected) return
    if (cacheRef.current.length === 0) setIsLoading(true)
    try {
      const data = await listExperiments()
      cacheRef.current = data
      setExperiments(data)
    } catch {
      // ignore
    } finally {
      setIsLoading(false)
    }
  }, [isBackendConnected])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleNew = async () => {
    try {
      const exp = await createExperiment()
      setExperiments((prev) => [exp, ...prev])
      onSelectExperiment(exp.id)
    } catch {
      onNewExperiment()
    }
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    setExperiments((prev) => prev.filter((ex) => ex.id !== id))
    try {
      await deleteExperiment(id)
      if (activeExperimentId === id) {
        onNewExperiment()
      }
    } catch {
      refresh()
    }
  }

  const handleSelect = (id: string) => {
    if (id === activeExperimentId) return
    onSelectExperiment(id)
  }

  return (
    <div className="h-full flex flex-col bg-[hsl(var(--sidebar-bg))] backdrop-blur-sm">
      {/* Header */}
      <div className="flex-shrink-0 p-3 border-b border-border/25">
        <Button
          variant="outline"
          size="sm"
          onClick={handleNew}
          className="w-full justify-start gap-2 h-9 text-sm font-medium bg-transparent border-border/40 hover:bg-muted/50"
          disabled={!isBackendConnected}
        >
          <Plus className="h-4 w-4" />
          New Experiment
        </Button>
      </div>

      {/* Experiment List */}
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-0.5">
          {isLoading && experiments.length === 0 && (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
              Loading...
            </div>
          )}
          {!isLoading && experiments.length === 0 && (
            <div className="text-center py-8 text-muted-foreground text-xs">
              No experiments yet
            </div>
          )}
          {experiments.map((exp) => {
            const isSwitching = switchingTo === exp.id
            return (
            <button
              key={exp.id}
              onClick={() => handleSelect(exp.id)}
              className={cn(
                "w-full text-left px-3 py-2.5 rounded-2xl transition-all duration-150 group relative",
                "hover:bg-muted/50",
                isSwitching && "opacity-70",
                activeExperimentId === exp.id
                  ? "bg-muted/70 border-l-2 border-l-primary"
                  : "border-l-2 border-l-transparent"
              )}
            >
              <div className="flex items-start gap-2">
                <div className="mt-0.5 flex-shrink-0">
                  {isSwitching ? <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" /> : statusIcon(exp.status)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    {exp.name}
                  </div>
                  {exp.last_message && (
                    <div className="text-xs text-muted-foreground truncate mt-0.5">
                      {exp.last_message}
                    </div>
                  )}
                  <div className="text-[10px] text-muted-foreground/60 mt-1">
                    {relativeTime(exp.updated_at || exp.created_at)}
                  </div>
                </div>
                <button
                  onClick={(e) => handleDelete(e, exp.id)}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-destructive/20"
                >
                  <Trash2 className="h-3 w-3 text-muted-foreground" />
                </button>
              </div>
            </button>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
