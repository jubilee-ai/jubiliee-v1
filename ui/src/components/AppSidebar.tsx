import { useState, useEffect, useCallback } from "react"
import { Plus, MessageSquare, Loader2, CheckCircle2, AlertCircle, Trash2, FlaskConical, Database, Box, Settings, Moon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { ExperimentSummary } from "@/lib/api"
import {
  useExperimentsList,
  useDeleteExperimentMutation,
  useUpdateExperimentMutation,
} from "@/lib/queries"
import { cn } from "@/lib/utils"

function sidebarDebug(event: string, payload?: Record<string, unknown>) {
  const ts = new Date().toISOString()
  if (payload) {
    console.log(`[sidebar:experiment-switch][${ts}] ${event}`, payload)
    return
  }
  console.log(`[sidebar:experiment-switch][${ts}] ${event}`)
}

export type AppTab = "experiment_lab" | "datasets" | "models" | "settings"

interface AppSidebarProps {
  activeTab: AppTab
  onTabChange: (tab: AppTab) => void
  activeExperimentId: string | null
  onSelectExperiment: (id: string) => void | Promise<void>
  /** Clear to a blank draft locally; experiment row is created on first message send. */
  onStartBlankChat: () => void | Promise<void>
  isBackendConnected: boolean
  switchingTo?: string | null
  /** When the user deletes the currently open experiment, return to lab home instead of creating a new one. */
  onActiveExperimentDeleted?: () => void
}

const navItems: { id: AppTab; label: string; icon: React.ReactNode }[] = [
  { id: "experiment_lab", label: "Experiments", icon: <FlaskConical className="h-4 w-4" /> },
  { id: "models", label: "Models", icon: <Box className="h-4 w-4" /> },
  { id: "datasets", label: "Datasets", icon: <Database className="h-4 w-4" /> },
  { id: "settings", label: "Settings", icon: <Settings className="h-4 w-4" /> },
]

function statusIcon(status: string) {
  switch (status) {
    case "running":
      return <Loader2 className="h-3 w-3 animate-spin text-primary" />
    case "completed":
      return <CheckCircle2 className="h-3 w-3 text-[hsl(var(--step-complete))]" />
    case "failed":
      return <AlertCircle className="h-3 w-3 text-destructive" />
    default:
      return <MessageSquare className="h-3 w-3 text-muted-foreground/40" />
  }
}

function experimentRowIcon(exp: ExperimentSummary) {
  const isBackgroundTask = exp.lab_mode === "task" || exp.task_status != null
  if (isBackgroundTask) {
    if (exp.task_status === "running") {
      return <Loader2 className="h-3 w-3 animate-spin text-primary" />
    }
    if (exp.task_status === "completed") {
      return <CheckCircle2 className="h-3 w-3 text-[hsl(var(--step-complete))]" />
    }
    if (exp.task_status === "failed") {
      return <AlertCircle className="h-3 w-3 text-destructive" />
    }
    return <Moon className="h-3 w-3 text-primary/70" aria-hidden />
  }
  return statusIcon(exp.status)
}

function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/__(.+?)__/g, "$1")
    .replace(/_(.+?)_/g, "$1")
    .replace(/`(.+?)`/g, "$1")
    .replace(/\[(.+?)\]\(.+?\)/g, "$1")
    .replace(/#{1,6}\s/g, "")
    .replace(/\n/g, " ")
    .trim()
}

function EditableExperimentTitle({
  experimentId,
  name,
  isActive,
  disabled,
}: {
  experimentId: string
  name: string
  isActive: boolean
  disabled?: boolean
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(name)
  const updateMutation = useUpdateExperimentMutation()

  useEffect(() => {
    setDraft(name)
  }, [name])

  const commit = useCallback(async () => {
    setEditing(false)
    const t = draft.trim()
    if (!t || t === name) {
      setDraft(name)
      return
    }
    try {
      await updateMutation.mutateAsync({ id: experimentId, updates: { name: t } })
    } catch {
      setDraft(name)
    }
  }, [draft, name, experimentId, updateMutation])

  if (editing) {
    return (
      <Input
        autoFocus
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur()
          if (e.key === "Escape") {
            setDraft(name)
            setEditing(false)
          }
        }}
        className="h-7 text-[13px] px-2 py-0"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
      />
    )
  }

  return (
    <span
      className={cn(
        "text-[13px] leading-snug line-clamp-2 break-all overflow-hidden min-w-0 flex-1",
        isActive ? "font-medium text-foreground" : "text-foreground/85",
        !disabled && "cursor-text",
      )}
      title="Double-click to rename"
      onDoubleClick={(e) => {
        e.stopPropagation()
        if (!disabled) setEditing(true)
      }}
    >
      {name}
    </span>
  )
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

export function AppSidebar({
  activeTab,
  onTabChange,
  activeExperimentId,
  onSelectExperiment,
  onStartBlankChat,
  isBackendConnected,
  switchingTo,
  onActiveExperimentDeleted,
}: AppSidebarProps) {
  /** Always fetch the list — do not gate on health ping; that left queries disabled with no network activity when /api/health lagged or failed first. */
  const { data: experiments = [], isLoading, isFetching, isError, error } = useExperimentsList()
  const deleteMutation = useDeleteExperimentMutation()

  useEffect(() => {
    sidebarDebug("list-query-state", {
      isLoading,
      isFetching,
      isError,
      count: experiments.length,
      error: error ? String(error) : null,
      activeExperimentId,
      switchingTo: switchingTo ?? null,
    })
  }, [isLoading, isFetching, isError, error, experiments.length, activeExperimentId, switchingTo])

  const handleNew = () => {
    onTabChange("experiment_lab")
    void onStartBlankChat()
  }

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation()
    sidebarDebug("delete-click", { id, activeExperimentId })
    try {
      await deleteMutation.mutateAsync(id)
      sidebarDebug("delete-success", { id })
      if (activeExperimentId === id) {
        onActiveExperimentDeleted?.()
      }
    } catch {
      sidebarDebug("delete-failed", { id })
      // list refetches on success; errors are rare
    }
  }

  const handleSelect = (id: string) => {
    sidebarDebug("select-click", {
      clickedId: id,
      activeExperimentId,
      switchingTo: switchingTo ?? null,
    })
    onTabChange("experiment_lab")
    if (id === activeExperimentId) {
      sidebarDebug("select-skip-already-active", { clickedId: id })
      return
    }
    sidebarDebug("select-forward-to-parent", { clickedId: id })
    onSelectExperiment(id)
  }

  return (
    <div className="h-full min-w-0 flex flex-col border-r border-border/50 bg-background">
      {/* Section Navigation */}
      <nav className="flex flex-col gap-0.5 px-3 pt-4">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className={cn(
              "flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-colors",
              activeTab === item.id
                ? "text-foreground font-semibold bg-muted"
                : "text-muted-foreground font-medium hover:bg-muted/50 hover:text-foreground"
            )}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Experiments list */}
      <div className="mt-5 px-5 flex items-center justify-between">
        <span className="text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-widest">Recent</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleNew}
          className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
          disabled={!isBackendConnected}
        >
          <Plus className="h-3 w-3" />
        </Button>
      </div>

      <ScrollArea className="flex-1 min-h-0 min-w-0 mt-1.5">
        <div className="px-2 pb-2 space-y-1 min-w-0">
          {isLoading && experiments.length === 0 && (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-xs">
              <Loader2 className="h-3.5 w-3.5 animate-spin mr-2" />
              Loading…
            </div>
          )}
          {!isLoading && experiments.length === 0 && (
            <div className="text-center py-8 text-muted-foreground/50 text-xs">
              No experiments yet
            </div>
          )}
          {experiments.map((exp) => {
            const isSwitching = switchingTo === exp.id
            const isActive = activeTab === "experiment_lab" && activeExperimentId === exp.id
            const preview = exp.last_message ? stripMarkdown(exp.last_message) : null

            return (
              <div
                key={exp.id}
                className={cn(
                  "w-full min-w-0 overflow-hidden rounded-lg border border-transparent transition-colors",
                  "hover:bg-muted/50 hover:border-border/30",
                  isSwitching && "opacity-50 pointer-events-none",
                  isActive && "bg-muted border-border/40"
                )}
              >
                <div className="flex gap-2 min-w-0 items-start p-2.5">
                  <div className="shrink-0 pt-0.5" aria-hidden>
                    {isSwitching ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                    ) : (
                      experimentRowIcon(exp)
                    )}
                  </div>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => handleSelect(exp.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault()
                        handleSelect(exp.id)
                      }
                    }}
                    className="min-w-0 flex-1 text-left rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background cursor-pointer"
                  >
                    <div className="flex items-start justify-between gap-2 min-w-0">
                      <EditableExperimentTitle
                        experimentId={exp.id}
                        name={exp.name}
                        isActive={isActive}
                        disabled={isSwitching || !isBackendConnected}
                      />
                      <span className="text-[10px] text-muted-foreground/50 shrink-0 tabular-nums pt-0.5">
                        {relativeTime(exp.updated_at || exp.created_at)}
                      </span>
                    </div>
                    {preview ? (
                      <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground/70 line-clamp-2 break-all overflow-hidden">
                        {preview}
                      </p>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    aria-label={`Delete ${exp.name}`}
                    className="h-7 w-7 shrink-0 p-0 text-muted-foreground/35 hover:text-destructive hover:bg-destructive/10"
                    onClick={(e) => handleDelete(e, exp.id)}
                    disabled={isSwitching}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
