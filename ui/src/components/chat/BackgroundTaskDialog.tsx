import { useEffect, useMemo, useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Moon, Database } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Dataset as ApiDataset } from "@/lib/api"

export interface BackgroundTaskPayload {
  goal: string
  preferences: string
  linkedKeys: string[]
}

export interface BackgroundTaskDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  datasets: ApiDataset[]
  isSubmitting?: boolean
  onSubmit: (payload: BackgroundTaskPayload) => void
}

function datasetKey(d: ApiDataset): string {
  return d.file || d.id || d.name || ""
}

export function BackgroundTaskDialog({
  open,
  onOpenChange,
  datasets,
  isSubmitting = false,
  onSubmit,
}: BackgroundTaskDialogProps) {
  const [goal, setGoal] = useState("")
  const [preferences, setPreferences] = useState("")
  const [selected, setSelected] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!open) {
      setGoal("")
      setPreferences("")
      setSelected(new Set())
    }
  }, [open])

  const trainable = useMemo(
    () => datasets.filter((d) => d.trainable !== false && datasetKey(d)),
    [datasets],
  )

  const toggleDataset = (key: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const handleSubmit = () => {
    const g = goal.trim()
    if (!g || isSubmitting) return
    onSubmit({
      goal: g,
      preferences: preferences.trim(),
      linkedKeys: [...selected],
    })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-2">
          <div className="flex items-center gap-2 text-primary">
            <Moon className="h-5 w-5" />
            <DialogTitle className="text-lg font-semibold tracking-tight">Assign a task</DialogTitle>
          </div>
          <DialogDescription className="text-sm text-muted-foreground leading-snug pt-1">
            What should Jubilee train? You’ll confirm the plan, then run in the background if you want.
          </DialogDescription>
        </DialogHeader>

        <div className="px-6 py-4 space-y-4 max-h-[min(420px,70vh)] overflow-y-auto">
          <div className="space-y-2">
            <label htmlFor="bg-goal" className="text-sm font-medium">
              Goal
            </label>
            <Textarea
              id="bg-goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Predict churn from subscription data."
              className="min-h-[88px] resize-y"
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="bg-prefs" className="text-sm font-medium">
              Preferences (optional)
            </label>
            <Input
              id="bg-prefs"
              value={preferences}
              onChange={(e) => setPreferences(e.target.value)}
              placeholder="e.g. Fast, interpretable model."
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <span className="text-sm font-medium flex items-center gap-2">
              <Database className="h-3.5 w-3.5 text-muted-foreground" />
              Datasets (optional)
            </span>
            <ScrollArea className="h-[min(200px,28vh)] rounded-lg border border-border/40">
              <div className="p-2 space-y-0.5">
                {trainable.length === 0 ? (
                  <p className="text-xs text-muted-foreground px-2 py-4 text-center">No datasets in catalog yet.</p>
                ) : (
                  trainable.map((d) => {
                    const key = datasetKey(d)
                    const active = selected.has(key)
                    return (
                      <button
                        key={key}
                        type="button"
                        disabled={isSubmitting}
                        onClick={() => toggleDataset(key)}
                        className={cn(
                          "w-full text-left rounded-lg px-3 py-2 text-sm transition-colors",
                          active ? "bg-primary/10 text-foreground" : "hover:bg-muted/60 text-muted-foreground",
                        )}
                      >
                        <span className="font-medium text-foreground">{d.name}</span>
                        {d.rows != null && (
                          <span className="block text-xs text-muted-foreground mt-0.5">
                            {d.rows.toLocaleString()} rows
                          </span>
                        )}
                      </button>
                    )
                  })
                )}
              </div>
            </ScrollArea>
          </div>
        </div>

        <DialogFooter className="px-6 py-4 border-t border-border/30 bg-muted/20">
          <Button type="button" variant="ghost" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            type="button"
            onClick={handleSubmit}
            disabled={!goal.trim() || isSubmitting}
            className="gap-2"
          >
            <Moon className="h-3.5 w-3.5" />
            Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
