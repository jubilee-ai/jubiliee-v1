import React, { useEffect, useMemo, useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Send, Database, Cpu, X, ShieldCheck, ChevronRight, Download } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Dataset } from "@/types/agent"
import type { Dataset as ApiDataset, ModelType } from "@/lib/api"

interface ChatInputProps {
  draft: string
  onDraftChange: (value: string) => void
  onSend: () => void
  isDisabled: boolean
  placeholder: string
  selectedDatasets: string[]
  onDatasetSelect: (dataset: Dataset) => void
  onDatasetRemove: (datasetFile: string) => void
  selectedModel: string | null
  onModelSelect: (modelId: string) => void
  onModelRemove: () => void
  availableDatasets: ApiDataset[]
  availableModels: ModelType[]
  useHitl: boolean
  onToggleHitl: () => void
  experimentId?: string | null
}

export function ChatInput({
  draft,
  onDraftChange,
  onSend,
  isDisabled,
  placeholder,
  selectedDatasets,
  onDatasetSelect,
  onDatasetRemove,
  selectedModel,
  onModelSelect,
  onModelRemove,
  availableDatasets,
  availableModels,
  useHitl,
  onToggleHitl,
  experimentId,
}: ChatInputProps) {
  const [showDatasetPicker, setShowDatasetPicker] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const [artifacts, setArtifacts] = useState<{ datasets: Array<{ id?: string; ref?: string; name?: string; role?: string; rows?: number }>; models: Array<{ name: string; metrics?: { accuracy?: number } }> }>({ datasets: [], models: [] })
  const [showArtifacts, setShowArtifacts] = useState(false)

  useEffect(() => {
    if (experimentId) {
      fetch(`/api/experiments/${experimentId}/artifacts`)
        .then(r => r.json())
        .then(setArtifacts)
        .catch(() => {})
    } else {
      setArtifacts({ datasets: [], models: [] })
    }
  }, [experimentId])

  const datasetChipLabel = useMemo(() => {
    const lookup = new Map<string, string>()
    for (const d of availableDatasets) {
      if (d.file) lookup.set(d.file, d.name)
      if (d.id) lookup.set(d.id, d.name)
      if (d.name) lookup.set(d.name, d.name)
    }
    return (key: string) => lookup.get(key) ?? key.split("/").pop() ?? key
  }, [availableDatasets])

  const modelChipLabel = useMemo(() => {
    if (!selectedModel) return null
    const m = availableModels.find((x) => x.id === selectedModel)
    return m?.name ?? selectedModel
  }, [selectedModel, availableModels])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      onSend()
    }
  }

  const handleDatasetSelect = (dataset: ApiDataset) => {
    onDatasetSelect(dataset as Dataset)
    setShowDatasetPicker(false)
  }

  const handleModelSelect = (modelId: string) => {
    onModelSelect(modelId)
    setShowModelPicker(false)
  }

  return (
    <div className="flex-shrink-0 z-10 border-t border-border/20 bg-background/95 backdrop-blur-md px-5 pt-3 pb-5 max-w-3xl mx-auto w-full supports-[backdrop-filter]:bg-background/80">
      <div className="rounded-2xl bg-card/95 backdrop-blur-sm border border-border/25 shadow-[0_2px_20px_-4px_rgba(0,0,0,0.06),0_0_0_1px_hsl(var(--border)/0.25)] transition-shadow duration-300 ease-out focus-within:border-border/40 focus-within:shadow-[0_0_0_1px_hsl(var(--primary)/0.18),0_0_24px_-4px_hsl(var(--primary)/0.14),0_12px_40px_-12px_rgba(0,0,0,0.08)]">
        {/* Linked datasets + model: pinned above the reply field for the session */}
        {(selectedDatasets.length > 0 || selectedModel) && (
          <div className="flex items-center gap-2 px-4 py-2 flex-wrap border-b border-border/20 bg-muted/5">
            {selectedDatasets.filter(Boolean).map((ds) => (
              <Badge 
                key={ds} 
                variant="secondary" 
                className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2"
              >
                <Database className="h-3 w-3 text-muted-foreground" />
                {datasetChipLabel(ds)}
                <button
                  type="button"
                  onClick={() => onDatasetRemove(ds)}
                  className="ml-0.5 hover:text-foreground text-muted-foreground"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
            {selectedModel && modelChipLabel && (
              <Badge 
                variant="secondary" 
                className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2"
              >
                <Cpu className="h-3 w-3 text-muted-foreground" />
                {modelChipLabel}
                <button
                  type="button"
                  onClick={onModelRemove}
                  className="ml-0.5 hover:text-foreground text-muted-foreground"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            )}
          </div>
        )}

        <Textarea
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="composer-textarea min-h-[60px] max-h-[180px] resize-none border-0 shadow-none px-4 py-3 text-[14px] leading-6 placeholder:text-muted-foreground/40 bg-transparent"
          disabled={isDisabled}
        />

        {/* Actions bar */}
        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-1">
            <Dialog open={showDatasetPicker} onOpenChange={setShowDatasetPicker}>
              <DialogTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                >
                  <Database className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle className="text-lg font-medium">Select Dataset</DialogTitle>
                </DialogHeader>
                <div className="space-y-1 max-h-[350px] overflow-y-auto -mx-2">
                  {availableDatasets.filter((ds) => ds.trainable !== false).map((ds) => (
                    <button
                      key={ds.id || ds.file || ds.name}
                      className="w-full text-left px-4 py-3 rounded-xl hover:bg-muted/60 transition-colors"
                      onClick={() => handleDatasetSelect(ds)}
                    >
                      <div className="font-medium">{ds.name}</div>
                      <div className="text-sm text-muted-foreground mt-0.5">
                        {ds.rows?.toLocaleString()} rows
                      </div>
                    </button>
                  ))}

                  {experimentId && (artifacts.datasets.length > 0 || artifacts.models.length > 0) && (
                    <div className="border-t border-border/30 pt-2 mt-2 mx-2">
                      <button
                        onClick={() => setShowArtifacts(!showArtifacts)}
                        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                      >
                        <ChevronRight className={cn("h-3 w-3 transition-transform", showArtifacts && "rotate-90")} />
                        Experiment artifacts ({artifacts.datasets.length} datasets, {artifacts.models.length} models)
                      </button>
                      {showArtifacts && (
                        <div className="mt-1 space-y-1">
                          {artifacts.datasets.map(d => (
                            <button
                              key={d.ref || d.id}
                              className="w-full text-left px-4 py-2.5 rounded-xl hover:bg-muted/60 transition-colors"
                              onClick={() => handleDatasetSelect({ file: d.ref || d.id || "", name: d.name, rows: d.rows } as ApiDataset)}
                            >
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-sm">{d.name || d.ref}</span>
                                {d.role && (
                                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                                    {d.role}
                                  </span>
                                )}
                              </div>
                              {d.rows != null && (
                                <div className="text-xs text-muted-foreground mt-0.5">
                                  {d.rows.toLocaleString()} rows
                                </div>
                              )}
                            </button>
                          ))}

                          {artifacts.models.map(m => (
                            <div key={m.name} className="flex items-center justify-between px-4 py-2 text-sm">
                              <span className="text-foreground">
                                {m.name} — {m.metrics?.accuracy ? `acc: ${(m.metrics.accuracy * 100).toFixed(1)}%` : "No metrics"}
                              </span>
                              <a
                                href={`/api/trained-models/${m.name}/download`}
                                className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                              >
                                <Download className="h-3 w-3" />
                                Download
                              </a>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </DialogContent>
            </Dialog>

            <Dialog open={showModelPicker} onOpenChange={setShowModelPicker}>
              <DialogTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                >
                  <Cpu className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle className="text-lg font-medium">Select Model</DialogTitle>
                </DialogHeader>
                <div className="space-y-1 -mx-2">
                  {availableModels.map((model) => (
                    <button
                      key={model.id}
                      className="w-full text-left px-4 py-3 rounded-xl hover:bg-muted/60 transition-colors"
                      onClick={() => handleModelSelect(model.id)}
                    >
                      <div className="font-medium">{model.name}</div>
                      <div className="text-sm text-muted-foreground mt-0.5">
                        {model.description}
                      </div>
                    </button>
                  ))}
                </div>
              </DialogContent>
            </Dialog>

            <div className="h-5 w-px bg-muted mx-1" />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={useHitl ? "default" : "ghost"}
                  size="sm"
                  className={`h-8 w-8 p-0 ${useHitl ? "" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={onToggleHitl}
                >
                  <ShieldCheck className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p>{useHitl ? "Human review (on)" : "Human review (off)"}</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <Button
            size="sm"
            className="h-8 px-3 rounded-md"
            onClick={onSend}
            disabled={isDisabled || !draft.trim()}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
