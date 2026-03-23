import React, { useState } from "react"
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
import { Send, Database, Cpu, X, ShieldCheck } from "lucide-react"
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
}: ChatInputProps) {
  const [showDatasetPicker, setShowDatasetPicker] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(false)

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
    <div className="flex-shrink-0 px-6 py-4 max-w-3xl mx-auto w-full">
      <div className="rounded-2xl border border-border/25 bg-card shadow-sm/50">
        {/* Selected items */}
        {(selectedDatasets.length > 0 || selectedModel) && (
          <div className="flex items-center gap-2 px-4 pt-3 flex-wrap">
            {selectedDatasets.map((ds) => (
              <Badge 
                key={ds} 
                variant="secondary" 
                className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2"
              >
                <Database className="h-3 w-3 text-muted-foreground" />
                {ds.split("/").pop()}
                <button
                  onClick={() => onDatasetRemove(ds)}
                  className="ml-0.5 hover:text-foreground text-muted-foreground"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
            {selectedModel && (
              <Badge 
                variant="secondary" 
                className="gap-1.5 h-7 text-xs font-normal rounded-full pl-3 pr-2"
              >
                <Cpu className="h-3 w-3 text-muted-foreground" />
                {selectedModel}
                <button
                  onClick={onModelRemove}
                  className="ml-0.5 hover:text-foreground text-muted-foreground"
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            )}
          </div>
        )}

        {/* Textarea */}
        <Textarea
          value={draft}
          onChange={(e) => onDraftChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="min-h-[56px] max-h-[200px] resize-none border-0 shadow-none focus-visible:ring-0 px-4 py-3 text-[15px] placeholder:text-muted-foreground/60"
          disabled={isDisabled}
        />

        {/* Actions bar */}
        <div className="flex items-center justify-between px-3 pb-3">
          <div className="flex items-center gap-1">
            <Dialog open={showDatasetPicker} onOpenChange={setShowDatasetPicker}>
              <DialogTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-8 px-3 text-muted-foreground hover:text-foreground"
                >
                  <Database className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle className="text-lg font-medium">Select Dataset</DialogTitle>
                </DialogHeader>
                <div className="space-y-1 max-h-[350px] overflow-y-auto -mx-2">
                  {availableDatasets.map((ds) => (
                    <button
                      key={ds.file}
                      className="w-full text-left px-4 py-3 rounded-xl hover:bg-muted/60 transition-colors"
                      onClick={() => handleDatasetSelect(ds)}
                    >
                      <div className="font-medium">{ds.name}</div>
                      <div className="text-sm text-muted-foreground mt-0.5">
                        {ds.rows?.toLocaleString()} rows
                      </div>
                    </button>
                  ))}
                </div>
              </DialogContent>
            </Dialog>

            <Dialog open={showModelPicker} onOpenChange={setShowModelPicker}>
              <DialogTrigger asChild>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-8 px-3 text-muted-foreground hover:text-foreground"
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

            <div className="h-5 w-px bg-border mx-1" />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={useHitl ? "default" : "ghost"}
                  size="sm"
                  className={`h-8 px-3 ${useHitl ? "" : "text-muted-foreground hover:text-foreground"}`}
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
            className="h-8 px-3 rounded-lg"
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
