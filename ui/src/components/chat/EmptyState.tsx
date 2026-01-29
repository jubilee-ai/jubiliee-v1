import React from "react"
import { SuggestionChip } from "./SuggestionChip"

interface EmptyStateProps {
  onSelectSuggestion: (text: string) => void
}

export function EmptyState({ onSelectSuggestion }: EmptyStateProps) {
  return (
    <div className="text-center py-16">
      <h2 className="text-xl font-medium tracking-tight mb-3">What would you like to train?</h2>
      <p className="text-muted-foreground mb-8">
        Describe your goal and I'll handle the rest.
      </p>
      <div className="flex flex-wrap gap-3 justify-center">
        <SuggestionChip
          label="Predict loan defaults"
          onClick={() => onSelectSuggestion("Train a model to predict loan defaults")}
        />
        <SuggestionChip
          label="Predict insurance costs"
          onClick={() => onSelectSuggestion("Train a model to predict insurance costs")}
        />
      </div>
    </div>
  )
}
