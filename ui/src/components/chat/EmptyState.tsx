import { SuggestionChip } from "./SuggestionChip"

interface EmptyStateProps {
  onSelectSuggestion: (text: string) => void
}

export function EmptyState({ onSelectSuggestion }: EmptyStateProps) {
  return (
    <div className="text-center py-16">
      <h2 className="text-xl font-medium tracking-tight mb-3">How can I help?</h2>
      <p className="text-muted-foreground mb-8">
        Ask a question, explore data, or attach a dataset to train a model.
      </p>
      <div className="flex flex-wrap gap-3 justify-center">
        <SuggestionChip
          label="What datasets are available?"
          onClick={() => onSelectSuggestion("What datasets are available?")}
        />
        <SuggestionChip
          label="Do I have any trained models?"
          onClick={() => onSelectSuggestion("Do I have any trained models?")}
        />
        <SuggestionChip
          label="Analyze the loan default data"
          onClick={() => onSelectSuggestion("Analyze the loan default dataset and show key statistics")}
        />
        <SuggestionChip
          label="Predict insurance costs"
          onClick={() => onSelectSuggestion("Predict insurance costs for a 35-year-old non-smoker")}
        />
      </div>
    </div>
  )
}
