import { SuggestionChip } from "./SuggestionChip"

interface EmptyStateProps {
  onSelectSuggestion: (text: string) => void
}

export function EmptyState({ onSelectSuggestion }: EmptyStateProps) {
  return (
    <div className="text-center py-16 relative">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden>
        <div className="w-72 h-72 bg-primary/[0.03] rounded-full blur-3xl" />
      </div>
      <div className="relative">
        <span className="text-[10px] font-bold text-muted-foreground/50 tracking-[0.2em] uppercase block mb-3">Experimental Lab</span>
        <h2 className="font-headline text-2xl font-semibold tracking-tight mb-3">How can I help?</h2>
        <p className="text-muted-foreground text-sm mb-10 leading-relaxed max-w-md mx-auto">
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
    </div>
  )
}
