
interface SuggestionChipProps {
  label: string
  onClick: () => void
}

export function SuggestionChip({ label, onClick }: SuggestionChipProps) {
  return (
    <button
      className="px-5 py-2.5 rounded-full border border-border bg-background hover:bg-muted/50 text-sm font-medium transition-all hover:border-muted-foreground/20"
      onClick={onClick}
    >
      {label}
    </button>
  )
}
