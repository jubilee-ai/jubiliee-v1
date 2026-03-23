
interface SuggestionChipProps {
  label: string
  onClick: () => void
}

export function SuggestionChip({ label, onClick }: SuggestionChipProps) {
  return (
    <button
      className="px-5 py-2.5 rounded-lg bg-card shadow-[0_2px_8px_rgba(45,51,53,0.06)] hover:shadow-[0_4px_12px_rgba(84,94,131,0.1)] text-sm font-medium transition-all active:scale-[0.98] text-foreground"
      onClick={onClick}
    >
      {label}
    </button>
  )
}
