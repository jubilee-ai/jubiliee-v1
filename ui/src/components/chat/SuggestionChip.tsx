
interface SuggestionChipProps {
  label: string
  onClick: () => void
}

export function SuggestionChip({ label, onClick }: SuggestionChipProps) {
  return (
    <button
      className="px-5 py-2.5 rounded-full border border-primary/20 bg-primary/5 hover:bg-primary/10 text-sm font-medium transition-all hover:border-primary/30"
      onClick={onClick}
    >
      {label}
    </button>
  )
}
