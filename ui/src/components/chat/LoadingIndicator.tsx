const BAR_COUNT = 5
const BAR_DELAYS = [0, 120, 240, 100, 200]

interface LoadingIndicatorProps {
  /** Shown next to the wave bars (current pipeline activity) */
  label?: string | null
}

export function LoadingIndicator({ label }: LoadingIndicatorProps) {
  const statusLabel = label?.trim() || "Working"

  return (
    <div className="flex items-center gap-3 py-3 px-1 min-h-10 animate-message-in">
      <div className="flex items-end gap-[3px] h-8 shrink-0">
        {Array.from({ length: BAR_COUNT }, (_, i) => (
          <div
            key={i}
            className="w-[3px] rounded-full bg-primary/40 animate-wave"
            style={{ animationDelay: `${BAR_DELAYS[i]}ms`, height: "14px" }}
          />
        ))}
      </div>
      <p className="text-ui text-muted-foreground leading-snug flex-1">{statusLabel}</p>
    </div>
  )
}
