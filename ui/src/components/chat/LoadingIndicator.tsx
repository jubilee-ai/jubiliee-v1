const BAR_COUNT = 5
const BAR_DELAYS = [0, 120, 240, 100, 200]

export function LoadingIndicator() {
  return (
    <div className="flex items-end gap-[3px] py-3 px-1 h-8 animate-message-in">
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <div
          key={i}
          className="w-[3px] rounded-full bg-primary/40 animate-wave"
          style={{ animationDelay: `${BAR_DELAYS[i]}ms`, height: "14px" }}
        />
      ))}
    </div>
  )
}
