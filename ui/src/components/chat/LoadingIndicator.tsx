import { Bot } from "lucide-react"

export function LoadingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-muted">
        <Bot className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <div className="flex items-center gap-1.5 py-3 px-1">
        <span 
          className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-pulse" 
          style={{ animationDelay: "0ms" }} 
        />
        <span 
          className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-pulse" 
          style={{ animationDelay: "150ms" }} 
        />
        <span 
          className="w-1.5 h-1.5 rounded-full bg-muted-foreground/40 animate-pulse" 
          style={{ animationDelay: "300ms" }} 
        />
      </div>
    </div>
  )
}
