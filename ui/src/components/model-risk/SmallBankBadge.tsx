import { Building2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface SmallBankBadgeProps {
  profile: string
  className?: string
}

/** Governance / institution profile label (demo: Small bank, Regional bank, …). */
export function SmallBankBadge({ profile, className }: SmallBankBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1 font-normal text-muted-foreground border-border/80 bg-muted/40",
        className,
      )}
    >
      <Building2 className="h-3 w-3 shrink-0" aria-hidden />
      <span>{profile}</span>
    </Badge>
  )
}
