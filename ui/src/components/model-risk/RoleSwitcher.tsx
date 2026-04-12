import { cn } from "@/lib/utils"
import type { ModelRiskPortalRole } from "@/lib/api"

const ROLES: { id: ModelRiskPortalRole; label: string }[] = [
  { id: "owner", label: "Owner" },
  { id: "validation", label: "Validation" },
  { id: "board", label: "Board" },
]

interface RoleSwitcherProps {
  value: ModelRiskPortalRole
  onChange: (role: ModelRiskPortalRole) => void
  className?: string
  disabled?: boolean
}

export function RoleSwitcher({ value, onChange, className, disabled }: RoleSwitcherProps) {
  return (
    <div
      className={cn(
        "inline-flex h-9 items-center rounded-lg border border-border bg-muted/60 p-0.5 text-muted-foreground",
        className,
      )}
      role="tablist"
      aria-label="Portal role"
    >
      {ROLES.map((r) => (
        <button
          key={r.id}
          type="button"
          role="tab"
          aria-selected={value === r.id}
          disabled={disabled}
          onClick={() => onChange(r.id)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            value === r.id
              ? "bg-background text-foreground shadow-sm"
              : "hover:text-foreground",
            disabled && "opacity-50 pointer-events-none",
          )}
        >
          {r.label}
        </button>
      ))}
    </div>
  )
}
