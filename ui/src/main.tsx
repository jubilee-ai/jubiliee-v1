import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ClerkProvider } from '@clerk/react'

const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY
if (!publishableKey) {
  throw new Error('Missing VITE_CLERK_PUBLISHABLE_KEY')
}

/** Matches Jubilee dark palette so Clerk surfaces (UserButton, UserProfile, etc.) aren’t default light theme. */
const clerkAppearance = {
  theme: 'simple' as const,
  variables: {
    colorPrimary: 'hsl(229 55% 78%)',
    colorPrimaryForeground: 'hsl(240 27% 14%)',
    colorForeground: 'hsl(220 9% 91%)',
    colorMutedForeground: 'hsl(220 6% 55%)',
    colorBackground: 'hsl(240 4% 11%)',
    colorInput: 'hsl(240 4% 13%)',
    colorInputForeground: 'hsl(220 9% 91%)',
    colorNeutral: 'hsl(240 4% 20%)',
    colorBorder: 'hsl(240 4% 20%)',
    colorRing: 'hsl(229 55% 78%)',
    colorDanger: 'hsl(352 55% 65%)',
    colorSuccess: 'hsl(130 30% 55%)',
    colorWarning: 'hsl(42 96% 58%)',
    colorShadow: 'rgba(0, 0, 0, 0.45)',
    colorModalBackdrop: 'rgba(8, 8, 12, 0.72)',
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    fontFamilyButtons: 'Inter, system-ui, -apple-system, sans-serif',
    borderRadius: '0.75rem',
  },
  elements: {
    userButtonPopoverCard:
      'rounded-xl border border-border bg-card text-card-foreground shadow-2xl',
    userButtonPopoverMain: 'bg-transparent',
    userButtonPopoverActionButton:
      'rounded-lg text-foreground hover:bg-accent hover:text-accent-foreground',
    userButtonPopoverActionButtonText: 'text-foreground',
    userButtonPopoverFooter:
      'border-t border-border bg-muted/50 text-muted-foreground',
  },
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ClerkProvider publishableKey={publishableKey} appearance={clerkAppearance}>
      <App />
    </ClerkProvider>
  </StrictMode>,
)
