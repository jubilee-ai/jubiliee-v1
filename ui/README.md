# Jubilee Training Agent UI

A modern, Cursor-inspired UI for the ML Training Agent. This UI provides a chat-first experience with real-time progress tracking and human-in-the-loop confirmation at each step.

## Features

- **Chat Panel (Right Side)**: Talk to the agent, link datasets and models using `@dataset` and `@model` buttons
- **Progress Panel (Left Side)**: Real-time visualization of pipeline steps with metrics
- **Human-in-the-Loop Confirmation**: At each step, you can:
  - **Accept**: Move to the next step
  - **Redo + Comment**: Redo the step with your feedback
  - **Accept All**: Auto-accept all remaining steps
- **Final Report**: Detailed report with metrics, features, and full audit trace
- **Real Agent Integration**: Connect to the actual Python training agent via FastAPI

## Quick Start

### 1. Start the Backend (FastAPI)

From the project root:

```bash
# Install Python dependencies (if not already done)
pip install fastapi uvicorn[standard]

# Start the backend server
uvicorn app:app --reload
```

The API will be available at http://localhost:8000

### 2. Start the Frontend

```bash
cd ui

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Then open http://localhost:5173 in your browser.

### Modes

The UI supports two modes (toggle in the header):

1. **Real Agent** (default): Calls the actual Python training agent via the FastAPI backend
2. **Mock Agent**: Uses simulated data for testing the UI without the backend

## Usage

1. **Start a Training Run**:
   - Type your goal in the chat (e.g., "Train a model to predict loan defaults")
   - Optionally link a dataset using the `@dataset` button
   - Optionally select a model type using the `@model` button

2. **Step-by-Step Confirmation**:
   - Each step pauses for your confirmation
   - Review the metrics and details in the left panel
   - Choose: Accept, Redo (with feedback), or Accept All

3. **View Results**:
   - After training completes, click "View Report" to see the full results
   - Download or copy the report for your records

## Project Structure

```
ui/
├── src/
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # Entry point
│   ├── index.css            # Global styles (Tailwind)
│   ├── components/
│   │   ├── ChatPanel.tsx    # Right-side chat interface
│   │   ├── ProgressPanel.tsx # Left-side progress visualization
│   │   ├── StepNode.tsx     # Individual step display
│   │   ├── FinalReport.tsx  # Results modal with tabs
│   │   └── ui/              # shadcn/ui components
│   ├── hooks/
│   │   └── useAgentState.ts # Main state management hook
│   ├── lib/
│   │   ├── mockAgent.ts     # Simulated agent (replace with real API)
│   │   └── utils.ts         # Utility functions
│   └── types/
│       └── agent.ts         # TypeScript types matching TrainingAgentState
├── package.json
├── tailwind.config.js
├── vite.config.ts
└── README.md
```

## Integration with Real Agent

The UI is currently using a mock agent (`src/lib/mockAgent.ts`) that simulates the training pipeline. To connect to the real agent:

1. **Replace `mockAgent.ts`** with real API calls to your backend
2. **Update `useAgentState.ts`** to:
   - Call `invoke_training_agent()` from your Python backend
   - Use WebSocket or polling to get real-time state updates
3. **Map the real `TrainingAgentState`** to the UI state

### Example Backend Integration

```typescript
// In useAgentState.ts
import { invoke } from '@tauri-apps/api' // or your backend client

async function runStep(stepId: string) {
  // Call the Python agent
  const result = await invoke('run_training_step', {
    stepId,
    state: stateRef.current,
  })
  
  // Update UI state with real results
  setAgentState(result.state)
  updateStep(stepId, {
    status: result.confirmationRequired ? 'awaiting_confirmation' : 'completed',
    metrics: result.metrics,
    details: result.details,
  })
}
```

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Radix UI** + **shadcn/ui** for components
- **Lucide** for icons

## Available Datasets

The UI includes these pre-configured datasets from the catalog:

1. **Loan Default Prediction Dataset** - 255,348 rows
2. **Financial Distress Dataset** - 3,673 rows
3. **Health Insurance Charges Dataset** - 1,338 rows

## Available Models

- Logistic Regression
- Random Forest
- XGBoost
- GLM
- Survival Analysis

## License

MIT
