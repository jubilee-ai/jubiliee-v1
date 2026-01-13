# Agents

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS AGENT v2                                   │
│                         (gpt-5.1, main orchestrator)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ has 3 tools
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ statistical_      │   │ model_execution_  │   │ data_lookup_      │
│ analysis_tool     │   │ tool              │   │ tool              │
├───────────────────┤   ├───────────────────┤   ├───────────────────┤
│ Find + analyze    │   │ Run pretrained    │   │ Simple data       │
│ data patterns     │   │ models on inputs  │   │ retrieval only    │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          ▼                       │                       │
┌─────────────────────────┐       │                       │
│ STATISTICAL ANALYSIS    │       │                       │
│ AGENT (gpt-5.1)         │       │                       │
├─────────────────────────┤       │                       │
│ Has both retrieval +    │       │                       │
│ analysis tools directly │       │                       │
└─────────┬───────────────┘       │                       │
          │                       │                       │
          ▼                       │                       │
┌─────────────────────────────────┴───────────────────────┴─────────────────┐
│                              TOOL LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DATA RETRIEVAL TOOLS          ANALYSIS TOOLS         PRETRAINED MODELS   │
│  ─────────────────────         ──────────────         ─────────────────   │
│  • catalog_search_tool         • eda_report           • credit_risk       │
│  • list_datasets_tool          • data_validation      • loan_default      │
│  • dataset_get_tool            • correlation_matrix   • finbert_tone      │
│  • sql_query_tool              • trend_analysis       • finbert_sentiment │
│  • get_sql_schema_tool         • group_summary        • claim_detection   │
│  • join_merge_tool             • concentration        • chronos2_forecast │
│                                • feature_diagnostics  • timesfm_forecast  │
│                                • distribution                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│  datasets/                                                                 │
│  ├── csv/insurance.csv          (health insurance costs)                   │
│  ├── csv/Loan_default.csv       (loan default prediction)                  │
│  ├── csv/Financial Distress.csv (corporate distress)                       │
│  └── sql/*.sql                  (SQL warehouse tables)                     │
│                                                                            │
│  catalog.json                   (dataset metadata & search index)          │
└────────────────────────────────────────────────────────────────────────────┘
```

## Tool Selection Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS AGENT v2                         │
│                                                              │
│  "What data do you have?"  ───────────►  data_lookup_tool   │
│                                                              │
│  "Analyze patterns in..."  ───────────►  statistical_       │
│  "What factors predict..."               analysis_tool      │
│                                                              │
│  "Score this applicant"    ───────────►  model_execution_   │
│  "Analyze sentiment of..." (text input)  tool               │
│                                                              │
│  "Find records to run      ───────────►  data_lookup_tool   │
│   through the model"                     then                │
│                                          model_execution_    │
│                                          tool                │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `analysis_agent_v2.py` | Main orchestrator agent with 3 tools |
| `statistical_analysis_agent.py` | Sub-agent for data retrieval + statistical analysis |
| `data-retrieval/agent.py` | Data retrieval agent (used by data_lookup_tool) |
| `model_index.py` | Registry of available pretrained models |
| `prompts.py` | System prompts for all agents |

## Tools Summary

### 1. `statistical_analysis_tool`
- **Purpose**: Find data and run statistical analysis
- **Use for**: Profiling, correlations, trends, group comparisons, pattern discovery
- **Behavior**: Can find and load data automatically, then analyze it

### 2. `model_execution_tool`
- **Purpose**: Run pretrained models on specific inputs
- **Use for**: Credit risk, loan default, sentiment analysis, forecasts
- **Requires**: Specific input data (e.g., applicant details, text to analyze)

### 3. `data_lookup_tool`
- **Purpose**: Simple data retrieval without analysis
- **Use for**: Showing available datasets, displaying sample records, finding data for model inputs
- **NOT for**: Statistical analysis or pattern discovery
