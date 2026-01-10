# =============================================================================
# Analysis Agent Architecture Reference
# =============================================================================
# See analysis_agent.py for the implementation.
#
# FLOW:
#
#   User Query + Optional Data
#            │
#            ▼
#   ┌─────────────────┐
#   │   DECOMPOSE     │  Break query into independent subtasks
#   └────────┬────────┘
#            │
#            ▼
#   ┌─────────────────────────────────────────────────────────────────────┐
#   │  SUBTASK LOOP (for each subtask, until complete or max iterations) │
#   │                                                                     │
#   │    ┌──────────────────┐                                             │
#   │    │  SELECT MODEL    │  LLM picks from top 3 candidates            │
#   │    └────────┬─────────┘                                             │
#   │             │                                                       │
#   │             ▼                                                       │
#   │    ┌──────────────────┐                                             │
#   │    │  PREPARE DATA    │  Map user data to schema or flag missing    │
#   │    └────────┬─────────┘                                             │
#   │             │                                                       │
#   │             ▼                                                       │
#   │    ┌──────────────────┐                                             │
#   │    │  EXECUTE MODEL   │  Run pretrained model on data               │
#   │    └────────┬─────────┘                                             │
#   │             │                                                       │
#   │             ▼                                                       │
#   │    ┌──────────────────┐                                             │
#   │    │  REFLECT         │  LLM interprets result, identifies gaps     │
#   │    └────────┬─────────┘                                             │
#   │             │                                                       │
#   │             ▼                                                       │
#   │    ┌──────────────────┐                                             │
#   │    │ CHECK COMPLETE   │  Success → next subtask                     │
#   │    │                  │  Failed + retries left → retry              │
#   │    │                  │  Failed + no retries → next subtask         │
#   │    └──────────────────┘                                             │
#   │                                                                     │
#   └─────────────────────────────────────────────────────────────────────┘
#            │
#            ▼
#   ┌─────────────────┐
#   │   SYNTHESIZE    │  Review all reflections, check if goal met
#   └────────┬────────┘
#            │
#            ├──── Gaps? → Generate new subtasks → back to SUBTASK LOOP
#            │
#            ▼
#   ┌─────────────────┐
#   │ GENERATE REPORT │  Final markdown report with findings
#   └─────────────────┘
#
# =============================================================================
# KEY COMPONENTS
# =============================================================================
#
# STATE (AgentState):
#   - query: User's original request
#   - user_data: Optional dict with input data
#   - subtasks: List of Subtask objects
#   - current_subtask_idx: Which subtask we're processing
#   - all_reflections: Collected reflections from all subtasks
#   - is_complete: Whether synthesis determined goal is met
#   - final_report: The generated report
#
# SUBTASK:
#   - goal: What this subtask is trying to accomplish
#   - status: pending | in_progress | complete | failed
#   - selected_model: Which pretrained model was chosen
#   - model_schema: Required input fields for the model
#   - prepared_data: Data formatted to match schema
#   - result: Output from model execution
#   - reflection: Structured interpretation of result
#   - iteration_count: How many times we've retried
#
# MODEL INDEX (model_index.py):
#   - 7 pretrained models with metadata
#   - Each has: name, description, when_to_use, when_not_to_use, schema, output
#   - LLM uses this to select the best model for each subtask
#
# =============================================================================
# USAGE
# =============================================================================
#
# from agents import run_analysis
#
# result = run_analysis(
#     query="Assess credit risk for this applicant",
#     user_data={
#         "gender": "m",
#         "marital": "married",
#         "age": 35,
#         "income": 75000,
#         ...
#     }
# )
# print(result)  # Markdown report
