"""
Prompt templates for the analysis agent.

Each function returns a formatted prompt string ready for LLM invocation.
"""

import json

# TODO: better prompt
# - more like a data scientist
# - have the iteration focus on hitting important parts

def decompose_query_prompt(query: str, user_data: dict | None) -> str:
    """Prompt for breaking a query into subtasks."""
    return f"""Break this query into the MINIMUM number of independent analysis tasks.

IMPORTANT: 
- Most queries need only 1 subtask
- Only create multiple subtasks if the query explicitly asks for DIFFERENT types of analysis
- Do NOT break a single analysis into multiple steps

Query: {query}

User provided data: {json.dumps(user_data) if user_data else "None"}

Return a JSON array with 1-3 subtasks maximum. Each subtask has a "goal" field.

Examples:
- "Assess credit risk" → [{{"goal": "Assess credit risk for the applicant"}}]
- "Assess credit risk AND analyze the sentiment of this text" → [{{"goal": "Assess credit risk"}}, {{"goal": "Analyze sentiment of text"}}]

Return ONLY the JSON array, no other text."""


def select_model_prompt(goal: str, user_data: dict | None, models_text: str) -> str:
    """Prompt for selecting the best model for a subtask."""
    return f"""Given this goal, select the best model from the options below.

Goal: {goal}

User provided data: {json.dumps(user_data) if user_data else "None"}

Available models:
{models_text}

Respond with JSON:
{{
    "decision": "select" | "train" | "skip",
    "model_name": "name of selected model (if decision is select)",
    "reason": "brief explanation"
}}

- "select": Use one of the available models
- "train": None of the models fit, need to train a custom model
- "skip": This subtask doesn't need a model

Return ONLY the JSON, no other text."""


def prepare_data_prompt(goal: str, schema: dict, user_data: dict | None) -> str:
    """Prompt for checking if user data satisfies model schema."""
    return f"""Check if the user's input data can satisfy the model's required schema.

Goal: {goal}

Required schema fields:
{json.dumps(schema, indent=2)}

User provided data:
{json.dumps(user_data, indent=2) if user_data else "None"}

If the user data contains all required fields (or equivalent mappings), extract and format them.
If data is missing, indicate what's needed.

Respond with JSON:
{{
    "data_ready": true | false,
    "prepared_data": {{...fields matching schema...}} (if data_ready is true),
    "missing_fields": ["field1", "field2"] (if data_ready is false),
    "notes": "any mapping or transformation notes"
}}

Return ONLY the JSON, no other text."""


def reflect_prompt(goal: str, model: str | None, data: dict | None, result: str | None) -> str:
    """Prompt for generating structured reflection on execution."""
    return f"""Generate a structured reflection on this analysis step.

Goal: {goal}
Model used: {model or "None"}
Data used: {json.dumps(data) if data else "None"}
Result: {result or "No result"}

Respond with JSON:
{{
    "data_summary": "Brief description of what data was used",
    "model_summary": "Which model was used and why",
    "result_interpretation": "What the result means in context",
    "confidence": "high" | "medium" | "low",
    "gaps": ["any missing information or caveats"]
}}

Return ONLY the JSON, no other text."""

# TODO: Better prompting. Not complete if more analysis needs to be done.
def check_complete_prompt(
    query: str,
    goal: str,
    model: str | None,
    data: dict | None,
    result: str | None,
    reflection: dict | None,
    iteration: int,
    max_iterations: int,
) -> str:
    """Prompt for LLM to review if subtask goal has been accomplished."""
    return f"""Review the work done for this subtask and decide if the goal has been accomplished.

ORIGINAL QUERY: {query}

SUBTASK GOAL: {goal}

WORK DONE:
- Model used: {model or "None"}
- Data prepared: {json.dumps(data) if data else "None"}
- Result: {result or "None"}
- Reflection: {json.dumps(reflection) if reflection else "None"}

ITERATION: {iteration + 1} of {max_iterations}

Respond with JSON:
{{
    "goal_accomplished": true | false,
    "confidence": "high" | "medium" | "low",
    "reason": "brief explanation",
    "next_action": "complete" | "retry_different_approach" | "need_more_data",
    "suggestion": "what to do if not complete (optional)"
}}

Return ONLY the JSON, no other text."""


def synthesize_prompt(query: str, reflections: list[dict]) -> str:
    """Prompt for synthesizing all subtask results."""
    return f"""Review all the analysis results and determine if the original goal is complete.

Original query: {query}

Subtask results:
{json.dumps(reflections, indent=2)}

Respond with JSON:
{{
    "is_complete": true | false,
    "summary": "Brief summary of findings"
}}

Return ONLY the JSON, no other text."""


def generate_report_prompt(query: str, reflections: list[dict]) -> str:
    """Prompt for generating the final analysis report."""
    return f"""Generate a comprehensive analysis report.

Original query: {query}

Analysis results:
{json.dumps(reflections, indent=2)}

Create a report with:
1. Executive Summary - Key findings and recommendations
2. Methodology - Models used, data sources
3. Results - Detailed findings from each analysis step
4. Limitations - Caveats and confidence levels

Format as clean markdown."""

