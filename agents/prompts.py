"""
Prompt templates for the analysis agent.

Each function returns a formatted prompt string ready for LLM invocation.
"""

import json

def plan_analysis_prompt(query: str, user_data: dict | None) -> str:
    """Prompt for creating an analysis execution plan."""
    data_summary = ""
    if user_data:
        data_summary = f"""
Available data provided by user:
{json.dumps(user_data, indent=2)}
"""
    else:
        data_summary = "\nNo data provided by user - will need to retrieve from data sources."

    return f"""You are a senior data scientist planning an analysis project.

USER REQUEST:
{query}
{data_summary}

Your task: Create an execution plan by breaking this into independent analysis tasks.

GUIDELINES:
- Each task should be a self-contained analysis that can run independently
- Simple requests need only 1 task (e.g., "assess credit risk" → 1 task)
- Complex requests may need multiple tasks when they require:
  * Different types of models (e.g., risk scoring AND sentiment analysis)
  * Analysis of different data sources or subjects
  * Comparisons across different scenarios
- Do NOT create separate tasks for steps within a single analysis (data prep, modeling, interpretation are ONE task)
- Be pragmatic: fewer well-defined tasks are better than many fragmented ones

EXAMPLES:
- "Assess credit risk for this applicant" → 1 task
- "Assess credit risk AND forecast next quarter revenue" → 2 tasks (different model types)
- "Compare credit risk across 3 applicants" → 1 task (same analysis, multiple subjects)
- "Analyze loan default risk, predict claim likelihood, and assess market sentiment" → 3 tasks

Respond with JSON:
{{
    "reasoning": "Brief explanation of how you broke down the request",
    "tasks": [
        {{
            "goal": "Clear, actionable goal for this analysis task",
            "type": "risk_assessment | forecasting | sentiment_analysis | classification | clustering | other",
            "priority": 1-3 (1 = highest)
        }}
    ]
}}

Return ONLY the JSON, no other text."""


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


def subagent_synthesize_prompt(
    query: str,
    goal: str,
    model: str | None,
    data: dict | None,
    result: str | None,
    reflection: dict | None,
) -> str:
    """Prompt for a subagent to synthesize its own subtask results."""
    return f"""You are a data science agent that just completed an analysis task. Synthesize your findings.

ORIGINAL USER QUERY: {query}

YOUR ASSIGNED TASK: {goal}

YOUR WORK:
- Model used: {model or "None"}
- Data analyzed: {json.dumps(data) if data else "None"}
- Raw result: {result or "None"}
- Reflection: {json.dumps(reflection) if reflection else "None"}

Create a synthesis of your findings that can be combined with other agents' work.

Respond with JSON:
{{
    "task_goal": "{goal}",
    "status": "complete" | "partial" | "failed",
    "model_used": "model name or null",
    "key_findings": "Main findings in 2-3 sentences",
    "metrics": {{"metric_name": "value"}} (any quantitative results),
    "confidence": "high" | "medium" | "low",
    "limitations": ["limitation 1", "limitation 2"],
    "recommendation": "Actionable recommendation based on findings"
}}

Return ONLY the JSON, no other text."""


def generate_report_prompt(query: str, syntheses: list[dict]) -> str:
    """Prompt for generating the final analysis report from all subagent syntheses."""
    return f"""Generate a comprehensive analysis report combining results from multiple analysis agents.

ORIGINAL USER QUERY: {query}

AGENT SYNTHESES:
{json.dumps(syntheses, indent=2)}

Create a unified report that:
1. Executive Summary - Key findings and recommendations from ALL agents
2. Methodology - Models used by each agent, data sources
3. Results - Detailed findings organized by analysis task
4. Cross-Analysis Insights - Connections between different analyses (if applicable)
5. Limitations - Combined caveats and confidence levels
6. Final Recommendations - Unified actionable recommendations

Format as clean markdown. Integrate the agents' findings cohesively, not just as separate sections."""


def decide_next_step_prompt(
    query: str, 
    syntheses: list[dict],
    iteration_count: int = 0,
    max_iterations: int = 3,
) -> str:
    """
    Prompt for reflecting on completed work and deciding whether to finalize or iterate.
    """
    # Extract completed task names for clear visibility
    completed_tasks = [s.get("task_goal", "Unknown task") for s in syntheses]
    completed_tasks_list = "\n".join(f"  - {task}" for task in completed_tasks)
    
    iteration_context = ""
    if iteration_count > 0:
        remaining = max_iterations - iteration_count
        iteration_context = f"""
## ITERATION STATUS
This is iteration {iteration_count + 1} of {max_iterations} maximum.
{"⚠️ FINAL ITERATION - You MUST choose 'finalize'." if remaining == 0 else ""}
"""

    return f"""## USER'S REQUEST
{query}
{iteration_context}
## TASKS ALREADY COMPLETED (DO NOT REPEAT)
{completed_tasks_list}

## FULL TASK DETAILS
{json.dumps(syntheses, indent=2)}

## DECISION RULES

1. Look at the tasks already completed above.
2. Did they answer the user's request? If YES → finalize.
3. Is there something MISSING that requires a DIFFERENT task? If YES → iterate.

⚠️ CRITICAL: You CANNOT run these tasks again or any similar variation:
{completed_tasks_list}

If the only work you can think of is similar to the tasks above, choose "finalize".
Only choose "iterate" if there is a genuinely DIFFERENT task needed.

## RESPONSE FORMAT
{{
    "reflection": "If iterating: describe what DIFFERENT task is needed. If finalizing: leave empty.",
    "decision": "finalize" | "iterate"
}}

Return ONLY valid JSON."""


def plan_analysis_with_reflection_prompt(
    query: str, 
    user_data: dict | None, 
    reflection: dict,
    completed_tasks: list[str],
) -> str:
    """
    Prompt for re-planning analysis based on prior work and reflection.
    
    Used when iterating after the decide_next_step determines more work is needed.
    """
    data_summary = ""
    if user_data:
        data_summary = f"""
Available data provided by user:
{json.dumps(user_data, indent=2)}
"""
    else:
        data_summary = "\nNo data provided by user - will need to retrieve from data sources."

    # Get the reflection text
    reflection_text = reflection.get("reflection", "No reflection available")
    
    # Format completed tasks list
    completed_tasks_list = "\n".join(f"  - {task}" for task in completed_tasks)

    return f"""## USER'S REQUEST
{query}
{data_summary}

## TASKS ALREADY COMPLETED (DO NOT REPEAT OR CREATE SIMILAR)
{completed_tasks_list}

## WHY WE'RE ITERATING
{reflection_text}

## YOUR TASK
Create tasks for ONLY genuinely DIFFERENT work.

⚠️ CRITICAL RULES:
- You CANNOT create any task similar to the ones listed above
- If you can't think of a genuinely DIFFERENT task, return empty tasks array
- Only add tasks that address the specific gap mentioned in "WHY WE'RE ITERATING"

Respond with JSON:
{{
    "reasoning": "What DIFFERENT task are you adding? Confirm it's not similar to completed tasks.",
    "tasks": [
        {{
            "goal": "Goal for a genuinely DIFFERENT task",
            "type": "risk_assessment | forecasting | sentiment_analysis | classification | clustering | other",
            "priority": 1-3
        }}
    ]
}}

If no genuinely different work is needed:
{{
    "reasoning": "All necessary work has been completed",
    "tasks": []
}}

Return ONLY valid JSON."""

