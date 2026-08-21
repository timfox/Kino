"""Appendix D prompt templates (Figures 7–9)."""

from __future__ import annotations

SEARCH_AGENT_TEMPLATE = """You are an expert assistant whose task is to answer the given question step by step.
Your question: {task_description}. So far, you have completed {step_count} step(s). Below is
the interaction history, where <search> and </search> enclose your previous search queries, and
<information> and </information> enclose the corresponding results returned by the external search
engine. History: {memory_context}
Now it is your turn to respond at the current step. Begin by conducting your reasoning process.
This reasoning must be enclosed within <think> and </think> tags.
After reasoning, choose only one of the following actions (do not attempt both):
(1) If you determine that you are missing some necessary information, you may use a search engine
to obtain more external knowledge by formatting your query as: <search> your query </search>.
(2) If you have sufficient knowledge to confidently answer the question, provide your final answer
enclosed within <answer> and </answer> tags, without any detailed explanation."""

FAILURE_ABSTRACTION_TEMPLATE = """You are an expert AI trainer specializing in diagnosing why AI agents fail at multi-step reasoning tasks.
## Task Context
{task}
## Failed Trajectory
The agent attempted the task above but failed. Here are the steps it took:
{trajectory}
## Your Analysis Task
Carefully examine the trajectory and produce a structured failure analysis.
**Step 1 – Root Cause Identification**
Identify the PRIMARY failure modes and describe briefly:
**Step 2 – Critical Step Identification**
Identify the SINGLE step where the failure became irreversible (the "point of no return").
**Step 3 – Core Lesson**
State a concise, generalizable lesson (1-2 sentences) that would help an agent avoid this class of
mistake on SIMILAR tasks in the future. Focus on the decision rule, not the specific content.
## Output Format
Wrap your entire analysis in <reflection> tags using this exact structure:
<reflection>
DOMINANT_TYPE: [category from Step 1]
DETAIL: [1-2 sentences explaining why this root cause applies]
CRITICAL_STEP: [step number and brief description of what went wrong]
CORE_LESSON: [the generalizable rule an agent should follow]
RETRIEVAL_QUERY: [the query for the retrieval stage]
</reflection>"""

TASK_RETRIEVAL_TEMPLATE = """You are an expert AI curriculum designer. Your job is to identify which historical training tasks are
most relevant for helping an agent overcome a specific failure pattern.
## Current Failure Pattern
The agent is currently struggling with the following error pattern:
{error_pattern}
## Historical Task Candidates
Below are historical tasks where the agent previously failed. Each entry shows the task description
and a brief failure analysis.
{candidates_text}
## Your Task
Select tasks from the list above that are MOST SIMILAR to the current failure pattern.
## Output Format
Output ONLY the following structured block, with no additional text:
<selected_tasks>
INDEX/TASK/REFLECTIONS: <index, task and reflections from the candidate list>
REASON: <one sentence explaining why this task matches the current failure pattern>
</selected_tasks>
Known failure modes in library:
{mode_library}"""

STATE_PREDICTION_TEMPLATE = """You are acting as the environment model (World-In-Agent).
Current observation:
{state}
Agent action:
{action}
Predict the environment observation {horizon} step(s) after this action.
Respond with plain text inside <pred>...</pred> tags."""

ALFWORLD_AGENT_TEMPLATE = """You are an expert embodied agent in ALFWorld.
Task: {task}
Current observation:
{state}
Choose exactly ONE next action from this list:
{action_list}
Reason briefly, then output the action inside <action>...</action> tags only."""

WEBSHOP_AGENT_TEMPLATE = """You are a WebShop shopping agent.
Task: {task}
Current page:
{state}
Choose exactly ONE next action from this list:
{action_list}
Output the action inside <action>...</action> tags only."""

TOY_AGENT_TEMPLATE = """You are a multi-step task agent.
Task: {task}
Current state:
{state}
Valid actions:
{action_list}
Pick one action and put it inside <action>...</action> tags."""
