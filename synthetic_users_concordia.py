# Copyright 2025 The Paper Authors of "Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions". All rights reserved.
import datetime
import re
from typing import Any
import uuid
from concordia.agents import simple_llm_agent
from concordia.clocks import game_clock
from concordia.typing import agent as agent_lib

class SyntheticUserAgent(simple_llm_agent.SimpleLLMAgent):
  """A simple LLM agent for PHA synthetic user."""

  @override
  def get_last_log(self) -> dict[str, Any]:
    """Returns debugging information in the form of a dictionary."""
    return {}

def get_action_spec(name) -> agent_lib.ActionSpec:
  """Returns the action spec for the given object."""
  call_to_action = f'Given the above, generate what {name} would say to the Coach next in this conversation. Respond in the format `{name} -- "..."` For example, {name} -- "How can I sleep better?" If the Coach has asked for input, make sure to generate specific and detailed answer to the question. Bear in mind the primary sleep concern, sleep goals, reasons for those goals, and barriers of {name} provided above. The tone and style of the conversation should match {name}\'s descriptions above. Do not generate more than 2 sentences.\n'
  return agent_lib.free_action_spec(
      call_to_action=call_to_action,
  )

def get_clock() -> game_clock.MultiIntervalClock:
  major_time_step = datetime.timedelta(minutes=1)
  minor_time_step = datetime.timedelta(seconds=10)
  setup_time = datetime.datetime(hour=20, year=2025, month=10, day=1)
  return game_clock.MultiIntervalClock(
      start=setup_time, step_sizes=[major_time_step, minor_time_step]
  )

def run_simulation(
    user: agent_lib.GenerativeAgent,
    coach: agent_lib.GenerativeAgent,
    action_spec: agent_lib.ActionSpec,
    clock: game_clock.MultiIntervalClock,
    max_turns: int,
) -> dict[str, Any]:
  """Runs a Concordia simulation."""
  persona_id = uuid.uuid4()
  persona_id_str = str(persona_id)
  coach.observe(f'/persona {persona_id_str}')
  user.observe('Coach -- "How may I help you today?"')
  user_speaches = []
  coach_speeches = []
  for _ in range(max_turns):
    clock.advance()
    user_speech = user.act(action_spec)
    user.observe(user_speech)
    user_speech = re.sub(r'^.* -- "', '', user_speech)
    coach.observe(user_speech)
    coach_speech = coach.act()
    user.observe(f'Coach -- "{coach_speech}"')
    user_speaches.append(user_speech)
    coach_speeches.append(coach_speech)

  retval = {
      'persona_id': persona_id_str,
      'user_speeches': user_speaches,
      'coach_speeches': coach_speeches,
  }
  return retval

synthetic_user_name = 'Alice'
synthetic_user = SyntheticUserAgent(
    agent_name=synthetic_user_name,
    agent_background=f'The following JSON object describes some facts about {synthetic_user_name}\'s sleep, including primary sleep concern, sleep goals, reasons for those goals, and barriers: {your_json_data}',
    model=YourLanguageModelAPI(),
    memories_length=100,
)
action_spec = get_action_spec(synthetic_user_name)
coach = YourCoachingAgent()
clock = get_clock()
simulation_results = run_simulation(
    user=synthetic_user,
    coach=coach,
    action_spec=action_spec,
    clock=clock,
    max_turns=10,
)
