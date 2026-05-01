from research_agent.graph.builder import build_graph
from research_agent.schemas import Mode

g = build_graph()
state = {
    'session_id': 'test',
    'mode': Mode.LOCAL,
    'message': 'what is the detection precision and recall?',
    'paper_ids': [],
    'history': [],
    'debug': {},
}
result = g.invoke(state)
print(result['debug'])
print(result['answer'])
