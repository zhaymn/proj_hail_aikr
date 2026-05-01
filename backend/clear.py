from src.research_agent.runtime import ResearchAgentRuntime
from src.research_agent.config import get_settings

r = ResearchAgentRuntime(get_settings())
result = r.clear_papers()
print(result)