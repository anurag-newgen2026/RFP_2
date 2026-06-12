# Import the compiled graph from your agent file
import os
from src.normal_agent import _GRAPH

print("Generating graph image...")

# Generate PNG byte data using LangGraph's built-in Mermaid renderer
png_data = _GRAPH.get_graph().draw_mermaid_png()

# Save it to a file named 'agent_architecture.png'
with open("agent_architecture.png", "wb") as f:
    f.write(png_data)

print("Success! Graph saved as agent_architecture.png")

os._exit(0)
