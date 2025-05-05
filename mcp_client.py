import asyncio
from langchain_groq import ChatGroq
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

groq_model = ChatGroq(model_name="deepseek-r1-distill-llama-70b")
server_params = StdioServerParameters(
    command = "python",
    args = ["mcp_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(groq_model, tools)
            agent_response = await agent.ainvoke({"messages": "Analyze how revenue of MSFT is changing over time?"})
            print(agent_response)


if __name__ == "__main__":
    asyncio.run(main())
