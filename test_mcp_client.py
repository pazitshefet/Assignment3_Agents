import asyncio
from fastmcp import Client

async def main():
    async with Client("mcp_server.py") as client:
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools:
            print("-", tool.name)

        print("\nCalling list_categories...")
        result = await client.call_tool("list_categories", {})
        print(result)

        print("\nCalling count_rows...")
        result = await client.call_tool("count_rows", {"category": "REFUND"})
        print(result)

        print("\nCalling sample_examples...")
        result = await client.call_tool("sample_examples", {"category": "REFUND", "limit": 3, "offset": 0})
        print(result)

if __name__ == "__main__":
    asyncio.run(main())