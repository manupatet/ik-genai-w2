# --- Imports ---
import gradio as gr # For creating the web-based chat UI
from smolagents import CodeAgent, MCPClient # CodeAgent executes tool-using code; MCPClient connects to the MCP server
from smolagents.models import OpenAIServerModel, ChatMessage # For LLM interaction and message formatting
import os

# Setup the client connection to MCP server
try:
    # Initialize the MCP client with the local server's SSE endpoint
    mcp_client = MCPClient({"url": "http://127.0.0.1:7860/gradio_api/mcp"}, structured_output=False)
    # Retrieve the tool list registered on the MCP server (e.g., sentiment_analysis)
    tools = mcp_client.get_tools()
    # Use the OpenRouter model exposed by the MCP server (e.g., nemotron)

    model = OpenAIServerModel(
            model_id="nvidia/nemotron-3-nano-30b-a3b:free",           # Use OpenRouter model ID format
            api_base="https://openrouter.ai/api/v1",
            api_key=os.environ["API_KEY"],
    )

    # Create a CodeAgent that can reason with tools + code
    agent = CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=["json", "ast", "urllib", "base64"]
    )

    # Function to handle each incoming user message
    def smart_agent(message, history):
        # Prompt that guides the model to either respond directly or trigger the tool+code execution
        decision_prompt = f"""You are a smart AI assistant.

Your job is to decide how to respond to the user's message.

If it's a vague or open-ended prompt (e.g., "Can you help me?", "What's up?"),
respond naturally in plain text — greet the user or ask for more input.

If it's a clear task or query that requires tools, code, or calculations (e.g., "Analyze this", "Summarize this", "What is the sentiment of..."), respond using:
<use_tools>
Thoughts: why you need tools
<code>
# Python code here
# Be sure to print the final result in a clear sentence
</code>

When the message asks for sentiment analysis (e.g., "What is the sentiment of ...", "Analyze this text for sentiment", "Do sentiment analysis of ..."), 
ALWAYS use the sentiment_analysis tool available via the MCP interface.

Think internally and respond appropriately.
Do NOT mention category labels like (A) or (B).
Only return what the user should see.

User message:
\"\"\"{message}\"\"\"
"""
        # Ask the model how to respond based on the user's message
        llm_reply = model.generate(messages=[ChatMessage(role="user", content=decision_prompt)])
        # If the model decides that tool + code execution is needed
        if "<use_tools>" in llm_reply.content and "<code>" in llm_reply.content and "</code>" in llm_reply.content:
            # Trigger the full agentic loop: reasoning + tool execution
            return str(agent.run(message))
        else:
            # Otherwise, return the plain model-generated reply
            return llm_reply.content.strip()

    demo = gr.ChatInterface(
        fn=smart_agent,
        examples=[...],
        title="Agent with MCP Tools",
        description="...",
    )

    # Launch the Gradio app locally
    demo.launch()

finally:
    # Disconnect from the MCP server when the script exits
    mcp_client.disconnect()