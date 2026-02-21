# Introduction

This is a demo project for the **Interview Kickstart GenAI course, Week 2**. This repository demonstrates a Model Context Protocol (MCP) server for sentiment analysis (using TextBlob), and a client application with an AI agent that can interact with the server to perform sentiment analysis tasks.

## Features

- **MCP Server**: Provides a simple sentiment analysis tool via Gradio's MCP integration.
- **AI Agent Client**: A chat interface powered by SmolAgents, capable of reasoning and using tools to analyze text sentiment.
- **Sentiment Analysis**: Analyzes text for polarity, subjectivity, and overall assessment (positive, negative, neutral).

## Setup

1. **Fork the Repository**: Start by forking this repository to your own GitHub account.

2. **Spawn a Codespace Agent**: Create a codespace for the forked repository. This will automatically set up the development environment for you.

3. **Run the Applications**:
   - Start the MCP server: `uv run mcp_server.py`
   - Start the MCP client: `uv run mcp_client.py`

## Usage

Once the server and client are running, use the Gradio UI that is spawned to communicate with the MCP client. The interface allows you to interact with the AI agent, which can perform sentiment analysis on provided text.

## Debugging with MCPInspector

MCP inspector is a debugging tool that uses model-context-protocol to connect to your server. You can use it to ensure your MCP server is up and working correctly.

To invoke MCP inspector tool:

On codespaces:
Note your github url - it should be of the form https://your-url-prefix.github.dev.

```
$ ALLOWED_ORIGINS=https://your-url-prefix-6274.app.github.dev HOST=0.0.0.0 CLIENT_PORT=6274 SERVER_PORT=6277 npx @modelcontextprotocol/inspector
Starting MCP inspector...
⚙️ Proxy server listening on 0.0.0.0:6277
🔑 Session token: d9af2f42281814c74b3dbe001aba833d5c358a06a997b02be110c90e28b55fce
   Use this token to authenticate requests or set DANGEROUSLY_OMIT_AUTH=true to disable auth
```

Note the token from the output.

Open a new tab with url: https://your-url-prefix-6274.app.github.dev/ This is MCP inspector tool hosted on your github VM.

On this MCP tool, open "configuration" dropdown and enter 2 values:

Inspector Proxy Address -> https://your-url-prefix-6277.app.github.dev/ (note the port is 6277 not 6274)
Proxy Session Token: copy from MCP-inspector console output.
Hit "connect", and you should see MCP-inspector come to life.
