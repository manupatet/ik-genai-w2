# --- Imports ---
import json # For converting Python dictionaries to JSON strings
import gradio as gr # Gradio is used to create the web-based UI and MCP server
from textblob import TextBlob # TextBlob is a simple NLP library for sentiment analysis

# --- Sentiment Analysis Function ---
def sentiment_analysis(text: str = "") -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    # Create a TextBlob object from input text
    blob = TextBlob(text)
    # Get sentiment properties (polarity & subjectivity)
    sentiment = blob.sentiment
    
    # Build the result dictionary
    result = {
        "polarity": round(sentiment.polarity, 2), # Polarity: [-1.0, 1.0]
        "subjectivity": round(sentiment.subjectivity, 2), # Subjectivity: [0.0, 1.0]
        "assessment": "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
    }
    # Return the result as a JSON string
    return json.dumps(result)

# Create the Gradio interface
demo = gr.Interface(
    fn=sentiment_analysis, # The function to call when the user submits input
    inputs=gr.Textbox(placeholder="Enter text to analyze..."), # User input box
    outputs=gr.Textbox(), # Output will be displayed as plain text
    title="Text Sentiment Analysis", # Title shown in the UI
    description="Analyze the sentiment of text using TextBlob" # Short description
)

# Launch with MCP
if __name__ == "__main__":
    # Launch Gradio with MCP support on default port (7860 or next free one)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        mcp_server=True)