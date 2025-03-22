from flask import Flask, render_template, request, jsonify
from typing import List, Dict, TypedDict
from langchain_ollama.llms import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain_groq import ChatGroq
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = "*****Enter your groq API here.*********"

app = Flask(__name__)

# Initialize the Ollama model
model = OllamaLLM(model="jaahas/tiger-gemma-v2:latest", temperature=0.7)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    messages = state["messages"]
    # Get the last message for the model
    last_message = messages[-1].content
    # Include context from previous messages
    if len(messages) > 1:
        context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in messages[:-1]])
        prompt = f"Previous conversation:\n{context}\n\nCurrent message: {last_message}"
    else:
        prompt = last_message
    
    response = model.invoke(prompt)
    return {"messages": messages + [AIMessage(content=response)]}
# Create memory saver
memory = MemorySaver()

# Create workflow
workflow = StateGraph(State)
print("Workflow created")
# Add node
workflow.add_node("chatbot", chatbot)

# Set entry point and edges
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)

# Compile workflow
graph = workflow.compile(checkpointer=memory)

def stream_graph_updates(graph, initial_state, config):
    events = graph.stream(
        initial_state,
        config,
        stream_mode="values",
    )
    for event in events:
        yield event

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user message
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Get thread ID for memory management
        thread_id = request.headers.get('X-Thread-ID', 'default')
        
        # Initialize messages with the new message
        messages = [HumanMessage(content=message)]
            
        # Set up the initial state and config with thread ID
        initial_state = {
            "messages": messages,
            "next": "chatbot"
        }
        
        # Configure thread ID for persistence
        config = {"configurable": {"thread_id": thread_id}}
        # Process message and get response
        response_generator = stream_graph_updates(graph, initial_state, config)
        
        # Get the final message from the generator
        final_message = None
        for message in response_generator:
            final_message = message
        
        if final_message:
            assistant_message = final_message["messages"][-1].content
            return jsonify({
                "response": assistant_message,
                "thread_id": thread_id,
                "message_count": len(final_message["messages"])
            })
        else:
            return jsonify({"error": "No response generated"}), 500

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"Error in chat endpoint: {error_message}")
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
