from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.graph import create_graph
from app.schema import ClassifierState

def main():
    load_dotenv()

    graph = create_graph()

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_intent": "",
            "user_request": user_input,
            "confidence_level": 0.0
        }
        result = graph.invoke(initial_state)
    
        ai_msg = result["messages"][-1].content

        print()
        print(f"AI: {ai_msg}")
        print()


if __name__ == "__main__":
    main()