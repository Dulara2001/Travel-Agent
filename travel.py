import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

# Define the PlannerState
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "the messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="<Your Api key>",  # Groq API key
    model_name="llama-3.3-70b-versatile"
)

# Define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip."),
])

# Define node functions
def input_city(state: PlannerState) -> PlannerState:
    city = st.text_input("Please enter the city you want to visit for your day trip:")
    if city:
        return {
            **state,
            "city": city,
            "messages": state['messages'] + [HumanMessage(content=city)]
        }
    return state

def input_interest(state: PlannerState) -> PlannerState:
    interests = st.text_input(f"Please enter your interests for the trip to {state['city']} (comma-separated):")
    if interests:
        return {
            **state,
            "interests": [interest.strip() for interest in interests.split(",")],
            "messages": state['messages'] + [HumanMessage(content=interests)]
        }
    return state

def create_itinerary(state: PlannerState) -> PlannerState:
    if state['city'] and state['interests']:
        response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=','.join(state['interests'])))
        st.write("### Final Itinerary:")
        st.markdown(response.content)
        return {
            **state,
            "messages": state['messages'] + [AIMessage(content=response.content)],
            "itinerary": response.content,
        }
    return state

# Streamlit App
def main():
    st.title("🌍 Travel Planner")
    st.write("Welcome to the Travel Planner! Enter your details below to generate a day trip itinerary.")

    # Initialize state
    state = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    # Step 1: Input City
    state = input_city(state)

    # Step 2: Input Interests
    if state['city']:
        state = input_interest(state)

    # Step 3: Create Itinerary
    if state['city'] and state['interests']:
        state = create_itinerary(state)

# Run the app
if __name__ == "__main__":
    main()