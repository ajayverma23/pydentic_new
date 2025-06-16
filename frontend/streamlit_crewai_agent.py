import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app_schema.data_validation import ContentRequest, ContentResult
from agents.crewai_multiple_agents import MultiLevelCrewAI

# --- Streamlit UI ---
st.title('Content Creation App')
st.markdown("* Use the menu at left to create the content")
st.sidebar.markdown("## Select following setup")

col1, col2 = st.columns([2, 1])
user_input1 = col1.text_input("User Query", placeholder='Describe what you want to generate...')
reference_data1 = col2.text_input("Reference Data", placeholder='URL, weblink, or data')
context1 = st.text_input("Context of topic", placeholder='Additional context or background info')
subject1 = st.text_input("Subject of Content", placeholder='e.g., GenAI, HuggingFace')
content_size1 = st.sidebar.number_input("Content Size", min_value=100, max_value=1000000, value=1000)
target_audience1 = st.sidebar.selectbox("Target Audience", ["Doctors", "General Public", "Engineers"])
audience_level1 = st.sidebar.selectbox("Audience Level", ["Beginner", "Intermediate", "Expert"])
content_type1 = st.sidebar.selectbox("Content Type", ["Blog", "Training Contents", "Test cases", "Requirement", "Process"])
topic1 = st.sidebar.text_input("Topic", placeholder='e.g., GenAI, AIML')
role1 = st.sidebar.selectbox("Agent Role", ["Blog expert", "Tester", "Training Content Creator"])

st.write("Debug input values:", {
    "user_input": user_input1,
    "reference_data": reference_data1,
    "context": context1,
    "subject": subject1,
    "content_size": content_size1,
    "target_audience": target_audience1,
    "audience_level": audience_level1,
    "content_type": content_type1,
    "topic": topic1,
    "role": role1
})

content_request = ContentRequest(
            user_input=user_input1,
            reference_data=reference_data1,
            context=context1,
            subject=subject1,
            content_size=content_size1,
            target_audience=target_audience1,
            audience_level=audience_level1,
            content_type=content_type1,
            topic=topic1,
            role=role1
        )

if st.button("Generate Content"):
    try:
        crew = MultiLevelCrewAI()
        result = crew.run(content_request)
        print("result-type:", type(result))
        # Add PDF export here if needed
        # Display results
        st.markdown("## Final Output")
        #st.markdown(result['final'])
        st.markdown(result)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Check the terminal for full error details.")
        raise e  # Show full traceback in terminal