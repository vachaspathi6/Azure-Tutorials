# Imports the OpenAI API and Streamlit libraries.
import openai
import streamlit as st
import time
from streamlit_chat import message

# Streamlit to set the page header and icon.
st.set_page_config(
        page_title="Azure OpenAI GPT-3.5 Turbo",
        page_icon="🤖",
        layout="centered"
    )

# Sidebar for API key and other configuration inputs
st.sidebar.header("Configuration")
api_base = st.sidebar.text_input("API Base URL", placeholder="https://<name>.openai.azure.com/")
api_version = st.sidebar.text_input("API Version", "2023-03-15-preview")
api_key = st.sidebar.text_input("API Key", type="password")

# Streamlit page title
st.title("💬 Azure OpenAI ChatBot")
st.caption("🚀 Experience seamless conversations with our chatbot powered by Azure OpenAI's GPT-3.5 Turbo")

# Initializes the Streamlit session state with default values.
if 'prompts' not in st.session_state:
    st.session_state['prompts'] = [{"role": "system", "content": "How can I help you?"}]
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = []


# Define the 'generate_response' function to send the user's message to the AI model and append the response.
def generate_response(prompt):
    st.session_state['prompts'].append({"role": "user", "content": prompt})

    start_time = time.time()

    try:
        completion = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.95,
            messages=st.session_state['prompts']
        )

        end_time = time.time()

        message = completion.choices[0].message.content

        latency = end_time - start_time
        input_tokens = len(prompt.split())
        output_tokens = len(message.split())
        throughput = output_tokens / latency

        # Store metrics
        st.session_state['metrics'].append({
            'latency': latency,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'throughput': throughput
        })

        return message
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return "Sorry, I couldn't process that request."

# Function to reset the conversation history and introduce the AI assistant.
def new_topic():
    st.session_state['prompts'] = [{"role": "system", "content": "You are a robotic minion created by Eason in Minions.app Laboratory. You are upbeat and friendly. Your response should be as concise with a sense of humor. You introduce yourself when first saying, Bello! Buddy. If the user asks you for anything information about Minions, or Despicable Me, or Universal Studios, you will try to use our intelligence to reply. If the user asks you not related Minions, or Despicable Me, or Universal Studios, you will tell them I don't know."}]
    st.session_state['past'] = []
    st.session_state['generated'] = []
    st.session_state['user'] = ""
    st.session_state['metrics'] = []

# Function to send the user's message to the AI model and append the response to the conversation history.
def chat_click():
    if not api_key or not api_base or not api_version:
        st.sidebar.error("Please provide the API Key, API Base URL, and API Version.")
    else:
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_key = api_key

        if st.session_state['user'] != '':
            user_chat_input = st.session_state['user']
            output = generate_response(user_chat_input)
            st.session_state['past'].append(user_chat_input)
            st.session_state['generated'].append(output)
            st.session_state['prompts'].append({"role": "assistant", "content": output})
            st.session_state['user'] = ""

# Get user input using st.chat_input
user_input = st.chat_input("You:")

if user_input:
    st.session_state['user'] = user_input
    chat_click()

# Add a "New Topic" button at the top of the page
if st.button("New Topic"):
    new_topic()

# Display messages in the conversation history.
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state['generated'][i], avatar_style='bottts', key=str(i))
        message(st.session_state['past'][i], is_user=True, avatar_style='thumbs', key=str(i) + '_user')

# Display metrics in the sidebar
if st.session_state['metrics']:
    st.sidebar.subheader("Evaluation Metrics")
    latest_metrics = st.session_state['metrics'][-1]
    st.sidebar.write(f"- **Throughput:** {latest_metrics['throughput']:.6f} tokens/second")
    st.sidebar.write(f"- **Latency:** {latest_metrics['latency']:.6f} seconds")
    st.sidebar.write(f"- **Input Tokens:** {latest_metrics['input_tokens']}")
    st.sidebar.write(f"- **Output Tokens:** {latest_metrics['output_tokens']}")
