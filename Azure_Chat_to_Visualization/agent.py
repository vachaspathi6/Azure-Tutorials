import os
import openai
from langchain.llms import AzureOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import json

# Set environment variables for OpenAI configuration
os.environ['OPENAI_API_KEY'] = "1d7f43de41394cdb920f68a92b5a916f"
os.environ['OPENAI_API_BASE'] = "https://termai.openai.azure.com/"
os.environ['OPENAI_DEPLOYMENT_NAME'] = "gpt-35-turbo"  # Correct deployment name
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"
os.environ['OPENAI_API_TYPE'] = "azure"

# Fetch the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_DEPLOYMENT_NAME = os.getenv('OPENAI_DEPLOYMENT_NAME')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')

# Configure the OpenAI API
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_type = OPENAI_API_TYPE

def create_pd_agent(filename: str):
    llm = AzureOpenAI(
        openai_api_key=OPENAI_API_KEY,
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_type=OPENAI_API_TYPE
    )
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(llm, df, verbose=False, handle_parsing_errors=True)

def query_pd_agent(agent, query):
    prompt = (
        """
        You must use the matplotlib library if required to create any chart.

        If the query requires creating a chart, respond with "chart" as follows:
        {"chart": {"type": "bar", "x": ["Year"], "y": ["Number of Movies"]}}

        If the query requires creating a table, respond with "table" as follows:
        {"table": {"columns": ["Year", "Number of Movies"], "data": [[2018, 10], [2019, 15], ...]}}

        Let's think step by step.

        Here is the query: 
        """
        + query
    )

    response = agent.run(prompt)

    return response

def visualize_response(response):
    try:
        response_dict = json.loads(response)  # Try to decode JSON
        print("Raw Response:", response_dict)  # Print the raw response
        if 'final_answer' in response_dict:
            final_answer = response_dict['final_answer']
            print("Final Answer:", final_answer)
        if 'action' in response_dict:
            action = response_dict['action']
            print("Action:", action)
    except json.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
    except Exception as e:
        print("Other error occurred:", e)

def main():
    agent = create_pd_agent('your_csv_file.csv')  # Provide your CSV filename here
    query = 'Your query here'
    response = query_pd_agent(agent, query)
    visualize_response(response)

if __name__ == "__main__":
    main()
