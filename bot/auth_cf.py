from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-4")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# System message for the AI assistant
SYSTEM_MESSAGE = """
You are an intelligent assistant for a healthcare provider. 
Your role is to accurately answer authentication questions and coverage flow questions using the member details provided.

Please consider the following guidelines:
- Clearly respond with 'Yes' or 'No' for yes/no questions.
- Provide exact information for specific requests.
- Ensure consistency with previous answers.
- If information is unavailable, state so and suggest what additional details are needed.
- Maintain strict adherence to privacy standards.

Focus on clarity, accuracy, and conciseness in your answers.

Once authentication is confirmed, extract the CPT code from the member details and proceed with coverage flow questions.
"""

conversation.memory.chat_memory.add_message(SystemMessage(content=SYSTEM_MESSAGE))

# Sample member details (you can modify or load this dynamically)
payload_data = {
   "CalleeType":"HUMAN_SMALL",
   "Priority":"0",
   "Utterances":"demo transcript",
   "cptCode":"99213",
   "benefitCoverageServiceType":"Chiropractic care",
   "benefitCoverageType":"Rehabilitative",
   "birthDate":"4/5/2022",
   "memberDob":"12/17/1991",
   "memberFirstName":" Nancy ",
   "memberLastName":"Amaya",
   "memberPhoneNumber":"650-834-1151",
   "memberId":"0014168073-01",
   "providerNpi":"1750522140",
   # ... (other fields as needed)
}

coverage_flow_questions = [
    "Has the patient met the deductible? please respond with yes/no",
    "What is the copay for this code?",
    "What is the member's coinsurance amount?",
    "Is medical necessity review for any of the codes? please respond with yes/no",
]

def get_session_history(session_id: str):
    # This is a placeholder. In a real application, you'd implement
    # session management, possibly using a database.
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_message(SystemMessage(content=SYSTEM_MESSAGE))
    return memory

def respond_to_authentication(question: str, payload_data: dict, coverage_flow_questions: list):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Member Details: {payload_data}\n\nCoverage Flow Questions: {coverage_flow_questions}\n\nUser Question: {input}"),
    ])

    chain = prompt_template | llm

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="history",
        history_messages_key="history",
    )

    response = with_message_history.invoke(
        {
            "input": question,
            "payload": payload_data,
            "coverage_questions": coverage_flow_questions
        },
        config={"configurable": {"session_id": "abc123"}},
    )

    return response.content

def extract_info(message):
    # This function can be used to extract specific information from user messages
    # For example, extracting CPT codes, member IDs, etc.
    # Implement as needed for your use case
    pass

def handle_request(tag, input_text, parameters):
    if tag == 'welcome':
        return get_welcome_message()
    elif tag in ['authentication', 'coverage_flow', 'get_parking_info']:
        return respond_to_authentication(input_text, payload_data, coverage_flow_questions)
    else:
        return "I'm not sure how to help with that. Can you please rephrase your request?"

def get_welcome_message():
    return "Welcome! I'm here to assist you with authentication and coverage flow questions. How can I help you today?"

# You can add more helper functions as needed, similar to helpers.py

if __name__ == "__main__":
    # This is for testing purposes
    print(handle_request('welcome', '', {}))
    print(handle_request('authentication', "What is the member's date of birth?", {}))
    print(handle_request('coverage_flow', "What is the copay for the CPT code 99213?", {}))
