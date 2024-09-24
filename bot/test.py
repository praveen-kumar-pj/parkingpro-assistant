import os
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Set environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize the language model
llm = ChatOpenAI(model="gpt-4")

# Create a store for session histories
session_histories = {}

def get_session_history(session_id: str):
    """Retrieve or create a session history for the given session ID."""
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


# Define the configuration
config = {"configurable": {"session_id": "abc2"}}

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent assistant for a healthcare provider. 
            Your role is to accurately answer the following authentication question using the member details provided below:
            
            Question: {question}
            Member Details: {payload_data}
            
            Please consider the following guidelines to formulate your response:
            - **Yes/No Questions:** Clearly respond with 'Yes' or 'No' based on the member details.
            - **Specific Information Requests:** Provide the exact information requested from the member data.
            - **Clarification/Follow-up Questions:** Ensure consistency with previous answers; reference past responses where necessary.
            - **Unanswerable Questions:** If a question cannot be answered with the provided data, state that the information is unavailable and suggest what additional details are needed.
            - **Data Confidentiality:** Maintain strict adherence to privacy standards in all responses.
            
            Focus on clarity, accuracy, and conciseness in your answers. 

            Once the insurace guy confirms authentication, extract the CPT code from {payload_data}. Begin asking the coverage flow questions one by one, waiting for the insurance representative's response after each question. Store the responses in a dictionary format.

            Coverage Flow Questions: {coverage_flow_questions}
            
            While asking the 1st coverage question, add/provide the cpt code too along with the question.
             
            After all coverage flow questions have been asked, provide the complete dictionary of responses when requested by the insurance representative. 

            Finally, inquire about the reference number and request feedback for the insurance representative before concluding. Donot proivide the summary of questions until the reference number and feedback is asked. 
            
            POINTS TO REMEMBER:
                - ask questions prefessionally and dont not mention "this is first question", "this is second question" and so on..
                - donot provide the record or summary or any other call/chat history details before asking the reference number and feedback and until the insurance guy asks.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# Create the chain
chain = prompt_template | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history",
)

def respond_to_authentication(question: str, payload_data: dict, coverage_flow_questions: list):
    """Respond to an authentication question using the provided payload data."""
    response = with_message_history.invoke(
        {
            "input": question,
            "question": question,
            "payload_data": payload_data,
            "coverage_flow_questions": coverage_flow_questions
        },
        config=config
    )
    return response.content


# Sample member details
payload_data = {
   "CalleeType":"HUMAN_SMALL",
   "Priority":"0",
   "Utterances":"demo transcript",
   "cptCode":"99213",
   "benefitCoverageServiceType":"Chiropractic care",
   "benefitCoverageType":"Rehabilitative",
   "birthDate":"4/5/2022",
   "callLogCollectionName":"log",
   "callPurpose":"benefits verification",
   "callSessionCollectionName":"coverageInfo",
   "callbackNumber":"3582375670",
   "calleeName":"Jennifer",
   "callee_number":"18006789133",
   "customersCollectionName":"patients",
   "data":"{}",
   "date":"02-26-2024",
   "date_priority":"None",
   "diagnosisCode":"[\"d789\",\"d123\",\"d234\"]",
   "extensionNumber":"21436587",
   "gender":"male",
   "healthPlanNumber":"12345",
   "healthPlanTypePrimary":"Nancy Amaya",
   "healthPlanTypeSecondary":"Aetna",
   "insurancePlanType":"Medical",
   "localInsurance":"Blue Cross",
   "memberCity":"CA",
   "memberDob":"12/17/1991",
   "memberFirstName":" Nancy ",
   "memberGroupNumber":"0000021740000",
   "memberId":"0014168073-01",
   "memberLastName":"Amaya",
   "memberPhoneNumber":"650-834-1151",
   "memberSecondaryFirstName":"Georgina",
   "memberSecondaryGroupNumber":"456432",
   "memberSecondaryId":"23456",
   "memberSecondaryLastName":"Mccullum",
   "memberSecondaryMiddleName":"smith",
   "memberSecondaryRelationship":"spouse",
   "memberSsn":"223344",
   "memberState":"San Mateo",
   "memberStreet":"120 N San Mateo Dr apt 309",
   "memberZipcode":"94401",
   "payorGreetingValue":"1",
   "payorName":"American Specialty Health",
   "payorPhoneNumber":"1-800-678-9133",
   "payorProviderRelation":"in network",
   "placeCategory":"clinic",
   "placeType":"Outpatient",
   "planType":"Medicare",
   "prior_authorization_field":"cpt code",
   "providerClinicAddress":"300 San Mateo Drive #12, San Mateo, CA 94401",
   "providerClinicName":"test clinic",
   "providerFacilityName":"test facility",
   "providerFaciltiyAddress":"none4",
   "providerFaxNumber":"1234",
   "providerFirstName":"Dr. Hamed",
   "providerGeneralType":"Specialist",
   "providerId":"j82eb2n2",
   "providerLastName":"Alereza",
   "providerMiddleName":"C",
   "providerNpi":"1750522140",
   "providerPhoneNumber":"650 393 4280",
   "providerPhoneNumberType":"direct",
   "providerSpecificType":"Chiropractor",
   "providerNpiId":"1750522140",
   "providerState":"San Mateo",
   "providerStreet":"400 N San Mateo Dr #1",
   "providerTaxId":"800367001",
   "providerZipcode":"94401",
   "purposeCollectionName":"Benefit",
   "reason":"None",
   "requiredCoverageAbsolutes":"{'copay':{'office visit':'','urgent care':'', 'emergency room':'', 'hospital admission':''}, 'deductibe':{'individual':'', 'family':''}, 'annual maximum':{'individual':'', 'family':''}, 'out of pocket':{'individual':'', 'family':''},}",
   "resourceType":"Clinic",
   "serviceType":"{'Medical': {'CHIROPRACTIC MANIPULATIVE TREATMENT (CMT); SPINAL, 1-2 REGIONS': {'cpt-codes': '98940'}}}",
   "time":"14:29:24",
   "time_priority":"None",
   "batchRequestDocumentId":"7av5gru1nltqyyueo2j3iw7moqva3f9d:04092024135208",
   "created_date":"03-06-2024",
   "call_count":0,
   "callDocumentId":"",
   "callLogDocumentId":"",
   "callRequestDocumentId":"",
   "customerDocumentId":"0014168073-01",
   "memberMiddleName":"K"
}


coverage_flow_questions = [
    "Has the patient met the deductible? please respond with yes/no",
    # "What is the prior auth turn around time?",
    "What is the copay for this code?",
    # "How do I obtain prior authorization?",
    # "Do you see an active prior authorization on file? please respond with yes/no",
    # "Is referral required for any of the codes? please respond with yes/no",
    # "Is prior authorization required for this code? please respond with yes/no",
    "What is the member's coinsurance amount?",
    # "What is the deductible?",
    "Is medical necessity review for any of the codes? please respond with yes/no",
    # "What is the out-of-pocket maximum?"
]
# Sample trickier authentication questions
questions = [
    "Can you please provide your full name?",
    "What is your date of birth?",
    "What is your NPI ID?",
    "Can u repeat it?",
    "Is the member name raj?"
]
# Respond to authentication questions
while True:
    question = input("Please enter question: ")
    if question.lower() == "exit":
        break
    auth_response = respond_to_authentication(question, payload_data, coverage_flow_questions)
    print(f"Assistant: {auth_response}")
