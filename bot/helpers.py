from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from google.cloud import bigquery
import os
import uuid
import logging
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize BigQuery client
bigquery_client = bigquery.Client()

# Set your BigQuery dataset and table name
DATASET_NAME = 'parkin_pro'
TABLE_NAME = 'parking_sessions'

# Initialize LangChain components
llm = ChatOpenAI(temperature=0.7, model="gpt-4")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# System message for the AI assistant
SYSTEM_MESSAGE = """You are a helpful assistant for booking parking in malls and restaurants in India. Provide information based on these common rates and terms:

Parking Rates:
- Two-wheeler: ₹20 for first 2 hours, ₹10/hour after that, or ₹100/day
- Four-wheeler: ₹40 for first 2 hours, ₹20/hour after that, or ₹200/day
- Heavy vehicle: ₹80 for first 2 hours, ₹40/hour after that, or ₹400/day

Terms and Conditions:
1. Parking is at owner's risk
2. Lost ticket fee: ₹500
3. No overnight parking for two-wheelers
4. Maximum stay: 24 hours
5. Parking fees are non-refundable

Vehicle Types:
- Two-wheeler: Motorcycles, scooters
- Four-wheeler: Cars, SUVs, vans
- Heavy vehicle: Trucks, buses, tempos

After providing information, ask for the vehicle type, vehicle number, and parking duration to create an entry."""

conversation.memory.chat_memory.add_message(SystemMessage(content=SYSTEM_MESSAGE))

def generate_new_session():
    session_id = str(uuid.uuid4())
    query = f"""
    INSERT INTO `{DATASET_NAME}.{TABLE_NAME}` (session_id, start_time, active)
    VALUES ('{session_id}', CURRENT_TIMESTAMP(), TRUE)
    """
    query_job = bigquery_client.query(query)
    query_job.result()
    logger.info(f"New session created: {session_id}")
    return session_id

def deactivate_session(session_id):
    query = f"""
    UPDATE `{DATASET_NAME}.{TABLE_NAME}`
    SET active = FALSE
    WHERE session_id = '{session_id}'
    """
    bigquery_client.query(query).result()
    logger.info(f"Deactivated session: {session_id}")

def extract_info(message):
    vehicle_types = {
        'two': 'two-wheeler',
        'bike': 'two-wheeler',
        'scooter': 'two-wheeler',
        'four': 'four-wheeler',
        'car': 'four-wheeler',
        'suv': 'four-wheeler',
        'heavy': 'heavy vehicle',
        'truck': 'heavy vehicle',
        'bus': 'heavy vehicle'
    }
    
    vehicle_type = next((vehicle_types[word] for word in message.lower().split() if word in vehicle_types), None)
    
    vehicle_number = re.search(r'\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4}\b', message, re.IGNORECASE)
    vehicle_number = vehicle_number.group() if vehicle_number else None
    
    hours = re.search(r'\b(\d+)\s*(hour|hr|h|hour)\b', message, re.IGNORECASE)
    hours = int(hours.group(1)) if hours else None
    
    return vehicle_type, vehicle_number, hours

def update_parking_entry(session_id, vehicle_type, vehicle_number, parking_hours):
    query = f"""
    UPDATE `{DATASET_NAME}.{TABLE_NAME}`
    SET vehicle_type = '{vehicle_type}', 
        vehicle_number = '{vehicle_number}', 
        parking_hours = {parking_hours}, 
        timestamp = CURRENT_TIMESTAMP(), 
        confirmed = FALSE
    WHERE session_id = '{session_id}'
    """
    query_job = bigquery_client.query(query)
    query_job.result()
    logger.info(f"Updated parking entry for session: {session_id}")

def confirm_parking_entry(session_id):
    query = f"""
    UPDATE `{DATASET_NAME}.{TABLE_NAME}`
    SET confirmed = TRUE, confirmation_timestamp = CURRENT_TIMESTAMP(), active = FALSE
    WHERE session_id = '{session_id}'
    """
    query_job = bigquery_client.query(query)
    query_job.result()
    logger.info(f"Confirmed parking entry for session: {session_id}")

def get_parking_entries(session_id):
    query = f"""
    SELECT * FROM `{DATASET_NAME}.{TABLE_NAME}`
    WHERE session_id = '{session_id}'
    """
    query_job = bigquery_client.query(query)
    results = query_job.result()
    
    entries = []
    for row in results:
        entry = dict(row.items())
        entry['start_time'] = entry['start_time'].isoformat() if entry['start_time'] else None
        entry['timestamp'] = entry['timestamp'].isoformat() if entry['timestamp'] else None
        entry['confirmation_timestamp'] = entry['confirmation_timestamp'].isoformat() if entry['confirmation_timestamp'] else None
        entries.append(entry)
    
    logger.info(f"Retrieved {len(entries)} entries for session: {session_id}")
    return entries

def generate_bot_response(user_message):
    return conversation.predict(input=user_message)

def get_welcome_message():
    return "Welcome! I'm here to help you with parking information for malls and restaurants in India. How can I assist you today?"