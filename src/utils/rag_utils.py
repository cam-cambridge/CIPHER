import openai
import json
from dotenv import load_dotenv
import os

class ContextManager:
    """
    A class to manage RAG (Retrieval-Augmented Generation) context and embeddings.
    """
    
    def __init__(self, facts_file_path="./prompts/processed_facts_openai.json"):
        """
        Initialize the ContextManager.
        
        Args:
            facts_file_path (str): Path to the processed facts JSON file
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.processed_facts = self.load_processed_facts(facts_file_path)
    
    def generate_RAG_embedding(self, text):
        """
        Generate embedding using OpenAI's API.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            list or None: Embedding vector or None if error occurs
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # Use OpenAI's embedding model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding for text: {text}\nError: {e}")
            return None

    def load_processed_facts(self, file_path):
        """
        Load processed facts from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            dict: Loaded data from JSON file
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def format_data_with_system(self, example, system_message=None, RAG=False):
        """
        Format data with system message for OpenAI API.
        
        Args:
            example (dict): Example data containing question and context
            system_message (str, optional): System message to include
            RAG (bool): Whether to include RAG context
            
        Returns:
            dict: Formatted data for OpenAI API
        """
        formatted_data = {"messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}],},
            {"role": "user", "content": []}
            ]
        }

        formatted_data["messages"][1]["content"].append(
            {
                "type": "text", "text": example["question"]
            }
        )

        if RAG:
            print(example["context"])

            formatted_data["messages"][1]["content"].append(
                {
                    "type": "text", "text": example["context"]
                }
            )

        return formatted_data