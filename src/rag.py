import openai
import json
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

class ContextManager:
    """
    A class to manage RAG (Retrieval-Augmented Generation) context and embeddings.
    """
    
    def __init__(self, facts_file_path="processed_facts_openai.json", cache_dir="./.cache"):
        """
        Initialize the ContextManager.
        
        Args:
            facts_file_path (str): Name of the processed facts JSON file in the HF dataset
            cache_dir (str): Directory to cache the downloaded file
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.processed_facts = self.load_processed_facts(facts_file_path, cache_dir)
    
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

    def load_processed_facts(self, filename, cache_dir="./.cache"):
        """
        Load processed facts from Hugging Face dataset.
        
        Args:
            filename (str): Name of the JSON file in the HF dataset
            cache_dir (str): Directory to cache the downloaded file
            
        Returns:
            list: Loaded data from JSON file
        """
        try:
            # Download the file from Hugging Face dataset
            file_path = hf_hub_download(
                repo_id="cemag/tl-caxton",
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir
            )
            
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            print(f"✅ Successfully loaded {len(data)} processed facts from Hugging Face dataset")
            return data
            
        except Exception as e:
            print(f"❌ Error loading processed facts from Hugging Face: {e}")
            return []

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
    
    def find_relevant_facts(self, question, num_facts=5):
        """
        Find the most relevant facts for a given question using cosine similarity.
        
        Args:
            question (str): The question to find relevant facts for
            num_facts (int): Number of most relevant facts to return
            
        Returns:
            str: String containing the most relevant facts
        """
        # Generate embedding for the question
        embedding = self.generate_RAG_embedding(question)
        if embedding is None:
            return ""
        
        # Calculate similarities
        similarities = []
        for fact in self.processed_facts:
            embedding_array = np.array(embedding)  # Convert to NumPy array
            embedding_reshaped = embedding_array.reshape(1, -1)  # Reshape to 2D
            similarity = cosine_similarity(embedding_reshaped, np.array(fact['embedding']).reshape(1, -1))
            similarities.append(similarity)
        
        # Get top N most relevant facts
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:num_facts]
        relevant_facts = [self.processed_facts[i] for i in top_indices]
        
        # Format as context string
        relevant_facts_string = "Here is some context that might be useful: " + " | ".join([fact['original_fact'] for fact in relevant_facts])
        
        return relevant_facts_string
    
    def add_context_to_examples(self, examples, num_facts=5):
        """
        Add RAG context to a list of examples.
        
        Args:
            examples (list): List of example dictionaries containing questions
            num_facts (int): Number of most relevant facts to add to each example
            
        Returns:
            list: Updated examples with context added
        """
        for example in examples:
            context = self.find_relevant_facts(example["question"], num_facts)
            example['context'] = context
        
        return examples