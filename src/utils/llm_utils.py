# generative agent utils
import os
import numpy as np
import pickle
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional, Any

from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.config import llm_temperature  # Import temperature from config

# Get API keys with better fallback mechanisms
def get_api_key(key_name, file_path=None):
    """Get API key from environment variable or from a file"""
    # First try environment variable
    key = os.getenv(key_name)
    
    # If not found and file_path is provided, try reading from file
    if not key and file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Extract the key value (format: KEY_NAME=value)
            match = re.search(fr'{key_name}=([^\s"]+)', content)
            if match:
                key = match.group(1).strip()
    
    return key

# Initialize clients with API keys
OPENAI_API_KEY = get_api_key('OPENAI_API_KEY', './OPENAI_API_KEY.env')
if not OPENAI_API_KEY:
    # Try looking for OPENAI_API_KEY.env in current directory
    api_key_files = ['./OPENAI_API_KEY.env', './.env', '../OPENAI_API_KEY.env']
    for file_path in api_key_files:
        if os.path.exists(file_path):
            OPENAI_API_KEY = get_api_key('OPENAI_API_KEY', file_path)
            if OPENAI_API_KEY:
                break

if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable or create an OPENAI_API_KEY.env file")

ANTHROPIC_API_KEY = get_api_key('ANTHROPIC_API_KEY')  # Optional
LLAMA_MODEL_PATH = "path/to/llama/model"  # Replace with your path

oai = OpenAI(api_key=OPENAI_API_KEY.strip())
ant = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

class LLMAdapter:
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = llm_temperature, max_tokens: int = 1000) -> str:
        raise NotImplementedError

class OpenAIAdapter(LLMAdapter):
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = llm_temperature, max_tokens: int = 1000) -> str:
        try:
            response = oai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            raise e

class ClaudeAdapter(LLMAdapter):
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = llm_temperature, max_tokens: int = 1000) -> str:
        try:
            response = ant.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude generation error: {e}")
            raise e

class LlamaAdapter(LLMAdapter):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = llm_temperature, max_tokens: int = 1000) -> str:
        try:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = self.pipe(prompt, max_length=max_tokens, temperature=temperature)
            return response[0]['generated_text']
        except Exception as e:
            print(f"Llama generation error: {e}")
            raise e

def get_llm_client(model_type: str) -> LLMAdapter:
    if model_type.startswith("gpt"):
        return OpenAIAdapter()
    elif model_type.startswith("claude"):
        return ClaudeAdapter()
    elif model_type.startswith("llama"):
        return LlamaAdapter()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def gen_completion(messages: List[Dict[str, str]], 
                  model: str = "gpt-4.1-nano", 
                  temperature: float = llm_temperature,
                  max_tokens: int = 1000,
                  max_retries: int = 3,
                  retry_delay: float = 2.0) -> str:
    """
    Generate a completion using the specified model.
    
    Args:
        messages: List of message dictionaries
        model: Model to use for generation
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Generated text
    """
    retry_count = 0
    while retry_count <= max_retries:
        try:
            llm = get_llm_client(model)
            return llm.generate(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Error generating completion after {max_retries} retries: {e}")
                raise e
            else:
                print(f"API call failed (attempt {retry_count}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 1.5

def simple_gen(prompt: str, model: str = "gpt-4.1-nano", temperature: float = llm_temperature) -> str:
    messages = [{"role": "user", "content": prompt}]
    return gen_completion(messages, model, temperature)

# Prompt utilities
def fill_prompt(prompt: str, placeholders: Dict[str, Any]) -> str:
    for placeholder, value in placeholders.items():
        placeholder_tag = f"!<{placeholder.upper()}>!"
        if placeholder_tag in prompt:
            prompt = prompt.replace(placeholder_tag, str(value))
    return prompt

def make_output_format(modules: List[Dict]) -> str:
    output_format = "Output Format:\n{\n"
    for module in modules:
        if 'name' in module and module['name']:
            output_format += f'    "{module["name"].lower()}": "<your response>",\n'
    output_format = output_format.rstrip(',\n') + "\n}"
    return output_format

def modular_instructions(modules: List[Dict]) -> str:
    prompt = ""
    step_count = 0
    for module in modules:
        if 'name' in module:
            step_count += 1
            prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
        else:
            prompt += f"{module['instruction']}\n"
    prompt += "\n"
    prompt += make_output_format(modules)
    return prompt

def parse_json(response: str, target_keys: Optional[List[str]] = None) -> Dict:
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    cleaned_response = response[json_start:json_end].replace('\\"', '"')
    
    try:
        parsed = json.loads(cleaned_response)
        if target_keys:
            parsed = {key: parsed.get(key, "") for key in target_keys}
        return parsed
    except json.JSONDecodeError:
        print("JSON parsing failed. Using regex fallback.")
        print(f"Response: {cleaned_response}")
        parsed = {}
        for key_match in re.finditer(r'"(\w+)":\s*', cleaned_response):
            key = key_match.group(1)
            if target_keys and key not in target_keys:
                continue
            value_start = key_match.end()
            if cleaned_response[value_start] == '"':
                value_match = re.search(r'"(.*?)"(?:,|\s*})', 
                                      cleaned_response[value_start:])
                if value_match:
                    parsed[key] = value_match.group(1)
            elif cleaned_response[value_start] == '{':
                nested_json = re.search(r'(\{.*?\})(?:,|\s*})', 
                                      cleaned_response[value_start:], re.DOTALL)
                if nested_json:
                    try:
                        parsed[key] = json.loads(nested_json.group(1))
                    except json.JSONDecodeError:
                        parsed[key] = {}
            else:
                value_match = re.search(r'([^,}]+)(?:,|\s*})', 
                                      cleaned_response[value_start:])
                if value_match:
                    parsed[key] = value_match.group(1).strip()
        
        if target_keys:
            parsed = {key: parsed.get(key, "") for key in target_keys}
        return parsed

def mod_gen(modules: List[Dict], placeholders: Dict, 
            target_keys: Optional[List[str]] = None,
            model: str = "gpt-4.1-nano") -> Dict:
    prompt = modular_instructions(modules)
    filled = fill_prompt(prompt, placeholders)
    response = simple_gen(filled, model)
    if len(response) == 0:
        print("Error: response was empty")
        return {}
    if target_keys == None:
        target_keys = [module["name"].lower() for module in modules if "name" in module]
    parsed = parse_json(response, target_keys)
    return parsed 