import logging,json, random, string, re, sys
from unicodedata import category
from openai import OpenAI
from langchain.prompts import PromptTemplate
sys.path.append('..')
from memengine import MemoryConfig
from memengine import FUMemory, STMemory, LTMemory, GAMemory, MMMemory, NGMemory, AUGUSTUSMemory, UniversalRAGMemory, MMFUMemory
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY, DEFAULT_LTMEMORY, DEFAULT_STMEMORY, DEFAULT_GAMEMORY, DEFAULT_NGMEMORY, DEFAULT_AUGUSTUSMEMORY, DEFAULT_UNIVERSALRAGMEMORY, DEFAULT_MMFUMEMORY
from default_config.DefaultMMMemoryConfig import DEFAULT_MMMEMORY  
from memengine import MGMemory, RFMemory
from default_config.DefaultMemoryConfig import DEFAULT_MGMEMORY, DEFAULT_RFMEMORY
import time
from tqdm import tqdm
import os
import argparse
import base64
import warnings
import numpy as np
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

# Get parent directory of script location (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

# Define relative path constants
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DIALOG_DIR = os.path.join(DATA_DIR, "dialog")
IMAGE_DIR = os.path.join(DATA_DIR, "image")
#RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result_debug")
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompt")


def get_available_datasets():
    """
    Scan dialog directory to get all available dataset names

    Returns:
        list: List of dataset names (without .json suffix)
    """
    datasets = []
    if os.path.exists(DIALOG_DIR):
        for filename in os.listdir(DIALOG_DIR):
            if filename.endswith('.json'):
                # Skip result files and other special files
                if '_results_' in filename or '_evaluate_result_' in filename:
                    continue
                # Extract dataset name: extract "DatasetName" from "DatasetName.json"
                dataset_name = filename.replace('.json', '')
                datasets.append(dataset_name)
    return sorted(datasets)


# Prompt file cache
_PROMPT_CACHE = {}

def load_prompt_file(filename):
    """
    Read prompt file from prompt directory with caching mechanism

    Args:
        filename: Prompt filename (e.g., "sys_prompt.txt")

    Returns:
        str: Prompt file content, returns empty string if file does not exist
    """
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    
    prompt_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            _PROMPT_CACHE[filename] = content
            return content
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_path}")
        _PROMPT_CACHE[filename] = ""
        return ""
    except Exception as e:
        print(f"Error loading prompt file {filename}: {e}")
        _PROMPT_CACHE[filename] = ""
        return ""

# Preload all prompt files
def _load_all_prompts():
    """Preload all prompt files when module loads"""
    prompt_files = ["sys_prompt.txt", "ar_prompt.txt", "cd_prompt.txt", "vs_prompt.txt"]
    for filename in prompt_files:
        load_prompt_file(filename)

# Execute preloading
_load_all_prompts()


# Load SystemPrompt from file
SystemPrompt = load_prompt_file("sys_prompt.txt")
if not SystemPrompt:
    # If file does not exist, use default value as fallback
    SystemPrompt = """Your task is to answer questions in a concise manner with the help of memory content.
When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"Mental health.\", as this is more concise.
"""

TextMsgPrompt = """
The retrieved memory contents are as follows:

{memory_context}
""" 

MsgStartPromptWOMemory = """
The retrieved memory contents are as follows:

"""

MMMemoryDialoguePrompt = """
{textual_context}
image:
image_id: {image_id}
image_content:
"""


DialogueAgentPrompt = """
Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} in a concise manner with the help of memory content.
Please only provide the content of the answer, without including introductory phrases like 'answer:'.
For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.
Generate answers primarily concise, yet complete enough to accurately answer the questions.

The current question is as follows:
{observation} {format_constraint}
"""

DialogueAgentPromptImage = """
Here is the attached image of the question:
"""

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


def encode_image(image_path):
    """
    Encode image to base64, supports multiple path formats
    - Absolute path (use directly)
    - Relative path (resolve based on IMAGE_DIR)
    - Relative path format: ../image/DatasetName/file.jpg
    """
    # If already absolute path, use directly
    if os.path.isabs(image_path):
        final_path = image_path
    else:
        # Handle relative path "../image/DatasetName/file.jpg"
        if image_path.startswith("../image/"):
            # Extract relative path part: DatasetName/file.jpg
            rel_path = image_path.replace("../image/", "")
            final_path = os.path.join(IMAGE_DIR, rel_path)
        else:
            # Try to resolve directly based on IMAGE_DIR
            final_path = os.path.join(IMAGE_DIR, image_path)

        # Normalize path
        final_path = os.path.normpath(final_path)
    
    try:
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Image file not found: {final_path} (original path: {image_path})")
        with open(final_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            if not encoded:
                raise ValueError(f"Encoded image is empty for {final_path}")
            return encoded
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except PermissionError as e:
        print(f"Error: Permission denied when reading image {final_path}: {e}")
        raise
    except Exception as e:
        print(f"Error encoding image {image_path} (tried {final_path}): {e}")
        raise


class VLMAgent():
    def __init__(self, model_name=None, seed=None):
        self.client = OpenAI(api_key=OPENAI_APIKEY, base_url=OPENAI_APIBASE)
        self.model_name = model_name or OPENAI_MODEL
        self.seed = seed
        # Check if it's a Gemini series model
        self.is_gemini = 'gemini' in self.model_name.lower()

    def parse_response(self, response):
        """Parse API response with error handling"""
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError(f"API response has no choices. Response: {response}")
        if not hasattr(response.choices[0], 'message'):
            raise ValueError(f"API response choice has no message. Response: {response}")
        if not hasattr(response.choices[0].message, 'content'):
            raise ValueError(f"API response message has no content. Response: {response}")
        return {'result': response.choices[0].message.content}

    def run(self, message_list):
        """Call API with error handling and retry mechanism"""
        # Build API call parameters
        api_params = {
            'model': self.model_name,
            'messages': message_list,
            'temperature': 0.0,  # Ensure deterministic generation
        }

        # Only pass seed parameter when seed is not None (some APIs may not support seed parameter)
        if self.seed is not None and "qwen" in self.model_name.lower():
            api_params['seed'] = self.seed
        
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                parsed_response = self.parse_response(response)
                return parsed_response
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
    
    def fast_run_with_mm_memory(self, memory_dict_list, query_prompt, query_img=None):
        conversation_info_flow = []
        print("memory_dict_list: ", memory_dict_list)

        # Handle empty list case
        if not memory_dict_list or len(memory_dict_list) == 0:
            # If no memory, only send query
            if query_img:
                query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
                user_content = [
                    {"type": "text", "text": query_prompt},
                ]
                if query_img_path:
                    try:
                        user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                    except Exception as e:
                        print(f"Warning: Failed to encode query image at {query_img_path}: {e}. Continuing without image.")
                else:
                    # If entered if query_img branch but no path, this is a data anomaly
                    raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
                
                response = self.run([
                    {"role": "system", "content": SystemPrompt},
                    {"role": "user", "content": user_content}
                ])
            else:
                response = self.run([
                    {"role": "system", "content": SystemPrompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": query_prompt}
                    ]}
                ])
            return response.get('result', '')

        for memory_dict in memory_dict_list:
            if memory_dict.get('image'):
                textual_content = PromptTemplate(
                    input_variables=['textual_context', 'image_id'], 
                    template=MMMemoryDialoguePrompt
                ).format(
                    textual_context=memory_dict.get('text', ''), 
                    image_id=memory_dict.get('image', {}).get('img_id', '')
                )
                conversation_info_flow.append({"type": "text", "text": textual_content})
                # Safely get image path
                image_path = memory_dict.get('image', {}).get('path')
                if image_path:
                    try:
                        conversation_info_flow.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
                    except Exception as e:
                        raise ValueError(f"Warning: Failed to encode image at {image_path}: {e}. Skipping image.")
                else:
                    # If entered if memory_dict.get('image') branch but no path, this is a data anomaly
                    raise ValueError(f"Image data found in memory_dict but path is missing. memory_dict: {memory_dict}")
            else:
                conversation_info_flow.append({"type": "text", "text": memory_dict.get('text')})
        

        if query_img:
            query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
            user_content = [
                {"type": "text", "text": MsgStartPromptWOMemory},
                *conversation_info_flow,
                {"type": "text", "text": query_prompt},
            ]
            if query_img_path:
                try:
                    user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                except Exception as e:
                    raise ValueError(f"Warning: Failed to encode query image at {query_img_path}: {e}. Continuing without image.")
            else:
                # If entered if query_img branch but no path, this is a data anomaly
                raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
            
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": user_content}
            ])
        else:
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": [
                    {"type": "text", "text": MsgStartPromptWOMemory},
                    *conversation_info_flow,
                    {"type": "text", "text": query_prompt}
                ]}
            ])
        return response.get('result', '')

    def fast_run_with_textual_memory(self, text_memory_prompt, query_prompt, query_img=None):
        if query_img:
            query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
            user_content = [
                {"type": "text", "text": text_memory_prompt},
                {"type": "text", "text": query_prompt},
            ]
            if query_img_path:
                try:
                    user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                except Exception as e:
                    print(f"Warning: Failed to encode query image at {query_img_path}: {e}. Continuing without image.")
            else:
                # If entered if query_img branch but no path, this is a data anomaly
                raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
            
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": user_content}
            ])
        else:
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": [
                    {"type": "text", "text": text_memory_prompt},
                    {"type": "text", "text": query_prompt}
                ]}
            ])
        return response.get('result', '')


# ----- Dialogue Agent -----
class DialogueAgent():
    def __init__(self, memory_name, DialogueAgentMemoryConfig, model_name=None, seed=None):
        self.vlm = VLMAgent(model_name=model_name, seed=seed)
        self.memory_name = memory_name
        self.memory = eval(memory_name)(MemoryConfig(DialogueAgentMemoryConfig))
        
        # Get is_multimodal attribute from config, provide default value for backward compatibility
        self.is_multimodal = DialogueAgentMemoryConfig.get('is_multimodal', False)
    
    def reset(self):
        self.memory.reset()
    
    def memory_store(self, message_dict):
        if self.is_multimodal:
            # Multimodal memory: store original image, not caption
            self.memory.store(message_dict)
        else:
            # Textual memory: always use caption (convert image to caption text)
            text_str = message_dict.get('text', '') if isinstance(message_dict, dict) else message_dict
            # Safely get image info
            image_info = message_dict.get('image')
            if image_info:
                img_id = image_info.get('img_id', '') if isinstance(image_info, dict) else ''
                img_caption = image_info.get('caption', '') if isinstance(image_info, dict) else ''
                if img_id or img_caption:
                    text_str += '\nimage:' + '\nimage_id: ' + str(img_id) + '\nimage_caption: ' + str(img_caption)
            if isinstance(message_dict, dict):
                message_dict['text'] = text_str
            self.memory.store(message_dict)

    def memory_recall(self, observation, observation_image=None):
        if self.is_multimodal:
            observation_dict = {
                'text': observation,
                'image': observation_image
            }
            return self.memory.recall(observation_dict)
        else:
            # For non-multimodal memory, add caption if observation_image has one, otherwise skip
            if observation_image:
                caption = observation_image.get('caption')
                if caption:
                    observation += '\nquestion\'s image:' + '\nimage_caption: ' + caption
            return self.memory.recall(observation)

    def response(self, memory_result, observation, speaker_a, speaker_b, observation_image=None, format_constraint=None):
        # Handle different types of memory return
        memory_context = memory_result

        # Build format constraint string (if any)
        format_constraint_str = ""
        if format_constraint:
            format_constraint_str = "\n\n" + format_constraint
        
        query_prompt = PromptTemplate(
                    input_variables=['observation', 'speaker_a', 'speaker_b', 'format_constraint'],
                    template= DialogueAgentPrompt,
                ).format(observation = observation, speaker_a = speaker_a, speaker_b = speaker_b, format_constraint = format_constraint_str)

        if self.is_multimodal:
            res = self.vlm.fast_run_with_mm_memory(memory_context, query_prompt, observation_image)
        else:
            # Other caption-based memories use textual memory
            text_memory_prompt = PromptTemplate(
                input_variables=['memory_context'],
                    template= TextMsgPrompt,
                ).format(memory_context = memory_context)
            res = self.vlm.fast_run_with_textual_memory(text_memory_prompt, query_prompt, observation_image)
        return res


def process_conversation(conversation_data, data_dir=None, character_profile=None):
    """
    Process conversation data into memory system format.
    Handles both text-only and image-containing messages.
    Data format has separate 'user' and 'assistant' fields, and uses 'image_caption'.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    processed = []
    # Dynamically set speaker_a based on character_profile
    if character_profile and character_profile.get("name"):
        speaker_a = f"user ({character_profile.get('name')})"
    else:
        speaker_a = "user"
    speaker_b = "assistant"

    # Data has multi_session_dialogues as a list
    for session_idx, session_data in enumerate(conversation_data):
        session_id = session_data.get("session_id", "")
        session_date = session_data.get("date", "")
        dialogues = session_data.get("dialogues", [])

        for dialog in dialogues:
            # Collect user and assistant information
            user_text = dialog.get('user', '')
            assistant_text = dialog.get('assistant', '')

            # If both are empty, skip
            if not user_text and not assistant_text:
                continue

            # Merge text content
            text_parts = []
            if user_text:
                text_parts.append(f'{speaker_a}: {user_text}')
            if assistant_text:
                text_parts.append(f'{speaker_b}: {assistant_text}')
            
            combined_text = '\n'.join(text_parts)

            # Process image information (using image_caption field)
            img_list = []
            img_caption = []
            img_id_list = []

            if "input_image" in dialog and dialog["input_image"]:
                img_list = dialog["input_image"]
                img_caption = dialog.get("image_caption", [])  # Changed: blip_caption -> image_caption
                img_id_list = dialog.get("image_id", [])

            # Build merged record
            if img_list and len(img_list) > 0:
                try:
                    img_path = img_list[0]
                    # Handle relative path "../image/DatasetName/file.jpg"
                    if not os.path.isabs(img_path):
                        if img_path.startswith("../image/"):
                            # Extract relative path part
                            rel_path = img_path.replace("../image/", "")
                            full_img_path = os.path.join(IMAGE_DIR, rel_path)
                        else:
                            full_img_path = os.path.join(data_dir, img_path)
                    else:
                        full_img_path = img_path
                    
                    img_id = img_id_list[0] if img_id_list and len(img_id_list) > 0 else ""
                    caption = img_caption[0] if img_caption and len(img_caption) > 0 else ""
                    
                    processed.append({
                        'text': combined_text,
                        'image': {
                            'path': full_img_path,
                            'caption': caption,
                            'img_id': img_id
                        },
                        'timestamp': session_date,
                        'dialogue_id': dialog.get('round', '')
                    })
                except (IndexError, TypeError, AttributeError) as e:
                    print(f"Warning: Failed to process image in dialog {dialog.get('round', 'unknown')}: {e}. Storing without image.")
                    processed.append({
                        'text': combined_text,
                        'image': None,
                        'timestamp': session_date,
                        'dialogue_id': dialog.get('round', '')
                    })
            else:
                processed.append({
                    'text': combined_text,
                    'image': None,
                    'timestamp': session_date,
                    'dialogue_id': dialog.get('round', '')
                })

    return processed


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def run_mm_bench(llm_name, memory_name, DialogueAgentMemoryConfig, data_name, save_results, save_efficiency, eval_retrieval_metrics=False, eval_topk=5, model_name=None, seed=None):
    # Format data_name: replace underscore with empty string for file naming (dog_1 -> dog1)
    data_name_underscore = data_name
    
    # Determine memory type: textual memory uses caption, multimodal memory uses original image
    is_multimodal = DialogueAgentMemoryConfig.get('is_multimodal', False)

    output_file = os.path.join(
        RESULT_DIR,
        llm_name,
        memory_name,
        f"{data_name_underscore}_results.json"
    )

    # Extract directory path
    output_dir = os.path.dirname(output_file)

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset
    json_filename = f"{data_name}.json"
    json_path = os.path.join(DIALOG_DIR, json_filename)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {json_filename} dataset from {json_path}")
    except FileNotFoundError:
        print(f"Can't find {json_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    results = []
    
    # Process character_profile and multi_session_dialogues
    memory_agent = DialogueAgent(memory_name, DialogueAgentMemoryConfig, model_name=model_name, seed=seed)
    memory_agent.reset()

    # Get sample_id from character_profile
    character_profile = dataset.get("character_profile", {})
    sample_id = character_profile.get("name", data_name)
    
    conversation_data = dataset.get("multi_session_dialogues", [])  # Use .get() to avoid KeyError
    qa_pairs = dataset.get("human-annotated QAs", [])  # Use .get() to avoid KeyError
    
    # Process conversation data
    processed_dialogs = process_conversation(conversation_data, data_dir=DATA_DIR, character_profile=character_profile)
    
    if not processed_dialogs:
        print(f"No valid conversation data in {sample_id}, skip")
        return

    # Dynamically set speaker_a based on character_profile
    if character_profile and character_profile.get("name"):
        speaker_a = f"user ({character_profile.get('name')})"
    else:
        speaker_a = "user"
    speaker_b = "assistant"
    
    """
    We need to initialize the memory modules here with the historical data
    """
    memory_start = time.time()
    for dialog in tqdm(processed_dialogs, desc="Processing dialogs", total=len(processed_dialogs)):
        try:
            memory_agent.memory_store(dialog)
        except Exception as e:
            print(f"Warning: Failed to store dialog {dialog.get('dialogue_id', 'unknown')}: {e}. Skipping...")
            continue
    memory_duration = time.time() - memory_start if processed_dialogs else 0.0

    # process QA pairs
    qa_count = len(qa_pairs)
    qa_start = time.time()
    for qa_idx, qa in tqdm(enumerate(qa_pairs), desc="Processing QA pairs", total=qa_count):
        question = qa.get("question", "")
        question_image = None
        question_image_caption = None
        
        if qa.get("question_image"):
            question_image_path = qa.get("question_image", "")
            # Handle relative path "../image/DatasetName/file.jpg"
            if not os.path.isabs(question_image_path):
                if question_image_path.startswith("../image/"):
                    rel_path = question_image_path.replace("../image/", "")
                    question_image = os.path.join(IMAGE_DIR, rel_path)
                else:
                    question_image = os.path.join(IMAGE_DIR, question_image_path)
            else:
                question_image = question_image_path

            # Get caption for question_image (field name is image_caption)
            question_image_caption = qa.get("image_caption", None)
        
        original_answer = qa.get("answer", "")
        category = qa.get("point", "")  # Data format uses "point" instead of "category"
        
        # Load format constraint based on category (case insensitive)
        format_constraint = None
        if category:
            category_upper = category.upper()
            if category_upper == "AR":
                format_constraint = load_prompt_file("ar_prompt.txt")
            elif category_upper == "CD":
                format_constraint = load_prompt_file("cd_prompt.txt")
            elif category_upper == "VS":
                format_constraint = load_prompt_file("vs_prompt.txt")
        
        # Handle case where session_id might be a list
        qa_session_id = qa.get("session_id", "")
        if isinstance(qa_session_id, list) and len(qa_session_id) > 0:
            qa_session_id = qa_session_id[0]
        
        if not question:
            continue

        # Retrieve related memory - pass path and caption (if exists)
        try:
            observation_image = None
            if question_image:
                observation_image = {'path': question_image}
                if question_image_caption:
                    observation_image['caption'] = question_image_caption
            
            memory_context = memory_agent.memory_recall(
                question, 
                observation_image
            )
            # Ensure memory_context is not None
            if memory_context is None:
                memory_context = []
                print(f"Warning: Memory context is None for question: {question[:50]}...")
        except Exception as e:
            print(f"Error retrieving memory for question: {question[:50]}... Error: {e}")
            memory_context = []  # Set default value
                
        # Save retrieval information for later evaluation (if enabled)
        retrieved_ids = []
        if eval_retrieval_metrics:
            try:
                retrieved_ids = getattr(memory_agent.memory.recall_op, 'last_retrieved_ids', []) or []
            except AttributeError:
                # recall_op may not exist or may not have last_retrieved_ids attribute
                retrieved_ids = []
            except Exception as e:
                print(f"Warning: Failed to get retrieved_ids: {e}")
                retrieved_ids = []
            
            # Normalize to strings
            try:
                retrieved_ids = [str(x) for x in retrieved_ids]
                # Deduplicate preserving order
                seen = set()
                retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]
            except Exception as e:
                print(f"Warning: Failed to normalize retrieved_ids: {e}")
                retrieved_ids = []
        
        # Get ground truth clue IDs
        clue_ids = qa.get("clue", [])
        if not isinstance(clue_ids, list):
            clue_ids = []

        # Generate the system response with the retrieved memory
        try:
            # Build observation_image for response (containing path and caption)
            response_observation_image = None
            if question_image:
                response_observation_image = {'path': question_image}
                if question_image_caption:
                    response_observation_image['caption'] = question_image_caption
            
            system_answer = memory_agent.response(
                memory_context, 
                question, 
                speaker_a, 
                speaker_b, 
                response_observation_image,
                format_constraint=format_constraint  # Pass format constraint
            )
            # Ensure system_answer is not None
            if system_answer is None:
                system_answer = ""
                print(f"Warning: System answer is None for question: {question[:50]}...")
        except Exception as e:
            print(f"Error generating response for question: {question[:50]}... Error: {e}")
            system_answer = ""  # Set default value to avoid program crash

        # Save result for the current QA pair
        result_item = {
            "sample_id": sample_id,
            "session_id": qa_session_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "question": question,
            "system_answer": system_answer,
            "original_answer": original_answer,
            "category": category,
            "timestamp": get_timestamp(),
        }
        
        # Add retrieval information if enabled
        if eval_retrieval_metrics:
            result_item["retrieved_ids"] = retrieved_ids
            result_item["clue"] = clue_ids
        
        results.append(result_item)
    qa_duration = time.time() - qa_start if qa_pairs else 0.0
        
    # Final saving
    if save_results:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving result: {e}")

    if save_efficiency:
        efficiency_metrics = {
            "sample_id": sample_id,
            "llm_name": llm_name,
            "memory_name": memory_name,
            "is_multimodal": is_multimodal,
            "conversation_turns": len(processed_dialogs),
            "qa_count": qa_count,
            "memory_time_seconds": memory_duration,
            "qa_time_seconds": qa_duration,
            "total_time_seconds": memory_duration + qa_duration,
            "recorded_at": get_timestamp(),
        }
        efficiency_file = os.path.join(
            os.path.dirname(output_file),
            f"{data_name_underscore}_efficiency.json",
        )
        try:
            with open(efficiency_file, "w", encoding="utf-8") as f:
                json.dump(efficiency_metrics, f, ensure_ascii=False, indent=2)
            print(f"Efficiency metrics saved to {efficiency_file}")
        except Exception as e:
            print(f"Error saving efficiency metrics: {e}")


if __name__ == '__main__':
    ############################ get-args ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', default='qwen2-5-vl-7b', choices=['qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen2-5-vl-32b', 'gpt-4o-mini', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'])
    parser.add_argument('--memory_name', default='MMMemory',
                        choices=["FUMemory", "STMemory", "LTMemory", "GAMemory", "MGMemory", "RFMemory", "MMMemory", "MMFUMemory", "NGMemory", "AUGUSTUSMemory", "UniversalRAGMemory"])
    parser.add_argument('--data_name', default=None, help='Dataset name (e.g., Dog_Behavior_Research_Academic_Life). If not specified and --all_datasets is used, will process all datasets.')
    parser.add_argument('--all_datasets', action="store_true", help='Process all available datasets in the dialog directory')
    parser.add_argument('--save_results', action="store_true", help='Save QA performance results JSON')
    parser.add_argument('--save_efficiency', action="store_true", help='Save efficiency metrics JSON')
    parser.add_argument('--eval_retrieval_metrics', action="store_true", help='Evaluate retrieval metrics (mAP@K, Recall@K, HitRate@K, Precision@K)')
    parser.add_argument('--eval_topk', type=int, default=10, help='Top-K cutoff for retrieval metrics')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.llm_name == 'qwen2-5-7b' or args.llm_name == 'qwen2-5-vl-3b' or args.llm_name == 'qwen2-5-vl-7b' or args.llm_name == 'qwen2-5-vl-32b': # flexible to be extended
        # local VLLM API
        OPENAI_APIKEY = '' # [Replace with your API key]
        OPENAI_APIBASE = '' # [Replace with your API base url]
        OPENAI_MODEL = f'' # [Replace with your model path, e.g., xxx/{args.llm_name}]
    elif args.llm_name == 'gpt-4o-mini':
        # Openrouter API
        OPENAI_APIKEY = '' # [Replace with your API key]
        OPENAI_APIBASE = 'https://openrouter.ai/api/v1' # [Replace with your API base url]
        OPENAI_MODEL = 'openai/gpt-4o-mini' # [Replace with your model name, e.g., openai/gpt-4o-mini]
    elif args.llm_name == 'gemini-2.5-flash' or args.llm_name == 'gemini-2.5-flash-lite':
        # Google Gemini API
        OPENAI_APIKEY = '' # [Replace with your API key]
        OPENAI_APIBASE = 'https://generativelanguage.googleapis.com/v1beta/openai/' # [Replace with your API base url]
        OPENAI_MODEL = args.llm_name # [Replace with your model name, e.g., gemini-2.5-flash]   
    else:
        raise ValueError(f"Unsupported LLM name: {args.llm_name}")

    if args.memory_name == 'FUMemory':
        DialogueAgentMemoryConfig = DEFAULT_FUMEMORY
    elif args.memory_name == 'STMemory':
        DialogueAgentMemoryConfig = DEFAULT_STMEMORY
    elif args.memory_name == 'LTMemory':
        DialogueAgentMemoryConfig = DEFAULT_LTMEMORY
    elif args.memory_name == 'GAMemory':
        DialogueAgentMemoryConfig = DEFAULT_GAMEMORY
    elif args.memory_name == 'MGMemory':
        DialogueAgentMemoryConfig = DEFAULT_MGMEMORY
    elif args.memory_name == 'RFMemory':
        DialogueAgentMemoryConfig = DEFAULT_RFMEMORY
    elif args.memory_name == 'MMMemory':  
        DialogueAgentMemoryConfig = DEFAULT_MMMEMORY
    elif args.memory_name == 'MMFUMemory':  
        import copy
        DialogueAgentMemoryConfig = copy.deepcopy(DEFAULT_MMFUMEMORY)
        # Adjust tokens_per_image based on model type
        if args.llm_name in ['qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen2-5-vl-32b']:
            # Qwen series: 256 tokens per image
            DialogueAgentMemoryConfig['recall']['truncation']['tokens_per_image'] = 256
        elif args.llm_name in ['gpt-4o-mini', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']:
            # GPT/Gemini series: 576 tokens per image
            DialogueAgentMemoryConfig['recall']['truncation']['tokens_per_image'] = 576
        print(f"MMFUMemory configured with tokens_per_image={DialogueAgentMemoryConfig['recall']['truncation']['tokens_per_image']} for model {args.llm_name}")
    elif args.memory_name == 'NGMemory':  
        DialogueAgentMemoryConfig = DEFAULT_NGMEMORY
    elif args.memory_name == 'AUGUSTUSMemory':  
        DialogueAgentMemoryConfig = DEFAULT_AUGUSTUSMEMORY
    elif args.memory_name == 'UniversalRAGMemory':  
        DialogueAgentMemoryConfig = DEFAULT_UNIVERSALRAGMEMORY
        print("Using UniversalRAGMemory with dynamic routing (no, document, image)")
    else:
        raise ValueError(f"Unsupported memory name: {args.memory_name}")

    # Determine which datasets to process
    if args.all_datasets:
        datasets = get_available_datasets()
        if not datasets:
            print("No datasets found in dialog directory!")
            sys.exit(1)
        print(f"Found {len(datasets)} datasets. Processing all datasets...")
        print(f"Datasets: {', '.join(datasets)}")
    else:
        if args.data_name is None:
            print("Error: Either --data_name or --all_datasets must be specified")
            sys.exit(1)
        datasets = [args.data_name]
    
    # Process each dataset
    for data_name in datasets:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {data_name}")
        print(f"{'='*80}")
        try:
            run_mm_bench(
                args.llm_name,
                args.memory_name,
                DialogueAgentMemoryConfig,
                data_name,
                args.save_results,
                args.save_efficiency,
                args.eval_retrieval_metrics,
                args.eval_topk,
                model_name=OPENAI_MODEL,
                seed=args.seed,
            )
        except Exception as e:
            print(f"Error processing dataset {data_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if args.all_datasets:
        print(f"\n{'='*80}")
        print(f"Completed processing {len(datasets)} datasets")
        print(f"{'='*80}")
