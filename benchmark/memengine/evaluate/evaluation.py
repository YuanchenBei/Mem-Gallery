import json
import re
import sys
from typing import List, Dict
from collections import defaultdict
import statistics
import argparse
import regex
import string
from collections import Counter
from bert_score import score
from nltk.stem import PorterStemmer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
from pathlib import Path
from openai import OpenAI
import uuid
import numpy as np
import torch
import random
import os

# Add project root to path for importing default_config
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Define data directories
PROJECT_ROOT = project_root
DATA_DIR = PROJECT_ROOT / "data"
DIALOG_DIR = DATA_DIR / "dialog"
RESULT_DIR = PROJECT_ROOT / "result"


def get_available_datasets():
    """
    Scan the dialog directory to retrieve all available dataset names.

    Returns:
        list: List of dataset names (without .json extension)
    """
    datasets = []
    if DIALOG_DIR.exists():
        for filename in DIALOG_DIR.iterdir():
            if filename.is_file() and filename.name.endswith('.json'):
                # Skip result files and other special files
                if '_results_' in filename.name or '_evaluate_result_' in filename.name or filename.name == 'rename.py':
                    continue
                # Extract dataset name: extract "DatasetName" from "DatasetName.json"
                dataset_name = filename.name.replace('.json', '')
                datasets.append(dataset_name)
    return sorted(datasets)


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

ps = PorterStemmer()


def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Fix normalization logic
def normalize_answer_robust(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        # Replace punctuation with spaces to prevent word concatenation
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answer_universal(s):
    """
    Universal answer normalization function
    - Protect decimal points (3.14)
    - Protect underscores (IMG_001, D1:IMG_001)
    - Replace other punctuation with spaces
    - Remove articles (a, an, the, and)
    """
    # 1. Basic preprocessing: convert to lowercase
    s = str(s).lower()

    # 2. Use fixed placeholders (containing no punctuation to avoid corruption during subsequent punctuation processing)
    # CRITICAL FIX: Previously uuid.uuid4() generated different placeholders each time, making identical answers unmatchable
    dot_placeholder = 'DOTPLACEHOLDER'
    us_placeholder = 'UNDERSCOREPLACEHOLDER'

    # 3. Protect decimal points: match dots surrounded by digits
    s = regex.sub(r'(?<=\d)\.(?=\d)', dot_placeholder, s)

    # 4. Protect underscores: underscores are core features of IDs (e.g., IMG_001)
    s = s.replace('_', us_placeholder)

    # 5. Remove articles (a, an, the, and) - consistent with existing logic
    s = regex.sub(r'\b(a|an|the|and)\b', ' ', s)

    # 6. Handle punctuation: replace all other punctuation with spaces
    exclude = set(string.punctuation)
    s = ''.join(ch if ch not in exclude else ' ' for ch in s)

    # 7. Restore placeholders
    s = s.replace(dot_placeholder, '.')
    s = s.replace(us_placeholder, '_')

    # 8. Fix extra spaces
    s = ' '.join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str):
    """
    Calculate exact match score.

    Function: Compare whether normalized prediction and ground truth answers match exactly.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        bool: True if normalized strings match exactly, False otherwise
    """
    return normalize_answer_universal(prediction) == normalize_answer_universal(ground_truth)


def f1_score(prediction: str, ground_truth: str):
    """
    Calculate F1 score.

    Function: Calculate F1 score based on word overlap after stemming.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        float: F1 score in range [0,1], higher values indicate greater overlap
    """
    norm_p = normalize_answer_universal(prediction)
    norm_g = normalize_answer_universal(ground_truth)
    
    prediction_tokens = [ps.stem(w) for w in norm_p.split()]
    ground_truth_tokens = [ps.stem(w) for w in norm_g.split()]
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    
    return (2 * precision * recall) / (precision + recall)


def bleu_score(prediction: str, ground_truth: str, weights: List[float] = (0.25, 0.25, 0.25, 0.25)):
    """
    Calculate BLEU score.

    Function: Evaluate similarity between predicted and ground truth answers using BLEU metric.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer
    """
    norm_p = normalize_answer_universal(prediction)
    norm_g = normalize_answer_universal(ground_truth)
    
    pred_tokens = norm_p.split()
    ref_tokens = norm_g.split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    chencherry = SmoothingFunction()
    return sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=chencherry.method1)


def bert_score(prediction: str, ground_truth: str):
    """
    Calculate BERT score.

    Function: Use BERT model to compute semantic similarity between predicted and ground truth answers.

    Args:
        prediction (str): The predicted answer
        ground_truth (str): The ground truth answer

    Returns:
        float: BERT F1 score in range [0,1], higher values indicate greater semantic similarity
    """
    prediction = normalize_answer_universal(prediction)
    ground_truth = normalize_answer_universal(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    return max(0, F1[0].item())


def calculate_retrieval_metrics(retrieved_ids: List[str], clue_ids: List[str], k: int = 10) -> Dict[str, float]:
    """
    Calculate retrieval metrics: Precision@K, Recall@K, HitRate@K

    Args:
        retrieved_ids: List of retrieved IDs (strings)
        clue_ids: List of ground truth clue IDs (strings)
        k: Top-K cutoff for evaluation

    Returns:
        dict: Dictionary containing precision_k, recall_k, hitrate_k
    """
    # Normalize to strings and cut to K
    retrieved_ids = [str(x) for x in retrieved_ids][:k]
    # Deduplicate preserving order
    seen = set()
    retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]

    # Normalize ground truth to strings
    if isinstance(clue_ids, list):
        gt_set = set([str(x) for x in clue_ids])
    else:
        gt_set = set()

    R = retrieved_ids
    G = gt_set

    # If both are empty, return zero metrics
    if len(R) == 0 and len(G) == 0:
        return {
            'precision_k': 0.0,
            'recall_k': 0.0,
            'hitrate_k': 0.0
        }

    # Calculate hit count
    hit_rels = sum(1 for x in R if x in G)

    # Precision@K: use actual returned count as denominator
    denom = max(1, len(R))
    prec_k = hit_rels / denom

    # Recall@K: use ground truth count as denominator
    rec_k = (hit_rels / len(G)) if len(G) > 0 else 0.0

    # HitRate@K: binary indicator (1 if any hit, 0 otherwise)
    hitrate_k = 1.0 if hit_rels > 0 else 0.0

    return {
        'precision_k': prec_k,
        'recall_k': rec_k,
        'hitrate_k': hitrate_k
    }


def load_judge_prompt(prompt_path: str) -> str:
    """
    Load the LLM judge prompt template from file.
    
    Args:
        prompt_path: Path to the prompt template file (relative to evaluate directory)
        
    Returns:
        str: The prompt template content
    """
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    full_path = script_dir / prompt_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Prompt template not found at: {full_path}")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_judge_response(response_text: str) -> Dict[str, any]:
    """
    Parse the LLM judge response to extract score and reasoning.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        dict: {'score': float, 'reasoning': str} or None if parsing fails
    """
    try:
        # First, try to find JSON object using a more robust pattern
        # Look for { ... } that contains "score"
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"score"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            # Try to parse the JSON
            result = json.loads(json_str)
            
            # Validate score is in valid range [0, 0.5, 1]
            score = result.get('score', None)
            if score is not None:
                # Allow both int and float, but normalize to float
                score = float(score)
                # Round to nearest valid value (0, 0.5, or 1)
                if score < 0.25:
                    score = 0.0
                elif score < 0.75:
                    score = 0.5
                else:
                    score = 1.0
                result['score'] = score
                # Ensure reasoning exists
                if 'reasoning' not in result:
                    result['reasoning'] = ''
            
            return result
    except (json.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
        pass
    
    # If JSON parsing fails, try to extract score from text patterns
    # Look for patterns like "score": 1 or score: 0.5
    score_patterns = [
        r'"score"\s*:\s*([0-9.]+)',
        r'score\s*:\s*([0-9.]+)',
        r'Score\s*:\s*([0-9.]+)',
    ]
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Normalize to valid values
                if score < 0.25:
                    score = 0.0
                elif score < 0.75:
                    score = 0.5
                else:
                    score = 1.0
                return {
                    'score': score,
                    'reasoning': response_text[:200] if len(response_text) > 200 else response_text
                }
            except ValueError:
                continue
    
    return None


def llm_judge_score(
    question: str,
    ground_truth: str,
    model_output: str,
    client: OpenAI,
    model_name: str,
    prompt_template: str,
    max_retries: int = 3,
    timeout: int = 60,
    delay_base: float = 1.0
) -> Dict[str, any]:
    """
    Call LLM to judge the model output against ground truth.
    
    Args:
        question: The question asked
        ground_truth: The correct answer
        model_output: The model's answer
        client: OpenAI client instance
        model_name: Name of the judge model
        prompt_template: Prompt template string
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        delay_base: Base delay for exponential backoff (seconds)
        
    Returns:
        dict: {'score': float, 'reasoning': str} or None if all retries fail
    """
    # Replace placeholders in template
    prompt = prompt_template.replace('{{question}}', question)
    prompt = prompt.replace('{{ground_truth}}', ground_truth)
    prompt = prompt.replace('{{model_output}}', model_output)
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=timeout
            )
            
            response_text = response.choices[0].message.content.strip()
            result = parse_judge_response(response_text)
            
            if result is not None:
                return result
            else:
                # Parsing failed, retry
                last_error = ValueError(f"Failed to parse judge response: {response_text[:100]}")
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = delay_base * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                # Last attempt failed, but we'll retry until success as requested
                # So we raise the exception to be caught by the caller
                raise Exception(f"LLM judge failed after {max_retries} attempts: {str(last_error)}")
    
    # If we get here, all retries exhausted
    raise Exception(f"LLM judge failed after {max_retries} attempts: {str(last_error)}")


def load_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def aggregate_overall_metrics(individual_results: List[Dict], eval_retrieval_metrics: bool = False) -> Dict:
    """
    Aggregate metrics across multiple datasets to compute overall statistics.
    
    Args:
        individual_results: List of result dictionaries from individual dataset evaluations
        eval_retrieval_metrics: Whether retrieval metrics were evaluated
        
    Returns:
        dict: Aggregated metrics with overall statistics
    """
    # Initialize aggregators
    all_f1 = []
    all_bleu = []
    all_bleu_1 = []
    all_bleu_2 = []
    all_em = []
    all_llm_judge = []
    all_precision_k = []
    all_recall_k = []
    all_hitrate_k = []
    
    # Aggregate category-wise metrics
    category_f1 = defaultdict(list)
    category_bleu = defaultdict(list)
    category_bleu_1 = defaultdict(list)
    category_bleu_2 = defaultdict(list)
    category_em = defaultdict(list)
    category_llm_judge = defaultdict(list)
    category_precision_k = defaultdict(list)
    category_recall_k = defaultdict(list)
    category_hitrate_k = defaultdict(list)
    
    # Collect all metrics from individual results
    for result in individual_results:
        # Overall metrics
        if 'overall_f1' in result:
            all_f1.extend(result['overall_f1'])
        if 'overall_bleu' in result:
            all_bleu.extend(result['overall_bleu'])
        if 'overall_bleu_1' in result:
            all_bleu_1.extend(result['overall_bleu_1'])
        if 'overall_bleu_2' in result:
            all_bleu_2.extend(result['overall_bleu_2'])
        if 'overall_em' in result:
            all_em.extend(result['overall_em'])
        if 'overall_llm_judge' in result:
            all_llm_judge.extend(result['overall_llm_judge'])
        if eval_retrieval_metrics:
            if 'overall_precision_k' in result:
                all_precision_k.extend(result['overall_precision_k'])
            if 'overall_recall_k' in result:
                all_recall_k.extend(result['overall_recall_k'])
            if 'overall_hitrate_k' in result:
                all_hitrate_k.extend(result['overall_hitrate_k'])
        
        # Category-wise metrics
        for category, scores in result.get('category_f1', {}).items():
            category_f1[category].extend(scores)
        for category, scores in result.get('category_bleu', {}).items():
            category_bleu[category].extend(scores)
        for category, scores in result.get('category_bleu_1', {}).items():
            category_bleu_1[category].extend(scores)
        for category, scores in result.get('category_bleu_2', {}).items():
            category_bleu_2[category].extend(scores)
        for category, scores in result.get('category_em', {}).items():
            category_em[category].extend(scores)
        if 'category_llm_judge' in result:
            for category, scores in result['category_llm_judge'].items():
                category_llm_judge[category].extend(scores)
        if eval_retrieval_metrics:
            for category, scores in result.get('category_precision_k', {}).items():
                category_precision_k[category].extend(scores)
            for category, scores in result.get('category_recall_k', {}).items():
                category_recall_k[category].extend(scores)
            for category, scores in result.get('category_hitrate_k', {}).items():
                category_hitrate_k[category].extend(scores)
    
    # Compute overall averages
    avg_overall_metrics = {}
    if all_f1:
        avg_overall_metrics['F1'] = statistics.mean(all_f1)
    if all_bleu:
        avg_overall_metrics['BLEU'] = statistics.mean(all_bleu)
    if all_bleu_1:
        avg_overall_metrics['BLEU_1'] = statistics.mean(all_bleu_1)
    if all_bleu_2:
        avg_overall_metrics['BLEU_2'] = statistics.mean(all_bleu_2)
    if all_em:
        avg_overall_metrics['EM'] = statistics.mean(all_em)
    if all_llm_judge:
        avg_overall_metrics['LLM_JUDGE'] = statistics.mean(all_llm_judge)
    if eval_retrieval_metrics:
        if all_precision_k:
            avg_overall_metrics['Precision@K'] = statistics.mean(all_precision_k)
        if all_recall_k:
            avg_overall_metrics['Recall@K'] = statistics.mean(all_recall_k)
        if all_hitrate_k:
            avg_overall_metrics['HitRate@K'] = statistics.mean(all_hitrate_k)
    
    # Compute category-wise averages
    avg_category_metrics = defaultdict(dict)
    for category, scores in category_f1.items():
        if scores:
            avg_category_metrics[category]['F1'] = statistics.mean(scores)
    for category, scores in category_bleu.items():
        if scores:
            avg_category_metrics[category]['BLEU'] = statistics.mean(scores)
    for category, scores in category_bleu_1.items():
        if scores:
            avg_category_metrics[category]['BLEU_1'] = statistics.mean(scores)
    for category, scores in category_bleu_2.items():
        if scores:
            avg_category_metrics[category]['BLEU_2'] = statistics.mean(scores)
    for category, scores in category_em.items():
        if scores:
            avg_category_metrics[category]['EM'] = statistics.mean(scores)
    for category, scores in category_llm_judge.items():
        if scores:
            avg_category_metrics[category]['LLM_JUDGE'] = statistics.mean(scores)
    if eval_retrieval_metrics:
        for category, scores in category_precision_k.items():
            if scores:
                avg_category_metrics[category]['Precision@K'] = statistics.mean(scores)
        for category, scores in category_recall_k.items():
            if scores:
                avg_category_metrics[category]['Recall@K'] = statistics.mean(scores)
        for category, scores in category_hitrate_k.items():
            if scores:
                avg_category_metrics[category]['HitRate@K'] = statistics.mean(scores)
    
    return {
        'avg_overall_metrics_final': avg_overall_metrics,
        'avg_category_metrics_final': dict(avg_category_metrics)
    }

def main(file_path: str, enable_llm_judge: bool = False, judge_config: Dict = None, eval_retrieval_metrics: bool = False, eval_topk: int = 5):
    """Main function to calculate average F1 scores per category."""
    # Load data from file
    data = load_data(file_path)

    # Initialize category dictionary
    # Dataset uses string categories: IE, VTTL, TR, VR, AR, CD, IS, CR, MR
    category_f1 = defaultdict(list)
    overall_f1 = []
    category_bleu = defaultdict(list)
    overall_bleu = []
    category_bleu_1 = defaultdict(list)
    overall_bleu_1 = []
    category_bleu_2 = defaultdict(list)
    overall_bleu_2 = []
    category_em = defaultdict(list)
    overall_em = []

    # LLM Judge aggregations (only if enabled)
    category_llm_judge = defaultdict(list)
    overall_llm_judge = []
    llm_judge_details = []  # Store detailed results for each sample

    # Retrieval metrics aggregations (only if enabled)
    category_precision_k = defaultdict(list)
    overall_precision_k = []
    category_recall_k = defaultdict(list)
    overall_recall_k = []
    category_hitrate_k = defaultdict(list)
    overall_hitrate_k = []

    avg_category_metrics_final = defaultdict(list)
    avg_overall_metrics_final = defaultdict(list)

    # Initialize LLM judge if enabled
    judge_client = None
    judge_prompt_template = None
    judge_model_name = None
    if enable_llm_judge:
        if judge_config is None:
            raise ValueError("judge_config must be provided when enable_llm_judge is True")
        
        judge_client = OpenAI(
            api_key=judge_config.get('api_key'),
            base_url=judge_config.get('base_url')
        )
        judge_model_name = judge_config.get('name', 'gpt-4o-mini')
        
        # Load prompt template
        prompt_path = judge_config.get('prompt_path', 'llm_judge.txt')
        judge_prompt_template = load_judge_prompt(prompt_path)
        
        print(f"\nLLM Judge enabled using model: {judge_model_name}")
        print("Loading prompt template from:", prompt_path)

    # Calculate scores for each sample
    for sample in tqdm(data, total=len(data)):
        category = sample['category']
        # Skip if category is empty or None
        if not category:
            continue

        system_answer = str(sample['system_answer'])
        # For IS (Image Search) category, handle multiple answers separated by commas
        # if category == 'IS':
        #     system_answer = system_answer.split(',')[0].strip()

        original_answer = str(sample['original_answer'])
        question = str(sample.get('question', ''))

        f1_metric = f1_score(system_answer, original_answer)
        bleu_metric = bleu_score(system_answer, original_answer)
        bleu_metric_1 = bleu_score(system_answer, original_answer, weights=(1, 0, 0, 0))
        bleu_metric_2 = bleu_score(system_answer, original_answer, weights=(0.5, 0.5, 0, 0))
        em_metric = exact_match_score(system_answer, original_answer)

        # Append metrics to the corresponding category
        category_f1[category].append(f1_metric)
        overall_f1.append(f1_metric)
        category_bleu[category].append(bleu_metric)
        overall_bleu.append(bleu_metric)
        category_bleu_1[category].append(bleu_metric_1)
        overall_bleu_1.append(bleu_metric_1)
        category_bleu_2[category].append(bleu_metric_2)
        overall_bleu_2.append(bleu_metric_2)
        category_em[category].append(em_metric)
        overall_em.append(em_metric)

        # Evaluate retrieval metrics (if enabled and data available)
        if eval_retrieval_metrics:
            retrieved_ids = sample.get("retrieved_ids", [])
            clue_ids = sample.get("clue", [])

            if retrieved_ids is not None and clue_ids is not None:
                retrieval_metrics = calculate_retrieval_metrics(retrieved_ids, clue_ids, k=eval_topk)

                category_precision_k[category].append(retrieval_metrics['precision_k'])
                overall_precision_k.append(retrieval_metrics['precision_k'])
                category_recall_k[category].append(retrieval_metrics['recall_k'])
                overall_recall_k.append(retrieval_metrics['recall_k'])
                category_hitrate_k[category].append(retrieval_metrics['hitrate_k'])
                overall_hitrate_k.append(retrieval_metrics['hitrate_k'])

        # Evaluate with LLM Judge
        if enable_llm_judge:
            # Retry until successful
            while True:
                try:
                    judge_result = llm_judge_score(
                        question=question,
                        ground_truth=original_answer,
                        model_output=system_answer,
                        client=judge_client,
                        model_name=judge_model_name,
                        prompt_template=judge_prompt_template,
                        max_retries=judge_config.get('max_retries', 5),
                        timeout=judge_config.get('timeout', 60)
                    )
                    
                    judge_score = judge_result['score']
                    judge_reasoning = judge_result.get('reasoning', '')
                    
                    category_llm_judge[category].append(judge_score)
                    overall_llm_judge.append(judge_score)

                    # Store detailed results
                    llm_judge_details.append({
                        'sample_id': sample.get('sample_id', ''),
                        'category': category,
                        'question': question,
                        'ground_truth': original_answer,
                        'model_output': system_answer,
                        'score': judge_score,
                        'reasoning': judge_reasoning
                    })
                    break  # Successful, exit retry loop

                except Exception as e:
                    print(f"\nWarning: LLM judge failed for sample {sample.get('sample_id', 'unknown')}: {e}")
                    print("Retrying...")
                    time.sleep(2)  # Wait before retrying
                    # Continue the while loop to retry

    # Calculate and print average F1 scores for each category
    print("\n" + "="*60)
    print("Category-wise Results:")
    print("="*60)
    for category, f1_scores in sorted(category_f1.items()):
        avg_f1 = statistics.mean(f1_scores)
        avg_category_metrics_final[category].append({"F1": avg_f1})
        print(f"Category {category}: Average F1 Score = {avg_f1:.4f} (n={len(f1_scores)})")

    for category, bleu_scores in sorted(category_bleu.items()):
        avg_bleu = statistics.mean(bleu_scores)
        avg_category_metrics_final[category].append({"BLEU": avg_bleu})
        print(f"Category {category}: Average BLEU Score = {avg_bleu:.4f} (n={len(bleu_scores)})")

    for category, bleu_1_scores in sorted(category_bleu_1.items()):
        avg_bleu_1 = statistics.mean(bleu_1_scores)
        avg_category_metrics_final[category].append({"BLEU_1": avg_bleu_1})
        print(f"Category {category}: Average BLEU_1 Score = {avg_bleu_1:.4f} (n={len(bleu_1_scores)})")

    for category, bleu_2_scores in sorted(category_bleu_2.items()):
        avg_bleu_2 = statistics.mean(bleu_2_scores)
        avg_category_metrics_final[category].append({"BLEU_2": avg_bleu_2})
        print(f"Category {category}: Average BLEU_2 Score = {avg_bleu_2:.4f} (n={len(bleu_2_scores)})")
    
    for category, em_scores in sorted(category_em.items()):
        avg_em = statistics.mean(em_scores)
        avg_category_metrics_final[category].append({"EM": avg_em})
        print(f"Category {category}: Average EM Score = {avg_em:.4f} (n={len(em_scores)})")

    # Print LLM Judge results if enabled
    if enable_llm_judge:
        for category, judge_scores in sorted(category_llm_judge.items()):
            avg_judge = statistics.mean(judge_scores)
            avg_category_metrics_final[category].append({"LLM_JUDGE": avg_judge})
            print(f"Category {category}: Average LLM Judge Score = {avg_judge:.4f} (n={len(judge_scores)})")

    # Print retrieval metrics results if enabled
    if eval_retrieval_metrics:
        for category, prec_scores in sorted(category_precision_k.items()):
            avg_prec = statistics.mean(prec_scores)
            avg_category_metrics_final[category].append({"Precision@K": avg_prec})
            print(f"Category {category}: Average Precision@{eval_topk} = {avg_prec:.4f} (n={len(prec_scores)})")
        
        for category, rec_scores in sorted(category_recall_k.items()):
            avg_rec = statistics.mean(rec_scores)
            avg_category_metrics_final[category].append({"Recall@K": avg_rec})
            print(f"Category {category}: Average Recall@{eval_topk} = {avg_rec:.4f} (n={len(rec_scores)})")
        
        for category, hr_scores in sorted(category_hitrate_k.items()):
            avg_hr = statistics.mean(hr_scores)
            avg_category_metrics_final[category].append({"HitRate@K": avg_hr})
            print(f"Category {category}: Average HitRate@{eval_topk} = {avg_hr:.4f} (n={len(hr_scores)})")

    # Calculate and print average scores for overall
    print("\n" + "="*60)
    print("Overall Results:")
    print("="*60)
    overall_avg_f1 = statistics.mean(overall_f1)
    avg_overall_metrics_final["F1"] = overall_avg_f1
    print(f"Overall on all categories: Average F1 Score = {overall_avg_f1:.4f} (n={len(overall_f1)})")
    
    overall_avg_bleu = statistics.mean(overall_bleu)
    avg_overall_metrics_final["BLEU"] = overall_avg_bleu
    print(f"Overall on all categories: Average BLEU Score = {overall_avg_bleu:.4f} (n={len(overall_bleu)})")
    
    overall_avg_bleu_1 = statistics.mean(overall_bleu_1)
    avg_overall_metrics_final["BLEU_1"] = overall_avg_bleu_1
    print(f"Overall on all categories: Average BLEU_1 Score = {overall_avg_bleu_1:.4f} (n={len(overall_bleu_1)})")

    overall_avg_bleu_2 = statistics.mean(overall_bleu_2)
    avg_overall_metrics_final["BLEU_2"] = overall_avg_bleu_2
    print(f"Overall on all categories: Average BLEU_2 Score = {overall_avg_bleu_2:.4f} (n={len(overall_bleu_2)})")

    overall_avg_em = statistics.mean(overall_em)
    avg_overall_metrics_final["EM"] = overall_avg_em
    print(f"Overall on all categories: Average EM Score = {overall_avg_em:.4f} (n={len(overall_em)})")

    # Print overall LLM Judge result if enabled
    if enable_llm_judge:
        overall_avg_judge = statistics.mean(overall_llm_judge)
        avg_overall_metrics_final["LLM_JUDGE"] = overall_avg_judge
        print(f"Overall on all categories: Average LLM Judge Score = {overall_avg_judge:.4f} (n={len(overall_llm_judge)})")

    # Print overall retrieval metrics result if enabled
    if eval_retrieval_metrics:
        if len(overall_precision_k) > 0:
            overall_avg_prec = statistics.mean(overall_precision_k)
            avg_overall_metrics_final["Precision@K"] = overall_avg_prec
            print(f"Overall on all categories: Average Precision@{eval_topk} = {overall_avg_prec:.4f} (n={len(overall_precision_k)})")
        
        if len(overall_recall_k) > 0:
            overall_avg_rec = statistics.mean(overall_recall_k)
            avg_overall_metrics_final["Recall@K"] = overall_avg_rec
            print(f"Overall on all categories: Average Recall@{eval_topk} = {overall_avg_rec:.4f} (n={len(overall_recall_k)})")
        
        if len(overall_hitrate_k) > 0:
            overall_avg_hr = statistics.mean(overall_hitrate_k)
            avg_overall_metrics_final["HitRate@K"] = overall_avg_hr
            print(f"Overall on all categories: Average HitRate@{eval_topk} = {overall_avg_hr:.4f} (n={len(overall_hitrate_k)})")

        avg_overall_metrics_final["Eval@K"] = eval_topk

    # Build return dictionary
    result_dict = {
        "avg_category_metrics_final": avg_category_metrics_final,
        "avg_overall_metrics_final": avg_overall_metrics_final,
        "category_f1": {k: v for k, v in category_f1.items()},
        "overall_f1": overall_f1,
        "category_bleu": {k: v for k, v in category_bleu.items()},
        "overall_bleu": overall_bleu,
        "category_bleu_1": {k: v for k, v in category_bleu_1.items()},
        "overall_bleu_1": overall_bleu_1,
        "category_bleu_2": {k: v for k, v in category_bleu_2.items()},
        "overall_bleu_2": overall_bleu_2,
        "category_em": {k: v for k, v in category_em.items()},
        "overall_em": overall_em
    }
    
    # Add LLM Judge results if enabled
    if enable_llm_judge:
        result_dict["category_llm_judge"] = {k: v for k, v in category_llm_judge.items()}
        result_dict["overall_llm_judge"] = overall_llm_judge
        # result_dict["llm_judge_details"] = llm_judge_details  # Detailed results for each sample

    # Add retrieval metrics results if enabled
    if eval_retrieval_metrics:
        result_dict["category_precision_k"] = {k: v for k, v in category_precision_k.items()}
        result_dict["overall_precision_k"] = overall_precision_k
        result_dict["category_recall_k"] = {k: v for k, v in category_recall_k.items()}
        result_dict["overall_recall_k"] = overall_recall_k
        result_dict["category_hitrate_k"] = {k: v for k, v in category_hitrate_k.items()}
        result_dict["overall_hitrate_k"] = overall_hitrate_k
    
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', default='qwen2-5-vl-7b', choices=["qwen2-5-vl-3b", "qwen2-5-vl-7b", "gpt-4o-mini", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4.1-nano"])
    parser.add_argument('--memory_name', default='MMMemory',
                        choices=["FUMemory", "MMFUMemory", "STMemory", "LTMemory", "GAMemory", "MGMemory", "RFMemory", "MMMemory", "NGMemory", "AUGUSTUSMemory", "UniversalRAGMemory", "MemoryOS", "A-mem"])
    parser.add_argument('--data_name', default=None, help='Dataset name (e.g., Dog_Behavior_Research_Academic_Life). If not specified and --all_datasets is used, will evaluate all datasets.')
    parser.add_argument('--all_datasets', action="store_true", help='Evaluate all available datasets and compute overall statistics')
    # Retrieval metrics arguments
    parser.add_argument('--eval_retrieval_metrics', action="store_true", help='Evaluate retrieval metrics (Recall@K, HitRate@K, Precision@K)')
    parser.add_argument('--eval_topk', type=int, default=10, help='Top-K cutoff for retrieval metrics')
    # LLM Judge arguments
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--enable_llm_judge', action="store_true", help='Enable LLM-as-a-judge evaluation')
    parser.add_argument('--judge_model', default=None, help='Judge model name (overrides default config)')
    parser.add_argument('--judge_api_key', default=None, help='Judge API key (overrides default config)')
    parser.add_argument('--judge_api_base', default=None, help='Judge API base URL (overrides default config)')
    args = parser.parse_args()

    seed_everything(args.seed)

    # Feature string for file naming (empty for now, can be extended in the future)
    feature_str = ''

    # Load LLM judge configuration if enabled
    judge_config = None
    if args.enable_llm_judge:
        from default_config.DefaultEvalConfig import DEFAULT_LLM_JUDGE_CONFIG
        judge_config = DEFAULT_LLM_JUDGE_CONFIG.copy()

        # Override with command-line arguments if provided
        if args.judge_model:
            judge_config['name'] = args.judge_model
        if args.judge_api_key:
            judge_config['api_key'] = args.judge_api_key
        if args.judge_api_base:
            judge_config['base_url'] = args.judge_api_base

        # Build model name based on short name (similar to run_mmlongbench.py)
        judge_model_name = judge_config.get('name', 'qwen2-5-vl-7b')
        judge_api_base = judge_config.get('base_url', '') # setting base url
        judge_api_key = judge_config.get('api_key', '') # seeting your api key

        # Only convert if it's a short name (not already a path)
        if not judge_model_name.startswith('/') and not '/' in judge_model_name:
            # Convert short model name to full path for local VLLM API
            if judge_model_name in ['qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen-2.5-72b-instruct']:
                if 'localhost' in judge_api_base or '127.0.0.1' in judge_api_base:
                    judge_config['name'] = '' # setting your judge model path
            elif judge_model_name == 'gpt-4o-mini':
                if 'openrouter' in judge_api_base:
                    judge_config['name'] = 'openai/gpt-4o-mini'
            # For gemini-2.5-flash, the name stays as is

        # Set prompt path
        judge_config['prompt_path'] = 'llm_judge.txt'

    # Determine which datasets to evaluate
    if args.all_datasets:
        datasets = get_available_datasets()
        if not datasets:
            print("No datasets found in dialog directory!")
            # return
        print(f"Found {len(datasets)} datasets. Evaluating all datasets...")
        print(f"Datasets: {', '.join(datasets)}")
    else:
        if args.data_name is None:
            print("Error: Either --data_name or --all_datasets must be specified")
            # return
        datasets = [args.data_name]

    # Store individual results for overall aggregation
    all_individual_results = []

    # Evaluate each dataset
    for data_name in datasets:
        data_name_underscore = data_name  # Keep original name for file naming
        
        print(f"\n{'='*80}")
        print(f"Evaluating dataset: {data_name}")
        print(f"{'='*80}")
        
        # Build file path with feature flags
        result_dir = RESULT_DIR / args.llm_name / args.memory_name
        file_path = result_dir / f"{data_name_underscore}_results{feature_str}.json"
        
        if not file_path.exists():
            print(f"Warning: Result file not found: {file_path}")
            print("Skipping this dataset...")
            continue
        
        print(f"Loading results from: {file_path}")
        
        try:
            evaluate_result = main(
                str(file_path), 
                enable_llm_judge=args.enable_llm_judge, 
                judge_config=judge_config,
                eval_retrieval_metrics=args.eval_retrieval_metrics,
                eval_topk=args.eval_topk
            )
            
            # Store individual result with dataset name
            evaluate_result['dataset_name'] = data_name
            all_individual_results.append(evaluate_result)
            
            # Save individual result
            try:
                output_path = result_dir / f"{data_name_underscore}_evaluate_result{feature_str}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(evaluate_result, f, ensure_ascii=False, indent=2)
                print(f"Individual evaluation results saved to: {output_path}")
            except Exception as e:
                print(f"Error saving individual result: {e}")
                
        except Exception as e:
            print(f"Error evaluating dataset {data_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


    # Compute overall statistics if evaluating multiple datasets
    if args.all_datasets and len(all_individual_results) > 1:
        print(f"\n{'='*80}")
        print("Computing Overall Statistics Across All Datasets")
        print(f"{'='*80}")

        # Aggregate all metrics across datasets
        overall_aggregated = aggregate_overall_metrics(all_individual_results, eval_retrieval_metrics=args.eval_retrieval_metrics)

        # Print overall results
        print("\n" + "="*80)
        print("Overall Results Across All Datasets:")
        print("="*80)
        for metric_name, metric_value in sorted(overall_aggregated['avg_overall_metrics_final'].items()):
            if isinstance(metric_value, (int, float)):
                print(f"Overall {metric_name}: {metric_value:.4f}")

        # Save overall results
        overall_result = {
            'individual_results': all_individual_results,
            'overall_statistics': overall_aggregated,
            'num_datasets': len(all_individual_results),
            'datasets': datasets
        }
        
        try:
            result_dir = RESULT_DIR / args.llm_name / args.memory_name
            overall_output_path = result_dir / f"ALL_evaluate_result{feature_str}.json"
            with open(overall_output_path, "w", encoding="utf-8") as f:
                json.dump(overall_result, f, ensure_ascii=False, indent=2)
            print(f"\nOverall evaluation results saved to: {overall_output_path}")
        except Exception as e:
            print(f"Error saving overall result: {e}")
    elif args.all_datasets:
        print(f"\nOnly {len(all_individual_results)} dataset(s) evaluated. Skipping overall statistics.")
