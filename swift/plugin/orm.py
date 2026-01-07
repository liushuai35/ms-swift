import os
import re
from typing import TYPE_CHECKING, Dict, List, Union

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest

"""ORM (Outcome Reward Model)s for RLHF training.结果奖励模型，用于对生成的回答进行评分。
“虚假相关性” (Spurious Correlations) 和 “奖励欺骗” (Reward Hacking)。
"""
class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class SegmentFlagReward(ORM):
    """Reward that encourages a single well-placed segment flag token for semantic splitting.

    Components (each in [0,1], then weighted sum):
      - presence_score: exactly one occurrence of the special token.
      - spacing_score: both sides non-empty and at least minimal length.
      - length_score: soft penalty when total text is too long.
      - position_score: if reference provided, closer relative position to reference gets higher score.
      - semantic_score: if reference provided, Jaccard overlap of prefix/suffix tokens vs reference.
    """

    def __init__(self,
                 token: str = '<|segment_flag_token|>',
                 min_prefix_len: int = 5,
                 min_suffix_len: int = 5,
                 max_total_len: int = 4096,
                 position_tolerance: float = 0.1,
                 weights: Dict[str, float] = None):
        self.token = token
        self.min_prefix_len = min_prefix_len
        self.min_suffix_len = min_suffix_len
        self.max_total_len = max_total_len
        self.position_tolerance = position_tolerance
        # weights sum to 1.0 by default
        self.weights = weights or {
            'presence': 0.35,
            'spacing': 0.15,
            'length': 0.1,
            'position': 0.15,
            'semantic': 0.1,
            'fidelity': 0.15,
        }

    @staticmethod
    def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
        if not tokens_a or not tokens_b:
            return 0.0
        set_a, set_b = set(tokens_a), set(tokens_b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return inter / union

    def _split_once(self, text: str):
        if self.token not in text:
            return None, None, None
        parts = text.split(self.token, 1)
        prefix, suffix = parts[0], parts[1]
        return prefix, suffix, text.index(self.token)

    def _fidelity(self, pred: str, ref: str) -> float:
        """Check how much pred matches ref aside from the special token."""
        if not isinstance(ref, str):
            return 0.5  # neutral when no reference
        def norm(s: str) -> List[str]:
            s = s.replace(self.token, ' ')
            s = re.sub(r'\s+', ' ', s).strip().lower()
            return s.split(' ') if s else []
        p_tokens = norm(pred)
        r_tokens = norm(ref)
        if not p_tokens and not r_tokens:
            return 1.0
        if not p_tokens or not r_tokens:
            return 0.0
        inter = len(set(p_tokens) & set(r_tokens))
        union = len(set(p_tokens) | set(r_tokens))
        return inter / union if union > 0 else 0.0

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        # completions: list of generated strings
        # solution: optional list of reference strings containing the expected token position/content
        rewards = []
        sol_list = solution if solution is not None else [None] * len(completions)

        for text, ref in zip(completions, sol_list):
            text = text or ''
            count = text.count(self.token)

            # presence: reward exactly one
            if count == 1:
                presence_score = 1.0
            elif count == 0:
                presence_score = 0.0
            else:
                # multiple occurrences get decayed reward
                presence_score = max(0.0, 1.0 - 0.5 * (count - 1))

            prefix, suffix, pos = self._split_once(text)
            if prefix is None:
                spacing_score = 0.0
                length_score = 1.0 if len(text) <= self.max_total_len else max(0.0, 1.0 - (len(text) - self.max_total_len) / self.max_total_len)
                position_score = 0.0
                semantic_score = 0.0
            else:
                prefix_len = len(prefix.strip())
                suffix_len = len(suffix.strip())
                if prefix_len >= self.min_prefix_len and suffix_len >= self.min_suffix_len:
                    spacing_score = 1.0
                elif prefix_len > 0 and suffix_len > 0:
                    spacing_score = 0.5
                else:
                    spacing_score = 0.0

                total_len = len(text)
                if total_len <= self.max_total_len:
                    length_score = 1.0
                else:
                    length_score = max(0.0, 1.0 - (total_len - self.max_total_len) / self.max_total_len)

                # position relative difference if reference provided and contains token
                position_score = 0.5  # neutral default when token exists
                semantic_score = 0.0
                if isinstance(ref, str) and self.token in ref:
                    ref_pos = ref.index(self.token) / max(1, len(ref))
                    pred_pos = pos / max(1, len(text))
                    diff = abs(pred_pos - ref_pos)
                    position_score = max(0.0, 1.0 - diff / max(self.position_tolerance, 1e-6))

                    # semantic overlap for prefix/suffix vs reference segments
                    ref_prefix, ref_suffix, _ = self._split_once(ref)
                    if ref_prefix is not None:
                        pred_pref_tokens = prefix.strip().split()
                        pred_suff_tokens = suffix.strip().split()
                        ref_pref_tokens = ref_prefix.strip().split()
                        ref_suff_tokens = ref_suffix.strip().split()
                        pref_j = self._jaccard(pred_pref_tokens, ref_pref_tokens)
                        suff_j = self._jaccard(pred_suff_tokens, ref_suff_tokens)
                        semantic_score = 0.5 * (pref_j + suff_j)

            fidelity_score = self._fidelity(text, ref)

            w = self.weights
            reward = (
                w['presence'] * presence_score
                + w['spacing'] * spacing_score
                + w['length'] * length_score
                + w['position'] * position_score
                + w['semantic'] * semantic_score
                + w['fidelity'] * fidelity_score
            )
            # clamp
            reward = max(0.0, min(1.0, reward))
            rewards.append(float(reward))
        return rewards


class SegmentIndexReward(ORM):
    """Reward for predicting split indices (sparse, index-based) instead of emitting the flag token.

    - Model output format (per sample): a comma/space-separated list of integers, e.g. "12, 40, 88".
    - Reference: a list of ints (target split indices) passed via `solution` or in `kwargs['reference_indices']`.
    - Matching: an index p is a hit if there exists r with |p - r| <= tolerance.
    - Scores: F1 for hits, density for |P| vs |R|, position smoothness for hit offsets, gap penalty for very close splits.
    """

    def __init__(self,
                 tolerance: int = 8,
                 min_gap: int = 4,
                 weights: Dict[str, float] = None):
        self.tolerance = tolerance
        self.min_gap = min_gap
        self.weights = weights or {
            'f1': 0.55,
            'density': 0.15,
            'position': 0.2,
            'gap': 0.1,
        }

    @staticmethod
    def _parse_pred_indices(text: str) -> List[int]:
        if text is None:
            return []
        # Accept comma/space separated integers
        parts = re.split(r'[\s,]+', text.strip())
        res = []
        for p in parts:
            if not p:
                continue
            try:
                res.append(int(p))
            except Exception:
                continue
        return res

    @staticmethod
    def _to_int_list(obj) -> List[int]:
        if obj is None:
            return []
        if isinstance(obj, str):
            return SegmentIndexReward._parse_pred_indices(obj)
        try:
            return [int(x) for x in obj]
        except Exception:
            return []

    def _match_counts(self, preds: List[int], refs: List[int]):
        preds_sorted = sorted(preds)
        refs_sorted = sorted(refs)
        hit = 0
        used = set()
        offsets = []
        for p in preds_sorted:
            best = None
            best_d = None
            for i, r in enumerate(refs_sorted):
                if i in used:
                    continue
                d = abs(p - r)
                if d <= self.tolerance and (best_d is None or d < best_d):
                    best = i
                    best_d = d
            if best is not None:
                used.add(best)
                hit += 1
                offsets.append(best_d)
        return hit, len(preds_sorted), len(refs_sorted), offsets

    def _gap_penalty(self, preds: List[int]) -> float:
        if len(preds) < 2:
            return 1.0
        preds_sorted = sorted(preds)
        gaps = [preds_sorted[i+1] - preds_sorted[i] for i in range(len(preds_sorted)-1)]
        close = [g for g in gaps if g < self.min_gap]
        if not gaps:
            return 1.0
        ratio = len(close) / len(gaps)
        return max(0.0, 1.0 - ratio)

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        rewards = []
        # reference indices can be supplied either via solution (list per sample) or kwargs
        ref_all = solution if solution is not None else kwargs.get('reference_indices')
        if ref_all is None:
            ref_all = [None] * len(completions)

        for text, ref in zip(completions, ref_all):
            preds = self._parse_pred_indices(text)
            refs = self._to_int_list(ref)

            hit, p_cnt, r_cnt, offsets = self._match_counts(preds, refs)
            precision = hit / (p_cnt + 1e-6)
            recall = hit / (r_cnt + 1e-6)
            f1 = (2 * precision * recall) / (precision + recall + 1e-6) if (precision + recall) > 0 else 0.0

            # density: prefer |P| close to |R|
            if r_cnt == 0:
                density = 1.0 if p_cnt == 0 else 0.0
            else:
                density = max(0.0, 1.0 - abs(p_cnt - r_cnt) / r_cnt)

            # position: average normalized offset for hits
            if offsets:
                pos_score = sum(max(0.0, 1.0 - d / max(1, self.tolerance)) for d in offsets) / len(offsets)
            else:
                pos_score = 0.0

            gap_score = self._gap_penalty(preds)

            w = self.weights
            reward = (
                w['f1'] * f1
                + w['density'] * density
                + w['position'] * pos_score
                + w['gap'] * gap_score
            )
            reward = max(0.0, min(1.0, reward))
            rewards.append(float(reward))
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'segment_flag': SegmentFlagReward,
    'segment_index': SegmentIndexReward,
}
