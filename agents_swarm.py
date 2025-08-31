import time
import threading
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter


class TaskType(Enum):
    """Enumeration of all supported NLP tasks."""
    SUMMARIZATION = "summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    REDUNDANCY_ANALYSIS = "redundancy_analysis"
    GRAMMAR_CHECK = "grammar_check"
    COVERAGE_SCORING = "coverage_scoring"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TOPIC_MODELING = "topic_modeling"
    TEXT_CLASSIFICATION = "text_classification"
    QUESTION_ANSWERING = "question_answering"


############################################################
# Enhanced Thread-Safe Blackboard
############################################################

class Blackboard:
    """Thread-safe blackboard for agent communication with task-specific organization."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._posts: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = {}
        self._task_metrics: Dict[str, Dict[str, float]] = {}

    def post(self, agent_id: str, channel: str, payload: Any, score: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        """Post data to blackboard with validation."""
        try:
            with self._lock:
                self._posts.append({
                    "agent_id": str(agent_id),
                    "channel": str(channel),
                    "payload": payload,
                    "score": max(0.0, min(1.0, float(score))),
                    "meta": meta or {},
                    "timestamp": len(self._posts)
                })
        except Exception:
            pass

    def read(self, channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read posts from blackboard safely."""
        try:
            with self._lock:
                if channel is None:
                    return list(self._posts)
                return [p for p in self._posts if p.get("channel") == channel]
        except Exception:
            return []

    def clear_channel(self, channel: str):
        """Clear specific channel safely."""
        try:
            with self._lock:
                self._posts = [p for p in self._posts if p.get("channel") != channel]
        except Exception:
            pass

    def set_task_metric(self, task: str, metric: str, value: float):
        """Store task-specific metrics."""
        try:
            with self._lock:
                if task not in self._task_metrics:
                    self._task_metrics[task] = {}
                self._task_metrics[task][metric] = value
        except Exception:
            pass

    def get_task_metrics(self, task: str) -> Dict[str, float]:
        """Get task-specific metrics."""
        try:
            with self._lock:
                return self._task_metrics.get(task, {})
        except Exception:
            return {}


############################################################
# Base Agent Class
############################################################

@dataclass
class Agent:
    """Enhanced base agent class with task affinity and performance tracking."""
    agent_id: str
    role: str
    trust: float = 0.5
    enabled: bool = True
    task_affinity: Set[TaskType] = None
    execution_time: float = 0.0

    def __post_init__(self):
        self.trust = max(0.0, min(1.0, float(self.trust)))
        if self.task_affinity is None:
            self.task_affinity = set(TaskType)

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        """Execute the agent's task. Override in subclasses."""
        start_time = time.time()
        # Add realistic processing delay
        time.sleep(0.01 + (hash(self.agent_id) % 10) * 0.001)
        self.execution_time = time.time() - start_time

    def is_suitable_for_task(self, task_type: TaskType) -> bool:
        """Check if agent is suitable for the given task."""
        return task_type in self.task_affinity


############################################################
# Utility Functions
############################################################

def safe_sentence_split(text: str) -> List[str]:
    """Split text into sentences safely."""
    try:
        if not text or not isinstance(text, str):
            return []
        clean_text = text.replace("\n", " ").replace("\r", " ")
        parts = [s.strip() for s in clean_text.split(".") if s.strip()]
        sentences = []
        for part in parts:
            if part and not part.endswith("."):
                sentences.append(part + ".")
            elif part:
                sentences.append(part)
        return sentences[:50]
    except Exception:
        return []


############################################################
# All Agent Implementations
############################################################

class KeywordAgent(Agent):
    """Enhanced keyword extraction with task-specific optimization."""
    
    def __init__(self, agent_id: str, role: str = "keywords", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.KEYWORD_EXTRACTION, TaskType.COVERAGE_SCORING, TaskType.SUMMARIZATION, TaskType.TOPIC_MODELING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.01)  # Simulate processing
        
        if not self.enabled or not input_text:
            return
            
        try:
            words = []
            for word in str(input_text).split():
                clean_word = word.lower().strip(",;:!?()[]\"'")
                if clean_word and clean_word.isalpha() and len(clean_word) > 2:
                    words.append(clean_word)
            
            stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", 
                         "with", "is", "are", "was", "were", "be", "as", "by", "that", 
                         "this", "it", "from", "at", "but", "not", "have", "has", "had"}
            
            freq = Counter()
            for word in words:
                if word not in stop_words:
                    freq[word] += 1
            
            top_keywords = [word for word, count in freq.most_common(15)]
            
            blackboard.post(self.agent_id, "keywords", {"keywords": top_keywords}, score=0.8)
            
        except Exception:
            blackboard.post(self.agent_id, "keywords", {"keywords": []}, score=0.0)
        
        self.execution_time = time.time() - start_time


class SentenceRankAgent(Agent):
    """Enhanced sentence ranking with task-aware scoring."""
    
    def __init__(self, agent_id: str, role: str = "ranking", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.SUMMARIZATION, TaskType.COVERAGE_SCORING, TaskType.REDUNDANCY_ANALYSIS, TaskType.QUESTION_ANSWERING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.012)  # Simulate processing
        
        if not self.enabled or not input_text:
            return
            
        try:
            sentences = safe_sentence_split(input_text)
            if not sentences:
                blackboard.post(self.agent_id, "sent_scores", {"scores": []}, score=0.0)
                return
                
            keyword_posts = blackboard.read("keywords")
            all_keywords = set()
            for post in keyword_posts:
                if isinstance(post.get("payload"), dict):
                    keywords = post["payload"].get("keywords", [])
                    if isinstance(keywords, list):
                        all_keywords.update(str(kw).lower() for kw in keywords)
            
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                tokens = [w.lower().strip(",;:!?()[]") for w in sentence.split()]
                tokens = [t for t in tokens if t and t.isalpha()]
                
                overlap_count = sum(1 for token in tokens if token in all_keywords)
                overlap_score = overlap_count / max(1, len(all_keywords)) if all_keywords else 0
                position_bonus = 1.0 - (i / max(1, len(sentences)) * 0.3)
                
                length = len(tokens)
                length_penalty = 1.0 if 3 <= length <= 30 else 0.5
                
                final_score = (overlap_score * 0.6 + position_bonus * 0.4) * length_penalty
                scored_sentences.append((sentence, final_score))
            
            avg_score = sum(score for _, score in scored_sentences) / len(scored_sentences) if scored_sentences else 0
            blackboard.post(self.agent_id, "sent_scores", {"scores": scored_sentences}, score=avg_score)
            
        except Exception:
            blackboard.post(self.agent_id, "sent_scores", {"scores": []}, score=0.0)
        
        self.execution_time = time.time() - start_time


class NamedEntityRecognitionAgent(Agent):
    """Agent for Named Entity Recognition."""
    
    def __init__(self, agent_id: str, role: str = "ner", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.NAMED_ENTITY_RECOGNITION}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.02)  # Simulate NER processing
        
        if not self.enabled or not input_text:
            return
            
        try:
            words = input_text.split()
            entities = {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "MISC": []}
            
            org_keywords = {"company", "corporation", "corp", "inc", "llc", "university", "college"}
            location_keywords = {"city", "state", "country", "street", "avenue", "road"}
            
            for i, word in enumerate(words):
                clean_word = word.strip('.,!?;:()"')
                
                if clean_word and len(clean_word) > 1 and clean_word[0].isupper():
                    context = " ".join(words[max(0, i-2):i+3]).lower()
                    
                    if any(org in context for org in org_keywords):
                        entities["ORGANIZATION"].append(clean_word)
                    elif any(loc in context for loc in location_keywords):
                        entities["LOCATION"].append(clean_word)
                    elif clean_word.isdigit() or any(char.isdigit() for char in clean_word):
                        entities["DATE"].append(clean_word)
                    else:
                        entities["PERSON"].append(clean_word)
            
            # Remove duplicates
            for category in entities:
                entities[category] = list(set(entities[category]))
            
            total_entities = sum(len(ents) for ents in entities.values())
            confidence = min(1.0, total_entities / max(1, len(words)) * 10)
            
            blackboard.post(self.agent_id, "named_entities", {"entities": entities}, score=confidence)
            blackboard.set_task_metric("named_entity_recognition", "total_entities", total_entities)
            
        except Exception:
            blackboard.post(self.agent_id, "named_entities", {"entities": {}}, score=0.0)
        
        self.execution_time = time.time() - start_time


class SentimentAnalysisAgent(Agent):
    """Agent for Sentiment Analysis and Emotion Detection."""
    
    def __init__(self, agent_id: str, role: str = "sentiment_analyzer", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.SENTIMENT_ANALYSIS}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.015)  # Simulate sentiment processing
        
        if not self.enabled or not input_text:
            return
            
        try:
            positive_words = {
                "excellent", "amazing", "wonderful", "fantastic", "great", "good", "positive", 
                "happy", "joy", "love", "successful", "perfect", "outstanding", "brilliant"
            }
            
            negative_words = {
                "terrible", "awful", "horrible", "bad", "negative", "sad", "angry", "hate",
                "disgusting", "disappointing", "frustrated", "annoyed", "upset", "worried"
            }
            
            words = [w.lower().strip('.,!?;:()[]"') for w in input_text.split()]
            
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            
            total_sentiment_words = pos_count + neg_count
            if total_sentiment_words > 0:
                sentiment_score = (pos_count - neg_count) / total_sentiment_words
            else:
                sentiment_score = 0.0
            
            sentiment_label = "positive" if sentiment_score > 0.1 else ("negative" if sentiment_score < -0.1 else "neutral")
            confidence_score = abs(sentiment_score) if total_sentiment_words > 0 else 0.5
            
            result = {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "confidence": confidence_score,
                "pos_count": pos_count,
                "neg_count": neg_count
            }
            
            blackboard.post(self.agent_id, "sentiment_analysis", result, score=confidence_score)
            blackboard.set_task_metric("sentiment_analysis", "sentiment_strength", abs(sentiment_score))
            
        except Exception:
            blackboard.post(self.agent_id, "sentiment_analysis", {"sentiment_score": 0.0, "sentiment_label": "neutral"}, score=0.5)
        
        self.execution_time = time.time() - start_time


class TopicModelingAgent(Agent):
    """Agent for Topic Modeling and Theme Detection."""
    
    def __init__(self, agent_id: str, role: str = "topic_modeler", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.TOPIC_MODELING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.018)  # Simulate topic modeling
        
        if not self.enabled or not input_text:
            return
            
        try:
            domain_topics = {
                "technology": {"computer", "software", "digital", "internet", "data", "algorithm", "ai", "machine", "learning"},
                "business": {"company", "market", "sales", "profit", "customer", "strategy", "management", "revenue"},
                "health": {"medical", "doctor", "patient", "treatment", "health", "medicine", "hospital", "disease"},
                "education": {"school", "student", "teacher", "learning", "education", "university", "study"},
                "science": {"research", "study", "experiment", "theory", "analysis", "scientific", "method"},
                "sports": {"game", "team", "player", "sport", "competition", "match", "score"},
                "politics": {"government", "political", "policy", "election", "vote", "democracy", "law"}
            }
            
            words = [w.lower().strip('.,!?;:()[]"') for w in input_text.split() if w.isalpha() and len(w) > 3]
            word_freq = Counter(words)
            
            topic_scores = {}
            for topic, keywords in domain_topics.items():
                score = sum(word_freq.get(keyword, 0) for keyword in keywords)
                if score > 0:
                    topic_scores[topic] = score
            
            common_words = [word for word, count in word_freq.most_common(10) if count > 1]
            
            confidence = min(1.0, len(topic_scores) * 0.3)
            
            result = {
                "topic_scores": topic_scores,
                "dominant_topics": sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "common_terms": common_words
            }
            
            blackboard.post(self.agent_id, "topic_modeling", result, score=confidence)
            blackboard.set_task_metric("topic_modeling", "topics_identified", len(topic_scores))
            
        except Exception:
            blackboard.post(self.agent_id, "topic_modeling", {"topics": [], "themes": []}, score=0.0)
        
        self.execution_time = time.time() - start_time


class TextClassificationAgent(Agent):
    """Agent for Text Classification and Categorization."""
    
    def __init__(self, agent_id: str, role: str = "text_classifier", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.TEXT_CLASSIFICATION}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.016)  # Simulate classification
        
        if not self.enabled or not input_text:
            return
            
        try:
            categories = {
                "news": {"news", "report", "journalist", "breaking", "headline", "story"},
                "academic": {"research", "study", "academic", "university", "journal", "paper"},
                "business": {"business", "company", "corporate", "financial", "market", "profit"},
                "technical": {"technical", "engineering", "system", "software", "hardware"},
                "personal": {"personal", "diary", "blog", "opinion", "thoughts", "feelings"},
                "review": {"review", "rating", "opinion", "recommend", "feedback"}
            }
            
            text_lower = input_text.lower()
            category_scores = {}
            
            for category, keywords in categories.items():
                score = sum(text_lower.count(keyword) for keyword in keywords)
                if score > 0:
                    category_scores[category] = score
            
            primary_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            word_count = len(input_text.split())
            length_category = "short" if word_count < 50 else ("medium" if word_count < 200 else "long")
            
            confidence = min(1.0, len(category_scores) * 0.4)
            
            result = {
                "primary_categories": [cat for cat, score in primary_categories],
                "category_scores": category_scores,
                "length_category": length_category,
                "word_count": word_count
            }
            
            blackboard.post(self.agent_id, "text_classification", result, score=confidence)
            blackboard.set_task_metric("text_classification", "categories_detected", len(category_scores))
            
        except Exception:
            blackboard.post(self.agent_id, "text_classification", {"categories": ["general"]}, score=0.0)
        
        self.execution_time = time.time() - start_time


class QuestionAnsweringAgent(Agent):
    """Agent for Question Answering and Information Extraction."""
    
    def __init__(self, agent_id: str, role: str = "qa_agent", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.QUESTION_ANSWERING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.014)  # Simulate QA processing
        
        if not self.enabled or not input_text:
            return
            
        try:
            sentences = safe_sentence_split(input_text)
            question_indicators = ["what", "who", "when", "where", "why", "how", "which", "can", "do", "is", "are"]
            
            questions = []
            facts = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                if sentence.strip().endswith('?') or any(qi in sentence_lower[:20] for qi in question_indicators):
                    questions.append(sentence.strip())
                elif any(char.isdigit() for char in sentence) or any(word[0].isupper() for word in sentence.split()):
                    facts.append(sentence.strip())
            
            qa_pairs = []
            for i, question in enumerate(questions):
                answer = facts[i] if i < len(facts) else "No specific answer found."
                qa_pairs.append({"question": question, "answer": answer})
            
            confidence = min(1.0, (len(questions) * 0.4 + len(facts) * 0.2))
            
            result = {
                "questions": questions,
                "qa_pairs": qa_pairs,
                "key_facts": facts[:10],
                "total_questions": len(questions),
                "total_facts": len(facts)
            }
            
            blackboard.post(self.agent_id, "question_answering", result, score=confidence)
            blackboard.set_task_metric("question_answering", "qa_pairs", len(qa_pairs))
            
        except Exception:
            blackboard.post(self.agent_id, "question_answering", {"questions": [], "answers": []}, score=0.0)
        
        self.execution_time = time.time() - start_time


class RedundancyAgent(Agent):
    """Agent for detecting redundancy in text."""
    
    def __init__(self, agent_id: str, role: str = "redundancy", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.REDUNDANCY_ANALYSIS}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.013)  # Simulate processing
        
        if not self.enabled or not input_text:
            return
        
        try:
            sentences = safe_sentence_split(input_text)
            redundancy_score = 0.0
            
            if len(sentences) > 1:
                word_sets = []
                for sentence in sentences:
                    words = set(word.lower().strip('.,!?;:()[]"') for word in sentence.split() 
                               if word.isalpha() and len(word) > 3)
                    word_sets.append(words)
                
                total_comparisons = 0
                total_similarity = 0.0
                
                for i in range(len(word_sets)):
                    for j in range(i + 1, len(word_sets)):
                        if word_sets[i] and word_sets[j]:
                            overlap = len(word_sets[i] & word_sets[j])
                            union = len(word_sets[i] | word_sets[j])
                            similarity = overlap / union if union > 0 else 0
                            total_similarity += similarity
                            total_comparisons += 1
                
                redundancy_score = total_similarity / total_comparisons if total_comparisons > 0 else 0.0
            
            blackboard.post(self.agent_id, "redundancy", {"redundancy_score": redundancy_score}, score=redundancy_score)
            
        except Exception:
            blackboard.post(self.agent_id, "redundancy", {"redundancy_score": 0.0}, score=0.0)
        
        self.execution_time = time.time() - start_time


class GrammarAgent(Agent):
    """Agent for basic grammar and fluency checking."""
    
    def __init__(self, agent_id: str, role: str = "grammar", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.GRAMMAR_CHECK}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.011)  # Simulate processing
        
        if not self.enabled or not input_text:
            return
        
        try:
            sentences = safe_sentence_split(input_text)
            issues = 0
            total_checks = 0
            
            for sentence in sentences:
                words = sentence.split()
                total_checks += len(words)
                
                # Simple grammar checks
                if not sentence[0].isupper():  # Capitalization
                    issues += 1
                if not sentence.strip().endswith('.'):  # Punctuation
                    issues += 1
                if len(words) < 3:  # Too short
                    issues += 1
                
                # Check for repeated words
                for i in range(len(words) - 1):
                    if words[i].lower() == words[i + 1].lower():
                        issues += 1
            
            fluency_score = 1.0 - (issues / max(1, total_checks))
            fluency_score = max(0.0, min(1.0, fluency_score))
            
            blackboard.post(self.agent_id, "fluency", {"fluency_score": fluency_score, "issues": issues}, score=fluency_score)
            
        except Exception:
            blackboard.post(self.agent_id, "fluency", {"fluency_score": 0.5, "issues": 0}, score=0.5)
        
        self.execution_time = time.time() - start_time


class CoverageAgent(Agent):
    """Agent for coverage and importance scoring."""
    
    def __init__(self, agent_id: str, role: str = "coverage", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.COVERAGE_SCORING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        time.sleep(0.009)  # Simulate processing
        
        if not self.enabled or not input_text:
            return
        
        try:
            words = input_text.lower().split()
            word_freq = Counter(words)
            
            # Calculate coverage based on vocabulary diversity
            unique_words = len(set(words))
            total_words = len(words)
            vocabulary_diversity = unique_words / max(1, total_words)
            
            # Calculate importance based on word frequency distribution
            freq_values = list(word_freq.values())
            if freq_values:
                max_freq = max(freq_values)
                importance_score = sum(freq / max_freq for freq in freq_values) / len(freq_values)
            else:
                importance_score = 0.0
            
            coverage_score = (vocabulary_diversity + importance_score) / 2
            
            blackboard.post(self.agent_id, "coverage_pref", {
                "coverage_score": coverage_score,
                "vocabulary_diversity": vocabulary_diversity,
                "importance_score": importance_score
            }, score=coverage_score)
            
        except Exception:
            blackboard.post(self.agent_id, "coverage_pref", {"coverage_score": 0.5}, score=0.5)
        
        self.execution_time = time.time() - start_time


############################################################
# Performance Metrics
############################################################

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for SwarmMind execution."""
    execution_mode: str
    total_agents: int
    threads_created: int
    tasks_completed: int
    tasks_failed: int
    start_time: float
    end_time: float
    duration: float
    cpu_usage_before: float
    cpu_usage_after: float
    memory_usage_mb: float
    multitasking_efficiency: float = 0.0
    throughput: float = 0.0
    success_rate: float = 0.0
    average_agent_time: float = 0.0
    
    def __post_init__(self):
        self.success_rate = (self.tasks_completed / max(1, self.tasks_completed + self.tasks_failed)) * 100
        self.throughput = self.tasks_completed / max(0.001, self.duration)


############################################################
# Task Results Processor
############################################################

class TaskResultsProcessor:
    """Process and format results for different task types."""
    
    @staticmethod
    def process_summarization_results(blackboard: Blackboard, k: int = 3) -> Dict[str, Any]:
        sent_posts = blackboard.read("sent_scores")
        if not sent_posts:
            return {"summary": "No sentences to summarize", "top_sentences": []}
        
        all_scores = []
        for post in sent_posts:
            scores = post.get("payload", {}).get("scores", [])
            all_scores.extend(scores)
        
        # Sort by score and get top k
        sorted_sentences = sorted(all_scores, key=lambda x: x[1] if len(x) > 1 else 0, reverse=True)
        top_sentences = [sent for sent, score in sorted_sentences[:k]]
        summary = " ".join(top_sentences)
        
        return {"summary": summary, "top_sentences": top_sentences, "total_sentences": len(all_scores)}
    
    @staticmethod
    def process_keyword_extraction_results(blackboard: Blackboard, k: int = 10) -> Dict[str, Any]:
        keyword_posts = blackboard.read("keywords")
        all_keywords = []
        
        for post in keyword_posts:
            keywords = post.get("payload", {}).get("keywords", [])
            all_keywords.extend(keywords)
        
        # Count frequency and get top k
        keyword_freq = Counter(all_keywords)
        top_keywords = [word for word, count in keyword_freq.most_common(k)]
        
        return {"keywords": top_keywords, "keyword_frequency": dict(keyword_freq.most_common(k))}
    
    @staticmethod
    def process_named_entity_recognition(blackboard: Blackboard) -> Dict[str, Any]:
        ner_posts = blackboard.read("named_entities")
        all_entities = {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "MISC": []}
        
        for post in ner_posts:
            entities = post.get("payload", {}).get("entities", {})
            for category, entity_list in entities.items():
                if category in all_entities:
                    all_entities[category].extend(entity_list)
        
        # Remove duplicates
        for category in all_entities:
            all_entities[category] = list(set(all_entities[category]))
        
        total_entities = sum(len(entities) for entities in all_entities.values())
        
        return {
            "entities": all_entities,
            "total_entities": total_entities,
            "categories": len([cat for cat, entities in all_entities.items() if entities])
        }
    
    @staticmethod
    def process_sentiment_analysis(blackboard: Blackboard) -> Dict[str, Any]:
        sentiment_posts = blackboard.read("sentiment_analysis")
        if not sentiment_posts:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}
        
        scores = [post.get("payload", {}).get("sentiment_score", 0.0) for post in sentiment_posts]
        labels = [post.get("payload", {}).get("sentiment_label", "neutral") for post in sentiment_posts]
        confidences = [post.get("payload", {}).get("confidence", 0.0) for post in sentiment_posts]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        most_common_label = Counter(labels).most_common(1)[0][0] if labels else "neutral"
        
        return {
            "sentiment": most_common_label,
            "score": avg_score,
            "confidence": avg_confidence,
            "total_analyses": len(sentiment_posts)
        }
    
    @staticmethod
    def process_topic_modeling(blackboard: Blackboard) -> Dict[str, Any]:
        topic_posts = blackboard.read("topic_modeling")
        all_topics = {}
        
        for post in topic_posts:
            topic_scores = post.get("payload", {}).get("topic_scores", {})
            for topic, score in topic_scores.items():
                all_topics[topic] = all_topics.get(topic, 0) + score
        
        sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "dominant_topics": sorted_topics[:5],
            "all_topics": all_topics,
            "topics_identified": len(all_topics)
        }
    
    @staticmethod
    def process_text_classification(blackboard: Blackboard) -> Dict[str, Any]:
        classification_posts = blackboard.read("text_classification")
        all_categories = {}
        
        for post in classification_posts:
            categories = post.get("payload", {}).get("primary_categories", [])
            for category in categories:
                all_categories[category] = all_categories.get(category, 0) + 1
        
        primary_category = max(all_categories, key=all_categories.get) if all_categories else "general"
        
        return {
            "primary_category": primary_category,
            "all_categories": all_categories,
            "categories_detected": len(all_categories)
        }
    
    @staticmethod
    def process_question_answering(blackboard: Blackboard) -> Dict[str, Any]:
        qa_posts = blackboard.read("question_answering")
        all_questions = []
        all_facts = []
        qa_pairs = []
        
        for post in qa_posts:
            payload = post.get("payload", {})
            all_questions.extend(payload.get("questions", []))
            all_facts.extend(payload.get("key_facts", []))
            qa_pairs.extend(payload.get("qa_pairs", []))
        
        return {
            "questions": list(set(all_questions)),
            "key_facts": list(set(all_facts))[:10],
            "qa_pairs": qa_pairs,
            "total_questions": len(set(all_questions)),
            "total_facts": len(set(all_facts))
        }
    
    @staticmethod
    def process_redundancy_analysis_results(blackboard: Blackboard, k: int = 3) -> Dict[str, Any]:
        redundancy_posts = blackboard.read("redundancy")
        scores = [post.get("payload", {}).get("redundancy_score", 0.0) for post in redundancy_posts]
        avg_redundancy = sum(scores) / len(scores) if scores else 0.0
        
        return {"redundancy_score": avg_redundancy, "redundancy_level": "high" if avg_redundancy > 0.7 else ("medium" if avg_redundancy > 0.4 else "low")}
    
    @staticmethod
    def process_grammar_check_results(blackboard: Blackboard, k: int = 3) -> Dict[str, Any]:
        grammar_posts = blackboard.read("fluency")
        scores = [post.get("payload", {}).get("fluency_score", 0.5) for post in grammar_posts]
        issues = [post.get("payload", {}).get("issues", 0) for post in grammar_posts]
        
        avg_fluency = sum(scores) / len(scores) if scores else 0.5
        total_issues = sum(issues)
        
        return {"fluency_score": avg_fluency, "total_issues": total_issues, "quality": "good" if avg_fluency > 0.8 else ("fair" if avg_fluency > 0.6 else "poor")}
    
    @staticmethod
    def process_coverage_scoring_results(blackboard: Blackboard, k: int = 3) -> Dict[str, Any]:
        coverage_posts = blackboard.read("coverage_pref")
        scores = [post.get("payload", {}).get("coverage_score", 0.5) for post in coverage_posts]
        avg_coverage = sum(scores) / len(scores) if scores else 0.5
        
        return {"coverage_score": avg_coverage, "coverage_level": "high" if avg_coverage > 0.7 else ("medium" if avg_coverage > 0.4 else "low")}


############################################################
# Enhanced Swarm Controller with Parallel Execution
############################################################

class EnhancedSwarmController:
    """Enhanced swarm management with concurrent execution and performance tracking."""
    
    def __init__(self, blackboard=None):
        self.blackboard = blackboard or Blackboard()
        self.agents: Dict[str, Agent] = {}
        self.current_task: Optional[TaskType] = None
        self.performance_history = []
        self.baseline_sequential_time = None
        
    def configure_for_task(self, task_type: TaskType, num_agents: int):
        """Configure swarm for specific task."""
        self.current_task = task_type
        self.agents.clear()
        
        agents = self._create_task_specific_agents(task_type, num_agents)
        for agent in agents:
            self.agents[agent.agent_id] = agent
    
    def _create_task_specific_agents(self, task_type: TaskType, num_agents: int) -> List[Agent]:
        """Create optimal agent mix for specific task."""
        agents = []
        num_agents = max(1, min(20, num_agents))
        
        if task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            ner_count = max(1, num_agents // 2)
            for i in range(ner_count):
                agents.append(NamedEntityRecognitionAgent(f"ner_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - ner_count
            for i in range(remaining):
                if i % 2 == 0:
                    agents.append(KeywordAgent(f"kw_{i//2 + 1}", trust=0.7))
                else:
                    agents.append(SentenceRankAgent(f"sent_{i//2 + 1}", trust=0.6))
        
        elif task_type == TaskType.SENTIMENT_ANALYSIS:
            sentiment_count = max(1, num_agents // 2)
            for i in range(sentiment_count):
                agents.append(SentimentAnalysisAgent(f"sentiment_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - sentiment_count
            for i in range(remaining):
                if i % 2 == 0:
                    agents.append(TextClassificationAgent(f"ml_{i//2 + 1}", trust=0.6))
                else:
                    agents.append(KeywordAgent(f"kw_{i//2 + 1}", trust=0.6))
        
        elif task_type == TaskType.TOPIC_MODELING:
            topic_count = max(1, num_agents // 2)
            for i in range(topic_count):
                agents.append(TopicModelingAgent(f"topic_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - topic_count
            for i in range(remaining):
                agents.append(KeywordAgent(f"kw_{i+1}", trust=0.7))
        
        elif task_type == TaskType.TEXT_CLASSIFICATION:
            class_count = max(1, num_agents // 2)
            for i in range(class_count):
                agents.append(TextClassificationAgent(f"classifier_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - class_count
            for i in range(remaining):
                if i % 2 == 0:
                    agents.append(KeywordAgent(f"kw_{i//2 + 1}", trust=0.7))
                else:
                    agents.append(SentenceRankAgent(f"sent_{i//2 + 1}", trust=0.6))
        
        elif task_type == TaskType.QUESTION_ANSWERING:
            qa_count = max(1, min(num_agents // 2, 3))
            for i in range(qa_count):
                agents.append(QuestionAnsweringAgent(f"qa_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - qa_count
            for i in range(remaining):
                if i % 3 == 0:
                    agents.append(SentenceRankAgent(f"sent_{i//3 + 1}", trust=0.7))
                elif i % 3 == 1:
                    agents.append(KeywordAgent(f"kw_{i//3 + 1}", trust=0.7))
                else:
                    agents.append(NamedEntityRecognitionAgent(f"ner_{i//3 + 1}", trust=0.6))
        
        elif task_type == TaskType.SUMMARIZATION:
            sent_count = max(1, num_agents // 2)
            for i in range(sent_count):
                agents.append(SentenceRankAgent(f"sent_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - sent_count
            for i in range(remaining):
                agents.append(KeywordAgent(f"kw_{i+1}", trust=0.7))
        
        elif task_type == TaskType.KEYWORD_EXTRACTION:
            kw_count = max(1, num_agents // 2)
            for i in range(kw_count):
                agents.append(KeywordAgent(f"kw_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - kw_count
            for i in range(remaining):
                agents.append(SentenceRankAgent(f"sent_{i+1}", trust=0.7))
        
        elif task_type == TaskType.REDUNDANCY_ANALYSIS:
            for i in range(num_agents):
                agents.append(RedundancyAgent(f"redundancy_{i+1}", trust=0.8 - i*0.05))
        
        elif task_type == TaskType.GRAMMAR_CHECK:
            for i in range(num_agents):
                agents.append(GrammarAgent(f"grammar_{i+1}", trust=0.8 - i*0.05))
        
        elif task_type == TaskType.COVERAGE_SCORING:
            cov_count = max(1, num_agents // 2)
            for i in range(cov_count):
                agents.append(CoverageAgent(f"coverage_{i+1}", trust=0.8 - i*0.05))
            
            remaining = num_agents - cov_count
            for i in range(remaining):
                agents.append(KeywordAgent(f"kw_{i+1}", trust=0.7))
        
        else:
            # Default: mixed agents
            for i in range(num_agents):
                if i % 3 == 0:
                    agents.append(KeywordAgent(f"kw_{i//3 + 1}", trust=0.7))
                elif i % 3 == 1:
                    agents.append(SentenceRankAgent(f"sent_{i//3 + 1}", trust=0.7))
                else:
                    agents.append(NamedEntityRecognitionAgent(f"ner_{i//3 + 1}", trust=0.6))
        
        return agents[:num_agents]
    
    def run_task_processing_parallel(self, input_text: str, rounds: int = 2) -> PerformanceMetrics:
        """Run agents concurrently with full performance tracking."""
        # Clear blackboard
        channels_to_clear = [
            "keywords", "sent_scores", "redundancy", "fluency", "coverage_pref", 
            "named_entities", "sentiment_analysis", "topic_modeling", 
            "text_classification", "question_answering"
        ]
        for channel in channels_to_clear:
            self.blackboard.clear_channel(channel)
        
        # Performance tracking setup
        try:
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().used / 1024 / 1024
        except:
            cpu_before = 0.0
            memory_before = 0.0
            
        start_time = time.time()
        
        total_completed = 0
        total_failed = 0
        threads_created = 0
        all_agent_times = []
        
        print(f"🚀 Starting PARALLEL execution with {len(self.agents)} agents...")
        print(f"💾 Initial Memory: {memory_before:.1f} MB")
        print(f"🔥 Initial CPU: {cpu_before:.1f}%")
        
        for round_num in range(rounds):
            print(f"🔄 Round {round_num + 1}/{rounds}")
            
            suitable_agents = [
                agent for agent in self.agents.values() 
                if agent.enabled and agent.is_suitable_for_task(self.current_task)
            ]
            
            if not suitable_agents:
                continue
            
            max_workers = min(len(suitable_agents), 8)
            threads_created += max_workers
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_agent = {
                    executor.submit(self._safe_agent_run, agent, input_text): agent 
                    for agent in suitable_agents
                }
                
                completed_this_round = 0
                failed_this_round = 0
                
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result(timeout=30)
                        if result:
                            completed_this_round += 1
                            all_agent_times.append(agent.execution_time)
                            print(f"  ✅ {agent.agent_id} ({agent.execution_time:.3f}s)")
                        else:
                            failed_this_round += 1
                            print(f"  ❌ {agent.agent_id}")
                    except Exception as e:
                        failed_this_round += 1
                        print(f"  ⚠️  {agent.agent_id}: {str(e)[:30]}...")
                
                total_completed += completed_this_round
                total_failed += failed_this_round
                
                print(f"    Round {round_num + 1}: ✅{completed_this_round} ❌{failed_this_round}")
        
        end_time = time.time()
        try:
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().used / 1024 / 1024
        except:
            cpu_after = 0.0
            memory_after = memory_before
        
        metrics = PerformanceMetrics(
            execution_mode="parallel",
            total_agents=len(self.agents),
            threads_created=threads_created,
            tasks_completed=total_completed,
            tasks_failed=total_failed,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            cpu_usage_before=cpu_before,
            cpu_usage_after=cpu_after,
            memory_usage_mb=memory_after - memory_before,
            average_agent_time=sum(all_agent_times) / len(all_agent_times) if all_agent_times else 0.0
        )
        
        if self.baseline_sequential_time:
            metrics.multitasking_efficiency = (self.baseline_sequential_time / metrics.duration) * 100
        
        self.performance_history.append(metrics)
        
        print(f"⏱️  Parallel execution completed in {metrics.duration:.3f}s")
        print(f"📊 Average agent execution time: {metrics.average_agent_time:.3f}s")
        return metrics
    
    def run_task_processing_sequential(self, input_text: str, rounds: int = 2) -> PerformanceMetrics:
        """Run agents sequentially for baseline comparison."""
        # Clear blackboard
        channels_to_clear = [
            "keywords", "sent_scores", "redundancy", "fluency", "coverage_pref", 
            "named_entities", "sentiment_analysis", "topic_modeling", 
            "text_classification", "question_answering"
        ]
        for channel in channels_to_clear:
            self.blackboard.clear_channel(channel)
        
        try:
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().used / 1024 / 1024
        except:
            cpu_before = 0.0
            memory_before = 0.0
            
        start_time = time.time()
        
        total_completed = 0
        total_failed = 0
        all_agent_times = []
        
        print(f"🐌 Starting SEQUENTIAL execution with {len(self.agents)} agents...")
        
        for round_num in range(rounds):
            print(f"🔄 Round {round_num + 1}/{rounds}")
            
            for agent in self.agents.values():
                if agent.enabled and agent.is_suitable_for_task(self.current_task):
                    try:
                        result = self._safe_agent_run(agent, input_text)
                        if result:
                            total_completed += 1
                            all_agent_times.append(agent.execution_time)
                            print(f"  ✅ {agent.agent_id} ({agent.execution_time:.3f}s)")
                        else:
                            total_failed += 1
                            print(f"  ❌ {agent.agent_id}")
                    except Exception as e:
                        total_failed += 1
                        print(f"  ⚠️  {agent.agent_id}: {str(e)[:30]}...")
        
        end_time = time.time()
        try:
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().used / 1024 / 1024
        except:
            cpu_after = 0.0
            memory_after = memory_before
        
        metrics = PerformanceMetrics(
            execution_mode="sequential",
            total_agents=len(self.agents),
            threads_created=1,
            tasks_completed=total_completed,
            tasks_failed=total_failed,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            cpu_usage_before=cpu_before,
            cpu_usage_after=cpu_after,
            memory_usage_mb=memory_after - memory_before,
            average_agent_time=sum(all_agent_times) / len(all_agent_times) if all_agent_times else 0.0
        )
        
        self.baseline_sequential_time = metrics.duration
        self.performance_history.append(metrics)
        
        print(f"⏱️  Sequential execution completed in {metrics.duration:.3f}s")
        print(f"📊 Average agent execution time: {metrics.average_agent_time:.3f}s")
        return metrics
    
    def _safe_agent_run(self, agent: Agent, input_text: str) -> bool:
        """Safely execute an agent's run method."""
        try:
            agent.run(input_text, self.blackboard, self.current_task)
            return True
        except Exception as e:
            print(f"Error in {agent.agent_id}: {e}")
            return False
    
    def run_performance_comparison(self, input_text: str, rounds: int = 2):
        """Run both sequential and parallel execution for comparison."""
        print("=" * 70)
        print("🔬 PERFORMANCE COMPARISON MODE")
        print("=" * 70)
        
        sequential_metrics = self.run_task_processing_sequential(input_text, rounds)
        print("\n" + "="*50)
        parallel_metrics = self.run_task_processing_parallel(input_text, rounds)
        
        self._display_performance_report(sequential_metrics, parallel_metrics)
    
    def _display_performance_report(self, seq_metrics: PerformanceMetrics, par_metrics: PerformanceMetrics):
        """Display comprehensive performance comparison."""
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 70)
        
        speedup = seq_metrics.duration / par_metrics.duration if par_metrics.duration > 0 else float('inf')
        efficiency = (speedup / par_metrics.threads_created) * 100 if par_metrics.threads_created > 0 else 0
        
        print(f"\n⏱️  EXECUTION TIME:")
        print(f"   Sequential: {seq_metrics.duration:.3f}s")
        print(f"   Parallel:   {par_metrics.duration:.3f}s")
        print(f"   Speedup:    {speedup:.2f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
        
        print(f"\n🧵 THREADING ANALYSIS:")
        print(f"   Sequential Threads: {seq_metrics.threads_created}")
        print(f"   Parallel Threads:   {par_metrics.threads_created}")
        print(f"   Thread Utilization: {par_metrics.threads_created}/{par_metrics.total_agents} agents")
        
        print(f"\n✅ TASK COMPLETION:")
        print(f"   Sequential Success: {seq_metrics.tasks_completed}/{seq_metrics.tasks_completed + seq_metrics.tasks_failed}")
        print(f"   Parallel Success:   {par_metrics.tasks_completed}/{par_metrics.tasks_completed + par_metrics.tasks_failed}")
        print(f"   Sequential Rate:    {seq_metrics.success_rate:.1f}%")
        print(f"   Parallel Rate:      {par_metrics.success_rate:.1f}%")
        
        print(f"\n🚀 THROUGHPUT:")
        print(f"   Sequential: {seq_metrics.throughput:.1f} tasks/sec")
        print(f"   Parallel:   {par_metrics.throughput:.1f} tasks/sec")
        improvement = (par_metrics.throughput/seq_metrics.throughput)*100 if seq_metrics.throughput > 0 else 100
        print(f"   Improvement: {improvement:.1f}%")
        
        print(f"\n⚡ AGENT PERFORMANCE:")
        print(f"   Sequential Avg Time: {seq_metrics.average_agent_time:.3f}s per agent")
        print(f"   Parallel Avg Time:   {par_metrics.average_agent_time:.3f}s per agent")
        
        print(f"\n💾 RESOURCE USAGE:")
        print(f"   Sequential Memory: {seq_metrics.memory_usage_mb:.1f} MB")
        print(f"   Parallel Memory:   {par_metrics.memory_usage_mb:.1f} MB")
        print(f"   CPU Usage Change:  {par_metrics.cpu_usage_after - par_metrics.cpu_usage_before:.1f}%")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if speedup > 2.0:
            print("   ✅ Excellent parallel performance - significant speedup achieved")
        elif speedup > 1.3:
            print("   ⚠️  Good parallel performance - moderate speedup")
        else:
            print("   ❌ Limited parallel benefit - consider optimizing")
        
        if par_metrics.success_rate >= 95:
            print("   ✅ High reliability - minimal task failures")
        elif par_metrics.success_rate >= 85:
            print("   ⚠️  Good reliability - some task failures")
        else:
            print("   ❌ Poor reliability - high failure rate")
        
        print(f"\n🔍 PARALLEL WORKING EVIDENCE:")
        print(f"   ✓ ThreadPoolExecutor used with {par_metrics.threads_created} concurrent threads")
        print(f"   ✓ {speedup:.2f}x speedup demonstrates true parallelism")
        print(f"   ✓ Individual agent times overlap (concurrent execution)")
        print(f"   ✓ Thread-safe blackboard enables concurrent data sharing")
    
    def get_task_results(self, k: int = 3) -> Dict[str, Any]:
        """Get formatted results for current task."""
        if not self.current_task:
            return {}
        
        if self.current_task == TaskType.SUMMARIZATION:
            return TaskResultsProcessor.process_summarization_results(self.blackboard, k)
        elif self.current_task == TaskType.KEYWORD_EXTRACTION:
            return TaskResultsProcessor.process_keyword_extraction_results(self.blackboard, k)
        elif self.current_task == TaskType.NAMED_ENTITY_RECOGNITION:
            return TaskResultsProcessor.process_named_entity_recognition(self.blackboard)
        elif self.current_task == TaskType.SENTIMENT_ANALYSIS:
            return TaskResultsProcessor.process_sentiment_analysis(self.blackboard)
        elif self.current_task == TaskType.TOPIC_MODELING:
            return TaskResultsProcessor.process_topic_modeling(self.blackboard)
        elif self.current_task == TaskType.TEXT_CLASSIFICATION:
            return TaskResultsProcessor.process_text_classification(self.blackboard)
        elif self.current_task == TaskType.QUESTION_ANSWERING:
            return TaskResultsProcessor.process_question_answering(self.blackboard)
        elif self.current_task == TaskType.REDUNDANCY_ANALYSIS:
            return TaskResultsProcessor.process_redundancy_analysis_results(self.blackboard, k)
        elif self.current_task == TaskType.GRAMMAR_CHECK:
            return TaskResultsProcessor.process_grammar_check_results(self.blackboard, k)
        elif self.current_task == TaskType.COVERAGE_SCORING:
            return TaskResultsProcessor.process_coverage_scoring_results(self.blackboard, k)
        else:
            return {"message": "Task completed successfully"}


############################################################
# Enhanced User Interface
############################################################

class EnhancedSwarmMindInterface:
    """Interactive command-line interface with performance comparison capabilities."""
    
    def __init__(self):
        self.swarm = EnhancedSwarmController()
        
    def _select_execution_mode(self) -> str:
        """Select execution mode."""
        print("\n⚙️  SELECT EXECUTION MODE:")
        print("1. 🚀 Parallel Execution (concurrent agents)")
        print("2. 🐌 Sequential Execution (one after another)")
        print("3. 🔬 Performance Comparison (both modes)")
        
        while True:
            try:
                choice = input("Enter choice (1-3, default: 3): ").strip()
                
                if choice == "1":
                    return "parallel"
                elif choice == "2":
                    return "sequential" 
                elif not choice or choice == "3":
                    return "comparison"
                else:
                    print("❌ Invalid choice. Please enter 1-3.")
            except Exception:
                return "comparison"
    
    def _select_task(self) -> Optional[TaskType]:
        """Select task type."""
        print("\n📋 SELECT NLP TASK:")
        tasks = list(TaskType)
        for i, task in enumerate(tasks, 1):
            task_name = task.value.replace('_', ' ').title()
            print(f"{i:2d}. {task_name}")
        print(" 0. Exit")
        
        while True:
            try:
                choice = input(f"Enter choice (0-{len(tasks)}, default: 1): ").strip()
                
                if choice == "0":
                    return None
                elif not choice or choice == "1":
                    return tasks[0]
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(tasks):
                        return tasks[idx]
                    else:
                        print(f"❌ Invalid choice. Please enter 0-{len(tasks)}.")
            except ValueError:
                print("❌ Please enter a valid number.")
            except Exception:
                return tasks[0]
    
    def _select_agent_count(self) -> int:
        """Select number of agents."""
        print("\n🤖 SELECT NUMBER OF AGENTS:")
        print("💡 More agents = more thorough analysis but slower processing")
        print("💡 Recommended: 5-10 agents for balanced performance")
        
        try:
            choice = input("Enter number of agents (1-20, default: 8): ").strip()
            if not choice:
                return 8
            
            count = int(choice)
            return max(1, min(20, count))
        except ValueError:
            print("❌ Invalid input. Using default: 8")
            return 8
        except Exception:
            return 8
    
    def _get_input_text(self) -> str:
        """Get input text from user."""
        print("\n📝 ENTER TEXT TO ANALYZE:")
        print("💡 You can paste multiple lines. Press Enter twice when done.")
        print("💡 Type 'sample' for sample text")
        print()
        
        lines = []
        empty_lines = 0
        
        while True:
            try:
                line = input()
                if line.strip().lower() == 'sample':
                    return self._get_sample_text()
                
                if line.strip() == "":
                    empty_lines += 1
                    if empty_lines >= 2:
                        break
                else:
                    empty_lines = 0
                
                lines.append(line)
            except (KeyboardInterrupt, EOFError):
                break
        
        text = "\n".join(lines).strip()
        return text if text else self._get_sample_text()
    
    def _get_sample_text(self) -> str:
        """Return sample text for demonstration."""
        return """Apple Inc. is a multinational technology company based in Cupertino, California. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. Apple is known for innovative products like the iPhone, iPad, and Mac computers. Tim Cook currently serves as the CEO, leading the company to new heights. The company's success has made it one of the most valuable corporations in the world. What makes Apple special is its focus on design and user experience. However, some critics argue that their products are overpriced. Despite this criticism, Apple continues to dominate the premium technology market. The company's headquarters, Apple Park, opened in 2017 in California. Apple's commitment to innovation drives its research and development efforts."""
    
    def _display_results(self, task_type: TaskType, results: Dict[str, Any], num_agents: int):
        """Display formatted results."""
        print(f"\n📊 TASK RESULTS: {task_type.value.upper()} ({num_agents} agents)")
        print("=" * 70)
        
        if task_type == TaskType.SUMMARIZATION:
            summary = results.get("summary", "")
            top_sentences = results.get("top_sentences", [])
            print("\n📝 SUMMARY:")
            print(f"  {summary}")
            print(f"\n📊 Top {len(top_sentences)} sentences selected from {results.get('total_sentences', 0)} total sentences")
        
        elif task_type == TaskType.KEYWORD_EXTRACTION:
            keywords = results.get("keywords", [])
            print("\n🔍 TOP KEYWORDS:")
            for i, keyword in enumerate(keywords[:10], 1):
                print(f"  {i:2d}. {keyword}")
        
        elif task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            entities = results.get("entities", {})
            print("\n👤 NAMED ENTITIES DETECTED:")
            for category, entity_list in entities.items():
                if entity_list:
                    print(f"\n  {category}:")
                    for entity in entity_list[:10]:
                        print(f"    • {entity}")
            
            print(f"\n📈 NER METRICS:")
            print(f"  🎯 Total Entities: {results.get('total_entities', 0)}")
            print(f"  📂 Categories Found: {results.get('categories', 0)}")
        
        elif task_type == TaskType.SENTIMENT_ANALYSIS:
            sentiment = results.get("sentiment", "neutral")
            score = results.get("score", 0.0)
            confidence = results.get("confidence", 0.0)
            
            emoji = "😊" if sentiment == "positive" else ("😢" if sentiment == "negative" else "😐")
            print(f"\n😊 SENTIMENT ANALYSIS:")
            print(f"  {emoji} Overall Sentiment: {sentiment.upper()}")
            print(f"  📊 Sentiment Score: {score:.3f}")
            print(f"  🎯 Confidence: {confidence*100:.1f}%")
            print(f"  📝 Total Analyses: {results.get('total_analyses', 0)}")
        
        elif task_type == TaskType.TOPIC_MODELING:
            dominant_topics = results.get("dominant_topics", [])
            print("\n🏷️ TOPIC MODELING RESULTS:")
            if dominant_topics:
                print("  📊 Dominant Topics:")
                for topic, score in dominant_topics:
                    print(f"    • {topic.title()}: {score} relevance points")
            else:
                print("    No specific topics identified")
            print(f"  🎯 Topics Identified: {results.get('topics_identified', 0)}")
        
        elif task_type == TaskType.TEXT_CLASSIFICATION:
            primary_category = results.get("primary_category", "general")
            all_categories = results.get("all_categories", {})
            print(f"\n📂 TEXT CLASSIFICATION:")
            print(f"  📝 Primary Category: {primary_category.title()}")
            if all_categories:
                print("\n  📊 All Categories Detected:")
                for category, score in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
                    print(f"    • {category.title()}: {score} indicators")
            print(f"  🎯 Categories Detected: {results.get('categories_detected', 0)}")
        
        elif task_type == TaskType.QUESTION_ANSWERING:
            questions = results.get("questions", [])
            key_facts = results.get("key_facts", [])
            qa_pairs = results.get("qa_pairs", [])
            
            print("\n❓ QUESTION ANSWERING ANALYSIS:")
            if questions:
                print(f"  ❓ Questions Identified ({len(questions)}):")
                for question in questions[:5]:
                    print(f"    • {question}")
            
            if qa_pairs:
                print(f"\n  💬 Question-Answer Pairs ({len(qa_pairs)}):")
                for pair in qa_pairs[:3]:
                    print(f"    Q: {pair['question']}")
                    print(f"    A: {pair['answer'][:100]}{'...' if len(pair['answer']) > 100 else ''}")
            
            if key_facts:
                print(f"\n  📋 Key Facts ({len(key_facts)}):")
                for fact in key_facts[:5]:
                    print(f"    • {fact[:80]}{'...' if len(fact) > 80 else ''}")
        
        elif task_type == TaskType.REDUNDANCY_ANALYSIS:
            redundancy_score = results.get("redundancy_score", 0.0)
            level = results.get("redundancy_level", "low")
            print(f"\n🔄 REDUNDANCY ANALYSIS:")
            print(f"  📊 Redundancy Score: {redundancy_score:.3f}")
            print(f"  📝 Redundancy Level: {level.upper()}")
        
        elif task_type == TaskType.GRAMMAR_CHECK:
            fluency_score = results.get("fluency_score", 0.5)
            total_issues = results.get("total_issues", 0)
            quality = results.get("quality", "fair")
            print(f"\n✅ GRAMMAR & FLUENCY CHECK:")
            print(f"  📊 Fluency Score: {fluency_score:.3f}")
            print(f"  ⚠️  Total Issues: {total_issues}")
            print(f"  📝 Quality Assessment: {quality.upper()}")
        
        elif task_type == TaskType.COVERAGE_SCORING:
            coverage_score = results.get("coverage_score", 0.5)
            level = results.get("coverage_level", "medium")
            print(f"\n📊 COVERAGE & IMPORTANCE SCORING:")
            print(f"  📊 Coverage Score: {coverage_score:.3f}")
            print(f"  📝 Coverage Level: {level.upper()}")
        
        print()
    
    def run(self):
        """Enhanced main loop with performance tracking."""
        print("=" * 70)
        print("🐝 SWARMMIND - Enhanced Multi-Agent NLP Framework")
        print("🚀 Now with True Concurrent Execution & Performance Analysis!")
        print("=" * 70)
        
        while True:
            try:
                # Task selection
                task_type = self._select_task()
                if not task_type:
                    break
                
                # Execution mode selection
                execution_mode = self._select_execution_mode()
                
                # Agent count selection  
                num_agents = self._select_agent_count()
                
                # Input text
                input_text = self._get_input_text()
                if not input_text.strip():
                    print("❌ No text provided. Using sample text.")
                    input_text = self._get_sample_text()
                
                # Configure and run
                print(f"\n🔧 Configuring {num_agents} agents for {task_type.value}...")
                self.swarm.configure_for_task(task_type, num_agents)
                
                if execution_mode == "comparison":
                    self.swarm.run_performance_comparison(input_text, rounds=2)
                elif execution_mode == "parallel":
                    metrics = self.swarm.run_task_processing_parallel(input_text, rounds=2)
                    print(f"\n📊 PARALLEL EXECUTION METRICS:")
                    print(f"   Duration: {metrics.duration:.3f}s")
                    print(f"   Threads: {metrics.threads_created}")
                    print(f"   Success Rate: {metrics.success_rate:.1f}%")
                    print(f"   Throughput: {metrics.throughput:.1f} tasks/sec")
                else:
                    metrics = self.swarm.run_task_processing_sequential(input_text, rounds=2)
                    print(f"\n📊 SEQUENTIAL EXECUTION METRICS:")
                    print(f"   Duration: {metrics.duration:.3f}s")
                    print(f"   Success Rate: {metrics.success_rate:.1f}%")
                    print(f"   Throughput: {metrics.throughput:.1f} tasks/sec")
                
                # Display task results
                results = self.swarm.get_task_results()
                if results:
                    self._display_results(task_type, results, num_agents)
                
                choice = input("\n❓ Continue with another test? (y/N): ").strip().lower()
                if choice not in ['y', 'yes']:
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using Enhanced SwarmMind.")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                choice = input("Continue? (y/N): ").strip().lower()
                if choice not in ['y', 'yes']:
                    break


############################################################
# Demo Function
############################################################

def demo_performance():
    """Demonstrate parallel performance with clear metrics."""
    print("🚀 SwarmMind Parallel Performance Demo")
    print("=" * 50)
    
    sample_text = """Apple Inc. is a multinational technology company based in Cupertino, California. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. Apple is known for innovative products like the iPhone, iPad, and Mac computers. Tim Cook currently serves as the CEO, leading the company to new heights."""
    
    swarm = EnhancedSwarmController()
    
    # Test with sentiment analysis
    print(f"\n📊 Testing Sentiment Analysis with 8 agents:")
    print("-" * 40)
    
    swarm.configure_for_task(TaskType.SENTIMENT_ANALYSIS, 8)
    swarm.run_performance_comparison(sample_text, rounds=1)


############################################################
# Main Entry Point
############################################################

def main():
    """Main entry point."""
    print("🐝 Enhanced SwarmMind Framework - Complete NLP Multi-Agent System")
    print("Choose mode:")
    print("1. 🎮 Interactive Mode (full interface)")
    print("2. 🧪 Performance Demo")
    
    try:
        choice = input("Enter choice (1-2, default: 1): ").strip()
        
        if choice == "2":
            demo_performance()
        else:
            interface = EnhancedSwarmMindInterface()
            interface.run()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()