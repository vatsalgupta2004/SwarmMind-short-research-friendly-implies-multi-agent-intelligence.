import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from collections import Counter

class TaskType(Enum):
    """Enhanced task types with descriptions"""
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

class Blackboard:
    """Enhanced thread-safe blackboard with real-time capabilities"""
    def __init__(self):
        self._lock = threading.RLock()
        self._posts: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = {}
        self._task_metrics: Dict[str, Dict[str, float]] = {}
        self._subscribers: Dict[str, List[callable]] = {}

    def post(self, agent_id: str, channel: str, payload: Any, score: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        """Post with notification system"""
        try:
            with self._lock:
                post = {
                    "agent_id": str(agent_id),
                    "channel": str(channel),
                    "payload": payload,
                    "score": max(0.0, min(1.0, float(score))),
                    "meta": meta or {},
                    "timestamp": time.time()
                }
                self._posts.append(post)
                
                # Notify subscribers
                if channel in self._subscribers:
                    for callback in self._subscribers[channel]:
                        try:
                            callback(post)
                        except Exception:
                            pass
        except Exception:
            pass

    def subscribe(self, channel: str, callback: callable):
        """Subscribe to channel updates"""
        with self._lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(callback)

    def read(self, channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enhanced read with filtering"""
        try:
            with self._lock:
                if channel is None:
                    return list(self._posts)
                return [p for p in self._posts if p.get("channel") == channel]
        except Exception:
            return []

    def clear_channel(self, channel: str):
        """Clear specific channel"""
        try:
            with self._lock:
                self._posts = [p for p in self._posts if p.get("channel") != channel]
        except Exception:
            pass

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get real-time metrics"""
        with self._lock:
            channels = {}
            for post in self._posts:
                channel = post.get("channel", "unknown")
                if channel not in channels:
                    channels[channel] = {"count": 0, "avg_score": 0.0, "agents": set()}
                channels[channel]["count"] += 1
                channels[channel]["avg_score"] += post.get("score", 0.0)
                channels[channel]["agents"].add(post.get("agent_id"))
            
            # Calculate averages
            for channel_data in channels.values():
                if channel_data["count"] > 0:
                    channel_data["avg_score"] /= channel_data["count"]
                channel_data["agents"] = len(channel_data["agents"])
            
            return channels

def safe_sentence_split(text: str) -> List[str]:
    """Enhanced sentence splitting"""
    try:
        if not text or not isinstance(text, str):
            return []
        
        # Better sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[:50]
    except Exception:
        return []

@dataclass
class Agent:
    """Enhanced agent with performance tracking"""
    agent_id: str
    role: str
    trust: float = 0.5
    enabled: bool = True
    task_affinity: Set[TaskType] = None
    execution_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_execution: float = 0.0

    def __post_init__(self):
        self.trust = max(0.0, min(1.0, float(self.trust)))
        if self.task_affinity is None:
            self.task_affinity = set(TaskType)

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        """Enhanced run with metrics"""
        start_time = time.time()
        self.last_execution = start_time
        
        try:
            # Simulate realistic processing with variability
            processing_time = 0.01 + (hash(self.agent_id) % 10) * 0.001
            time.sleep(processing_time)
            
            self.execution_time = time.time() - start_time
            self.success_count += 1
            return True
        except Exception:
            self.failure_count += 1
            return False

    def is_suitable_for_task(self, task_type: TaskType) -> bool:
        """Check task suitability"""
        return task_type in self.task_affinity

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics"""
        total_runs = self.success_count + self.failure_count
        return {
            "success_rate": (self.success_count / max(1, total_runs)) * 100,
            "avg_execution_time": self.execution_time,
            "trust_level": self.trust,
            "total_runs": total_runs
        }

# Specialized Agent Classes (shortened for brevity)
class KeywordAgent(Agent):
    """Enhanced keyword extraction agent"""
    def __init__(self, agent_id: str, role: str = "keywords", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.KEYWORD_EXTRACTION, TaskType.COVERAGE_SCORING, TaskType.SUMMARIZATION, TaskType.TOPIC_MODELING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        start_time = time.time()
        
        if not self.enabled or not input_text:
            return False

        try:
            # Enhanced keyword extraction
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
            
            blackboard.post(self.agent_id, "keywords", {
                "keywords": top_keywords,
                "frequency_data": dict(freq.most_common(15))
            }, score=0.8)
            
            self.execution_time = time.time() - start_time
            self.success_count += 1
            return True
            
        except Exception:
            blackboard.post(self.agent_id, "keywords", {"keywords": []}, score=0.0)
            self.failure_count += 1
            return False

# Add other specialized agents (SentenceRankAgent, NamedEntityRecognitionAgent, etc.)
# ... (implementation similar to original but with enhanced features)

class TaskResultsProcessor:
    """Enhanced results processor with visualization data"""
    
    @staticmethod
    def process_summarization_results(blackboard: Blackboard, k: int = 3) -> Dict[str, Any]:
        sent_posts = blackboard.read("sent_scores")
        if not sent_posts:
            return {"summary": "No sentences to summarize", "top_sentences": []}

        all_scores = []
        for post in sent_posts:
            scores = post.get("payload", {}).get("scores", [])
            all_scores.extend(scores)

        sorted_sentences = sorted(all_scores, key=lambda x: x[1] if len(x) > 1 else 0, reverse=True)
        top_sentences = [sent for sent, score in sorted_sentences[:k]]
        summary = " ".join(top_sentences)
        
        return {
            "summary": summary,
            "top_sentences": top_sentences,
            "total_sentences": len(all_scores),
            "sentence_scores": dict(sorted_sentences[:k])
        }

    @staticmethod
    def process_keyword_extraction_results(blackboard: Blackboard, k: int = 10) -> Dict[str, Any]:
        keyword_posts = blackboard.read("keywords")
        all_keywords = []
        all_frequency_data = {}
        
        for post in keyword_posts:
            payload = post.get("payload", {})
            keywords = payload.get("keywords", [])
            all_keywords.extend(keywords)
            
            # Merge frequency data
            freq_data = payload.get("frequency_data", {})
            for word, freq in freq_data.items():
                all_frequency_data[word] = all_frequency_data.get(word, 0) + freq

        keyword_freq = Counter(all_keywords)
        top_keywords = [word for word, count in keyword_freq.most_common(k)]
        
        return {
            "keywords": top_keywords,
            "keyword_frequency": dict(keyword_freq.most_common(k)),
            "detailed_frequency": all_frequency_data
        }

    # Add other processing methods...