# enhanced_swarmmind_framework_with_new_tasks.py
# Interactive multi-agent swarm framework for NLP tasks with dynamic scaling
# Enhanced with 5 additional NLP tasks: NER, Sentiment Analysis, Topic Modeling, Text Classification, QA

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import threading
import math
import heapq
import sys
from enum import Enum
from collections import Counter


class TaskType(Enum):
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
# Enhanced Blackboard with Task-Specific Channels
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
                    "timestamp": len(self._posts)  # Simple timestamp
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

    def set_state(self, key: str, value: Any):
        try:
            with self._lock:
                self._state[key] = value
        except Exception:
            pass

    def get_state(self, key: str, default: Any = None) -> Any:
        try:
            with self._lock:
                return self._state.get(key, default)
        except Exception:
            return default


############################################################
# Base Agent with Enhanced Capabilities
############################################################

@dataclass
class Agent:
    """Enhanced base agent class with task affinity."""
    agent_id: str
    role: str
    trust: float = 0.5
    enabled: bool = True
    task_affinity: Set[TaskType] = None

    def __post_init__(self):
        self.trust = max(0.0, min(1.0, float(self.trust)))
        if self.task_affinity is None:
            self.task_affinity = set(TaskType)  # Default: works with all tasks

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        raise NotImplementedError

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
        if not parts:
            return []
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
# Existing Agents (keeping original implementations)
############################################################

class KeywordAgent(Agent):
    """Enhanced keyword extraction with task-specific optimization."""
    
    def __init__(self, agent_id: str, role: str = "keywords", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.KEYWORD_EXTRACTION, TaskType.COVERAGE_SCORING, TaskType.SUMMARIZATION, TaskType.TOPIC_MODELING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            words = []
            for word in str(input_text).split():
                clean_word = word.lower().strip(",;:!?()[]\"'")
                if clean_word and clean_word.isalpha() and len(clean_word) > 2:
                    words.append(clean_word)
            
            if not words:
                blackboard.post(self.agent_id, "keywords", {"keywords": []}, score=0.0)
                return
                
            base_stop_words = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", 
                             "with", "is", "are", "was", "were", "be", "as", "by", "that", 
                             "this", "it", "from", "at", "but", "not", "have", "has", "had"}
            
            if task_type == TaskType.KEYWORD_EXTRACTION:
                additional_stops = {"can", "will", "would", "could", "should", "may", "might"}
                stop_words = base_stop_words | additional_stops
                keyword_limit = 20
            else:
                stop_words = base_stop_words
                keyword_limit = 15
            
            freq = {}
            for word in words:
                if word not in stop_words:
                    freq[word] = freq.get(word, 0) + 1
            
            if not freq:
                blackboard.post(self.agent_id, "keywords", {"keywords": []}, score=0.0)
                return
                
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, count in sorted_freq[:keyword_limit]]
            
            total_words = len(words)
            unique_ratio = len(freq) / max(1, total_words)
            base_score = min(1.0, unique_ratio * 2)
            
            if task_type == TaskType.KEYWORD_EXTRACTION:
                diversity_bonus = min(0.3, len(top_keywords) / 20)
                final_score = min(1.0, base_score + diversity_bonus)
            else:
                final_score = base_score
            
            blackboard.post(self.agent_id, "keywords", {"keywords": top_keywords}, score=final_score)
            
        except Exception:
            blackboard.post(self.agent_id, "keywords", {"keywords": []}, score=0.0)


class SentenceRankAgent(Agent):
    """Enhanced sentence ranking with task-aware scoring."""
    
    def __init__(self, agent_id: str, role: str = "ranking", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.SUMMARIZATION, TaskType.COVERAGE_SCORING, TaskType.REDUNDANCY_ANALYSIS, TaskType.QUESTION_ANSWERING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
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
                try:
                    tokens = [w.lower().strip(",;:!?()[]") for w in sentence.split()]
                    tokens = [t for t in tokens if t and t.isalpha()]
                    
                    overlap_count = sum(1 for token in tokens if token in all_keywords)
                    overlap_score = overlap_count / max(1, len(all_keywords)) if all_keywords else 0
                    
                    if task_type == TaskType.SUMMARIZATION:
                        position_bonus = 1.0 - (i / max(1, len(sentences)) * 0.3)
                    else:
                        position_bonus = 1.0 - (i / max(1, len(sentences)) * 0.1)
                    
                    length = len(tokens)
                    if length < 3:
                        length_penalty = 0.3
                    elif length > 30:
                        length_penalty = 0.7
                    else:
                        length_penalty = 1.0
                    
                    if task_type == TaskType.COVERAGE_SCORING:
                        final_score = (overlap_score * 0.8 + position_bonus * 0.2) * length_penalty
                    else:
                        final_score = (overlap_score * 0.6 + position_bonus * 0.3) * length_penalty
                    
                    scored_sentences.append((sentence, final_score))
                    
                except Exception:
                    scored_sentences.append((sentence, 0.0))
            
            if scored_sentences:
                avg_score = sum(score for _, score in scored_sentences) / len(scored_sentences)
                confidence = min(1.0, avg_score)
            else:
                confidence = 0.0
                
            blackboard.post(self.agent_id, "sent_scores", {"scores": scored_sentences}, score=confidence)
            
        except Exception:
            blackboard.post(self.agent_id, "sent_scores", {"scores": []}, score=0.0)


# [Keep all existing agents: RedundancyAgent, GrammarAgent, CoverageAgent, SimpleMLClassifierAgent]
# For brevity, I'm not repeating them here, but they remain unchanged

############################################################
# NEW NLP TASK AGENTS
############################################################

class NamedEntityRecognitionAgent(Agent):
    """Agent for Named Entity Recognition."""
    
    def __init__(self, agent_id: str, role: str = "ner", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.NAMED_ENTITY_RECOGNITION}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            # Simple NER - extract capitalized words and common entity patterns
            words = input_text.split()
            entities = {
                "PERSON": [],
                "ORGANIZATION": [],
                "LOCATION": [],
                "DATE": [],
                "MISC": []
            }
            
            # Common organization keywords
            org_keywords = {"company", "corporation", "corp", "inc", "llc", "university", "college"}
            location_keywords = {"city", "state", "country", "street", "avenue", "road"}
            
            for i, word in enumerate(words):
                clean_word = word.strip('.,!?;:()"')
                
                if clean_word and clean_word[0].isupper():
                    # Check context for classification
                    context = " ".join(words[max(0, i-2):i+3]).lower()
                    
                    if any(org in context for org in org_keywords):
                        entities["ORGANIZATION"].append(clean_word)
                    elif any(loc in context for loc in location_keywords):
                        entities["LOCATION"].append(clean_word)
                    elif clean_word.isdigit() or any(char.isdigit() for char in clean_word):
                        entities["DATE"].append(clean_word)
                    elif len(clean_word) > 1:  # Likely person name
                        entities["PERSON"].append(clean_word)
                    else:
                        entities["MISC"].append(clean_word)
            
            # Remove duplicates and calculate confidence
            for category in entities:
                entities[category] = list(set(entities[category]))
            
            total_entities = sum(len(ents) for ents in entities.values())
            confidence = min(1.0, total_entities / max(1, len(words)) * 10)
            
            blackboard.post(self.agent_id, "named_entities", {"entities": entities}, score=confidence)
            blackboard.set_task_metric("named_entity_recognition", "total_entities", total_entities)
            
        except Exception:
            blackboard.post(self.agent_id, "named_entities", {"entities": {}}, score=0.0)


class SentimentAnalysisAgent(Agent):
    """Agent for Sentiment Analysis and Emotion Detection."""
    
    def __init__(self, agent_id: str, role: str = "sentiment_analyzer", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.SENTIMENT_ANALYSIS}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            # Enhanced sentiment lexicons
            positive_words = {
                "excellent", "amazing", "wonderful", "fantastic", "great", "good", "positive", 
                "happy", "joy", "love", "successful", "perfect", "outstanding", "brilliant",
                "awesome", "superb", "magnificent", "delightful", "pleased", "satisfied"
            }
            
            negative_words = {
                "terrible", "awful", "horrible", "bad", "negative", "sad", "angry", "hate",
                "disgusting", "disappointing", "frustrated", "annoyed", "upset", "worried",
                "concerned", "difficult", "problem", "issue", "failure", "worst"
            }
            
            neutral_words = {
                "okay", "fine", "normal", "average", "standard", "typical", "usual", "common"
            }
            
            # Emotion categories
            emotions = {
                "joy": {"happy", "joy", "excited", "delighted", "cheerful", "elated"},
                "anger": {"angry", "furious", "mad", "irritated", "annoyed", "rage"},
                "sadness": {"sad", "depressed", "miserable", "grief", "sorrow", "melancholy"},
                "fear": {"afraid", "scared", "terrified", "anxious", "worried", "nervous"},
                "surprise": {"surprised", "amazed", "astonished", "shocked", "stunned"}
            }
            
            # Process text
            words = [w.lower().strip('.,!?;:()[]"') for w in input_text.split()]
            
            # Sentiment scoring
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            neu_count = sum(1 for w in words if w in neutral_words)
            
            # Calculate sentiment score (-1 to 1)
            total_sentiment_words = pos_count + neg_count + neu_count
            if total_sentiment_words > 0:
                sentiment_score = (pos_count - neg_count) / total_sentiment_words
            else:
                sentiment_score = 0.0
            
            # Normalize to 0-1 for confidence
            confidence_score = abs(sentiment_score) if total_sentiment_words > 0 else 0.5
            
            # Emotion detection
            emotion_scores = {}
            for emotion, emotion_words in emotions.items():
                emotion_count = sum(1 for w in words if w in emotion_words)
                emotion_scores[emotion] = emotion_count
            
            dominant_emotion = max(emotion_scores, key=emotion_scores.get) if any(emotion_scores.values()) else "neutral"
            
            # Sentiment classification
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            result = {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "word_counts": {"positive": pos_count, "negative": neg_count, "neutral": neu_count}
            }
            
            blackboard.post(self.agent_id, "sentiment_analysis", result, score=confidence_score)
            blackboard.set_task_metric("sentiment_analysis", "sentiment_strength", abs(sentiment_score))
            
        except Exception:
            blackboard.post(self.agent_id, "sentiment_analysis", {"sentiment_score": 0.0, "sentiment_label": "neutral"}, score=0.5)


class TopicModelingAgent(Agent):
    """Agent for Topic Modeling and Theme Detection."""
    
    def __init__(self, agent_id: str, role: str = "topic_modeler", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.TOPIC_MODELING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            # Enhanced topic modeling with domain-specific keywords
            domain_topics = {
                "technology": {"computer", "software", "digital", "internet", "data", "algorithm", "ai", "machine", "learning", "code"},
                "business": {"company", "market", "sales", "profit", "customer", "strategy", "management", "revenue", "investment", "growth"},
                "health": {"medical", "doctor", "patient", "treatment", "health", "medicine", "hospital", "disease", "therapy", "care"},
                "education": {"school", "student", "teacher", "learning", "education", "university", "knowledge", "study", "academic", "research"},
                "science": {"research", "study", "experiment", "theory", "analysis", "scientific", "method", "hypothesis", "data", "evidence"},
                "sports": {"game", "team", "player", "sport", "competition", "match", "score", "championship", "athletic", "training"},
                "politics": {"government", "political", "policy", "election", "vote", "democracy", "citizen", "law", "rights", "society"}
            }
            
            # Process text
            words = [w.lower().strip('.,!?;:()[]"') for w in input_text.split() if w.isalpha() and len(w) > 3]
            word_freq = Counter(words)
            
            # Calculate topic scores
            topic_scores = {}
            for topic, keywords in domain_topics.items():
                score = sum(word_freq.get(keyword, 0) for keyword in keywords)
                if score > 0:
                    topic_scores[topic] = score
            
            # Extract most frequent terms as themes
            common_words = [word for word, count in word_freq.most_common(10) if count > 1]
            
            # Identify co-occurring terms (simple bigrams)
            sentences = safe_sentence_split(input_text)
            themes = []
            for sentence in sentences:
                sentence_words = [w.lower().strip('.,!?;:()[]"') for w in sentence.split() if w.isalpha()]
                for i in range(len(sentence_words) - 1):
                    bigram = f"{sentence_words[i]} {sentence_words[i+1]}"
                    if len(bigram) > 6:  # Avoid very short bigrams
                        themes.append(bigram)
            
            theme_freq = Counter(themes)
            top_themes = [theme for theme, count in theme_freq.most_common(5) if count > 1]
            
            # Calculate confidence
            confidence = min(1.0, len(topic_scores) * 0.2 + len(top_themes) * 0.1)
            
            result = {
                "topic_scores": topic_scores,
                "dominant_topics": sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                "common_terms": common_words,
                "themes": top_themes,
                "word_frequency": dict(word_freq.most_common(15))
            }
            
            blackboard.post(self.agent_id, "topic_modeling", result, score=confidence)
            blackboard.set_task_metric("topic_modeling", "topics_identified", len(topic_scores))
            
        except Exception:
            blackboard.post(self.agent_id, "topic_modeling", {"topics": [], "themes": []}, score=0.0)


class TextClassificationAgent(Agent):
    """Agent for Text Classification and Categorization."""
    
    def __init__(self, agent_id: str, role: str = "text_classifier", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.TEXT_CLASSIFICATION}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            # Multi-category classification
            categories = {
                "news": {"news", "report", "journalist", "breaking", "headline", "story", "media", "press"},
                "academic": {"research", "study", "academic", "university", "journal", "paper", "scholar", "thesis"},
                "business": {"business", "company", "corporate", "financial", "market", "profit", "revenue", "strategy"},
                "technical": {"technical", "engineering", "system", "software", "hardware", "specification", "manual"},
                "personal": {"personal", "diary", "blog", "opinion", "thoughts", "feelings", "experience", "life"},
                "review": {"review", "rating", "opinion", "recommend", "feedback", "evaluation", "assessment"},
                "instructional": {"how", "tutorial", "guide", "instructions", "steps", "method", "procedure", "tips"},
                "creative": {"story", "creative", "fiction", "novel", "poetry", "art", "imagination", "narrative"}
            }
            
            # Content type indicators
            content_indicators = {
                "formal": {"therefore", "furthermore", "however", "consequently", "moreover", "nevertheless"},
                "informal": {"really", "pretty", "kinda", "gonna", "wanna", "stuff", "things"},
                "argumentative": {"argue", "claim", "evidence", "prove", "support", "oppose", "debate", "assert"},
                "descriptive": {"describe", "illustrate", "depict", "characterize", "detail", "outline"},
                "narrative": {"story", "once", "then", "first", "next", "finally", "happened", "event"}
            }
            
            # Process text
            words = [w.lower().strip('.,!?;:()[]"') for w in input_text.split()]
            text_lower = input_text.lower()
            
            # Calculate category scores
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(text_lower.count(keyword) for keyword in keywords)
                if score > 0:
                    category_scores[category] = score
            
            # Calculate style scores
            style_scores = {}
            for style, indicators in content_indicators.items():
                score = sum(text_lower.count(indicator) for indicator in indicators)
                if score > 0:
                    style_scores[style] = score
            
            # Determine primary categories
            primary_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            primary_style = max(style_scores.items(), key=lambda x: x[1])[0] if style_scores else "neutral"
            
            # Length-based classification
            word_count = len(words)
            if word_count < 50:
                length_category = "short"
            elif word_count < 200:
                length_category = "medium"
            else:
                length_category = "long"
            
            # Calculate confidence
            confidence = min(1.0, (len(category_scores) * 0.3 + len(style_scores) * 0.2))
            
            result = {
                "primary_categories": [cat for cat, score in primary_categories],
                "category_scores": category_scores,
                "style": primary_style,
                "style_scores": style_scores,
                "length_category": length_category,
                "word_count": word_count,
                "formality_level": "formal" if "formal" in style_scores else "informal"
            }
            
            blackboard.post(self.agent_id, "text_classification", result, score=confidence)
            blackboard.set_task_metric("text_classification", "categories_detected", len(category_scores))
            
        except Exception:
            blackboard.post(self.agent_id, "text_classification", {"categories": ["general"]}, score=0.0)


class QuestionAnsweringAgent(Agent):
    """Agent for Question Answering and Information Extraction."""
    
    def __init__(self, agent_id: str, role: str = "qa_agent", trust: float = 0.7):
        super().__init__(agent_id, role, trust)
        self.task_affinity = {TaskType.QUESTION_ANSWERING}

    def run(self, input_text: str, blackboard: Blackboard, task_type: TaskType = None):
        if not self.enabled or not input_text:
            return
            
        try:
            # Extract potential questions and answers from text
            sentences = safe_sentence_split(input_text)
            
            # Question patterns
            question_indicators = ["what", "who", "when", "where", "why", "how", "which", "can", "do", "is", "are"]
            answer_indicators = ["because", "therefore", "the answer", "result", "conclusion", "solution"]
            
            questions = []
            answers = []
            facts = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Identify questions
                if sentence.strip().endswith('?') or any(qi in sentence_lower[:20] for qi in question_indicators):
                    questions.append(sentence.strip())
                
                # Identify potential answers
                elif any(ai in sentence_lower for ai in answer_indicators):
                    answers.append(sentence.strip())
                
                # Extract factual statements (sentences with numbers, names, or definitions)
                elif any(char.isdigit() for char in sentence) or any(word[0].isupper() for word in sentence.split()):
                    facts.append(sentence.strip())
            
            # Simple QA matching
            qa_pairs = []
            for i, question in enumerate(questions):
                # Try to find answer in subsequent sentences
                answer = answers[i] if i < len(answers) else (facts[i] if i < len(facts) else "No specific answer found.")
                qa_pairs.append({"question": question, "answer": answer})
            
            # Extract key information
            key_info = []
            for sentence in sentences[:5]:  # Focus on first few sentences
                if len(sentence.split()) > 5:  # Substantial sentences
                    key_info.append(sentence.strip())
            
            # Calculate confidence based on information richness
            confidence = min(1.0, (len(questions) * 0.3 + len(answers) * 0.3 + len(facts) * 0.1))
            
            result = {
                "questions": questions,
                "answers": answers,
                "qa_pairs": qa_pairs,
                "key_facts": facts[:10],  # Top 10 facts
                "key_information": key_info,
                "total_questions": len(questions),
                "total_facts": len(facts)
            }
            
            blackboard.post(self.agent_id, "question_answering", result, score=confidence)
            blackboard.set_task_metric("question_answering", "qa_pairs", len(qa_pairs))
            
        except Exception:
            blackboard.post(self.agent_id, "question_answering", {"questions": [], "answers": []}, score=0.0)


############################################################
# Enhanced Task Results Processor with New Tasks
############################################################

class TaskResultsProcessor:
    """Process and format results for different task types."""
    
    # [Keep all existing process_* methods for original tasks]
    
    @staticmethod
    def process_named_entity_recognition(blackboard: Blackboard) -> Dict[str, Any]:
        """Process NER results."""
        ner_posts = blackboard.read("named_entities")
        all_entities = {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "MISC": []}
        
        for post in ner_posts:
            payload = post.get("payload", {})
            entities = payload.get("entities", {})
            
            for category, entity_list in entities.items():
                if category in all_entities:
                    all_entities[category].extend(entity_list)
        
        # Remove duplicates
        for category in all_entities:
            all_entities[category] = list(set(all_entities[category]))
        
        total_entities = sum(len(entities) for entities in all_entities.values())
        metrics = blackboard.get_task_metrics("named_entity_recognition")
        
        return {
            "entities": all_entities,
            "total_entities": total_entities,
            "metrics": {"total_entities": metrics.get("total_entities", 0)},
            "raw_data": {"ner_posts": ner_posts}
        }
    
    @staticmethod
    def process_sentiment_analysis(blackboard: Blackboard) -> Dict[str, Any]:
        """Process sentiment analysis results."""
        sentiment_posts = blackboard.read("sentiment_analysis")
        
        if not sentiment_posts:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Aggregate results from multiple agents
        scores = []
        emotions = []
        
        for post in sentiment_posts:
            payload = post.get("payload", {})
            scores.append(payload.get("sentiment_score", 0.0))
            emotions.append(payload.get("dominant_emotion", "neutral"))
        
        avg_sentiment = sum(scores) / len(scores) if scores else 0.0
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
        
        if avg_sentiment > 0.1:
            sentiment_label = "positive"
        elif avg_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        metrics = blackboard.get_task_metrics("sentiment_analysis")
        
        return {
            "sentiment_score": avg_sentiment,
            "sentiment_label": sentiment_label,
            "dominant_emotion": dominant_emotion,
            "confidence": abs(avg_sentiment),
            "metrics": {"sentiment_strength": metrics.get("sentiment_strength", 0.0)},
            "raw_data": {"sentiment_posts": sentiment_posts}
        }
    
    @staticmethod
    def process_topic_modeling(blackboard: Blackboard) -> Dict[str, Any]:
        """Process topic modeling results."""
        topic_posts = blackboard.read("topic_modeling")
        
        all_topics = {}
        all_themes = []
        
        for post in topic_posts:
            payload = post.get("payload", {})
            topic_scores = payload.get("topic_scores", {})
            themes = payload.get("themes", [])
            
            for topic, score in topic_scores.items():
                all_topics[topic] = all_topics.get(topic, 0) + score
            
            all_themes.extend(themes)
        
        # Get top topics and themes
        sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
        unique_themes = list(set(all_themes))
        
        metrics = blackboard.get_task_metrics("topic_modeling")
        
        return {
            "dominant_topics": sorted_topics[:5],
            "all_topics": all_topics,
            "themes": unique_themes[:10],
            "metrics": {"topics_identified": metrics.get("topics_identified", 0)},
            "raw_data": {"topic_posts": topic_posts}
        }
    
    @staticmethod
    def process_text_classification(blackboard: Blackboard) -> Dict[str, Any]:
        """Process text classification results."""
        classification_posts = blackboard.read("text_classification")
        
        all_categories = {}
        styles = []
        
        for post in classification_posts:
            payload = post.get("payload", {})
            categories = payload.get("primary_categories", [])
            style = payload.get("style", "neutral")
            
            for category in categories:
                all_categories[category] = all_categories.get(category, 0) + 1
            
            styles.append(style)
        
        primary_category = max(all_categories, key=all_categories.get) if all_categories else "general"
        dominant_style = max(set(styles), key=styles.count) if styles else "neutral"
        
        metrics = blackboard.get_task_metrics("text_classification")
        
        return {
            "primary_category": primary_category,
            "all_categories": all_categories,
            "style": dominant_style,
            "metrics": {"categories_detected": metrics.get("categories_detected", 0)},
            "raw_data": {"classification_posts": classification_posts}
        }
    
    @staticmethod
    def process_question_answering(blackboard: Blackboard) -> Dict[str, Any]:
        """Process question answering results."""
        qa_posts = blackboard.read("question_answering")
        
        all_questions = []
        all_answers = []
        all_facts = []
        qa_pairs = []
        
        for post in qa_posts:
            payload = post.get("payload", {})
            all_questions.extend(payload.get("questions", []))
            all_answers.extend(payload.get("answers", []))
            all_facts.extend(payload.get("key_facts", []))
            qa_pairs.extend(payload.get("qa_pairs", []))
        
        metrics = blackboard.get_task_metrics("question_answering")
        
        return {
            "questions": list(set(all_questions)),
            "answers": list(set(all_answers)),
            "key_facts": list(set(all_facts))[:10],
            "qa_pairs": qa_pairs,
            "metrics": {"qa_pairs": metrics.get("qa_pairs", 0)},
            "raw_data": {"qa_posts": qa_posts}
        }


############################################################
# Enhanced Swarm Controller with New Task Support
############################################################

class SwarmController:
    """Enhanced swarm management with new task support."""
    
    def __init__(self, blackboard: Blackboard = None):
        self.blackboard = blackboard or Blackboard()
        self.agents: Dict[str, Agent] = {}
        self.current_task: Optional[TaskType] = None

    # [Keep all existing methods: add_agent, remove_agent, set_trust, enable_agent, run_task_processing, _adaptive_trust_adjustment]

    def configure_for_task(self, task_type: TaskType, num_agents: int):
        """Configure swarm for specific task."""
        self.current_task = task_type
        self.agents.clear()
        
        # Create task-specific agents
        agents = self._create_task_specific_agents(task_type, num_agents)
        
        # Add agents to swarm
        for agent in agents:
            self.agents[agent.agent_id] = agent

    def add_agent(self, agent: Agent):
        """Add agent to swarm."""
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str):
        """Remove agent from swarm."""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def set_trust(self, agent_id: str, trust: float):
        """Set trust level for an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].trust = max(0.0, min(1.0, trust))

    def enable_agent(self, agent_id: str, enabled: bool = True):
        """Enable or disable an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].enabled = enabled

    def run_task_processing(self, input_text: str, rounds: int = 2):
        """Run task processing with agents."""
        if not self.agents:
            return
            
        # Clear blackboard for new task
        self.blackboard = Blackboard()
        
        # Run agents for specified rounds
        for round_num in range(rounds):
            for agent in self.agents.values():
                if agent.enabled:
                    try:
                        agent.run(input_text, self.blackboard, self.current_task)
                    except Exception:
                        continue  # Skip failed agents
                        
            # Adaptive trust adjustment after each round
            self._adaptive_trust_adjustment()

    def _adaptive_trust_adjustment(self):
        """Adjust agent trust based on performance."""
        try:
            # Simple trust adjustment based on post count and scores
            for agent_id, agent in self.agents.items():
                posts = [p for p in self.blackboard.read() if p.get('agent_id') == agent_id]
                if posts:
                    avg_score = sum(p.get('score', 0) for p in posts) / len(posts)
                    # Slightly adjust trust based on performance
                    adjustment = (avg_score - 0.5) * 0.1
                    agent.trust = max(0.1, min(1.0, agent.trust + adjustment))
        except Exception:
            pass

    def _create_task_specific_agents(self, task_type: TaskType, num_agents: int) -> List[Agent]:
        """Create optimal agent mix for specific task including new tasks."""
        try:
            agents = []
            num_agents = max(1, min(20, num_agents))
            
            # New task configurations
            if task_type == TaskType.NAMED_ENTITY_RECOGNITION:
                ner_count = max(1, num_agents // 2)
                for i in range(ner_count):
                    agents.append(NamedEntityRecognitionAgent(f"ner_{i+1}", trust=0.8 - i*0.05))
                
                # Add supporting agents
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
                
                # Add supporting agents for context
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
                
                # Add keyword agents for support
                remaining = num_agents - topic_count
                for i in range(remaining):
                    agents.append(KeywordAgent(f"kw_{i+1}", trust=0.7))
            
            elif task_type == TaskType.TEXT_CLASSIFICATION:
                class_count = max(1, num_agents // 2)
                for i in range(class_count):
                    agents.append(TextClassificationAgent(f"classifier_{i+1}", trust=0.8 - i*0.05))
                
                # Add supporting agents
                remaining = num_agents - class_count
                for i in range(remaining):
                    if i % 2 == 0:
                        agents.append(KeywordAgent(f"kw_{i//2 + 1}", trust=0.7))
                    else:
                        agents.append(SentenceRankAgent(f"sent_{i//2 + 1}", trust=0.6))
            
            elif task_type == TaskType.QUESTION_ANSWERING:
                qa_count = max(1, min(num_agents // 2, 3))  # Limit QA agents
                for i in range(qa_count):
                    agents.append(QuestionAnsweringAgent(f"qa_{i+1}", trust=0.8 - i*0.05))
                
                # Add supporting agents
                remaining = num_agents - qa_count
                for i in range(remaining):
                    if i % 3 == 0:
                        agents.append(SentenceRankAgent(f"sent_{i//3 + 1}", trust=0.7))
                    elif i % 3 == 1:
                        agents.append(KeywordAgent(f"kw_{i//3 + 1}", trust=0.7))
                    else:
                        agents.append(NamedEntityRecognitionAgent(f"ner_{i//3 + 1}", trust=0.6))
            
            # [Keep existing task configurations for original 5 tasks]
            elif task_type == TaskType.SUMMARIZATION:
                sent_count = max(1, num_agents // 2)
                for i in range(sent_count):
                    agents.append(SentenceRankAgent(f"sent_{i+1}", trust=0.8 - i*0.05))
                
                # Add keyword agents for support
                remaining = num_agents - sent_count
                for i in range(remaining):
                    agents.append(KeywordAgent(f"kw_{i+1}", trust=0.7))
            
            elif task_type == TaskType.KEYWORD_EXTRACTION:
                kw_count = max(1, num_agents // 2)
                for i in range(kw_count):
                    agents.append(KeywordAgent(f"kw_{i+1}", trust=0.8 - i*0.05))
                
                # Add sentence agents for context
                remaining = num_agents - kw_count
                for i in range(remaining):
                    agents.append(SentenceRankAgent(f"sent_{i+1}", trust=0.7))
            
            elif task_type == TaskType.REDUNDANCY_ANALYSIS:
                # Use sentence ranking agents for redundancy analysis
                for i in range(num_agents):
                    agents.append(SentenceRankAgent(f"redundancy_{i+1}", trust=0.8 - i*0.05))
            
            elif task_type == TaskType.GRAMMAR_CHECK:
                # Use sentence ranking agents for grammar checking
                for i in range(num_agents):
                    agents.append(SentenceRankAgent(f"grammar_{i+1}", trust=0.8 - i*0.05))
            
            elif task_type == TaskType.COVERAGE_SCORING:
                # Mix of keyword and sentence agents
                kw_count = max(1, num_agents // 2)
                for i in range(kw_count):
                    agents.append(KeywordAgent(f"coverage_kw_{i+1}", trust=0.8 - i*0.05))
                
                remaining = num_agents - kw_count
                for i in range(remaining):
                    agents.append(SentenceRankAgent(f"coverage_sent_{i+1}", trust=0.7))
            
            # Default case
            else:
                # Fallback: create sentence ranking agents
                for i in range(num_agents):
                    agents.append(SentenceRankAgent(f"fallback_{i+1}", trust=0.7))
            
            return agents[:num_agents]
            
        except Exception:
            return [SentenceRankAgent("fallback_sent", trust=0.5)]

    def get_task_results(self, k: int = 3) -> Dict[str, Any]:
        """Get formatted results for current task including new tasks."""
        try:
            if not self.current_task:
                return {}
            
            # New task result processing
            if self.current_task == TaskType.NAMED_ENTITY_RECOGNITION:
                return TaskResultsProcessor.process_named_entity_recognition(self.blackboard)
            elif self.current_task == TaskType.SENTIMENT_ANALYSIS:
                return TaskResultsProcessor.process_sentiment_analysis(self.blackboard)
            elif self.current_task == TaskType.TOPIC_MODELING:
                return TaskResultsProcessor.process_topic_modeling(self.blackboard)
            elif self.current_task == TaskType.TEXT_CLASSIFICATION:
                return TaskResultsProcessor.process_text_classification(self.blackboard)
            elif self.current_task == TaskType.QUESTION_ANSWERING:
                return TaskResultsProcessor.process_question_answering(self.blackboard)
            
            # [Keep existing task result processing]
            elif self.current_task == TaskType.SUMMARIZATION:
                return TaskResultsProcessor.process_summarization_results(self.blackboard, k)
            elif self.current_task == TaskType.KEYWORD_EXTRACTION:
                return TaskResultsProcessor.process_keyword_extraction_results(self.blackboard, k)
            elif self.current_task == TaskType.REDUNDANCY_ANALYSIS:
                return TaskResultsProcessor.process_redundancy_analysis_results(self.blackboard, k)
            elif self.current_task == TaskType.GRAMMAR_CHECK:
                return TaskResultsProcessor.process_grammar_check_results(self.blackboard, k)
            elif self.current_task == TaskType.COVERAGE_SCORING:
                return TaskResultsProcessor.process_coverage_scoring_results(self.blackboard, k)
            
            else:
                return {}
                
        except Exception:
            return {}


############################################################
# Enhanced User Interface with New Tasks
############################################################

class SwarmMindInterface:
    """Interactive command-line interface with new NLP tasks."""
    
    def __init__(self):
        self.swarm = SwarmController()
        
    def _select_task(self) -> Optional[TaskType]:
        """Enhanced task selection menu with new tasks."""
        print("\n📋 SELECT NLP TASK:")
        print("1. 📝 Summarization (extract key sentences)")
        print("2. 🔍 Keyword Extraction (find important terms)")
        print("3. 🔄 Redundancy Analysis (detect duplicate content)")
        print("4. ✅ Grammar & Fluency Check (assess text quality)")
        print("5. 📊 Coverage & Importance Scoring (measure topic coverage)")
        print("6. 👤 Named Entity Recognition (identify people, places, organizations)")
        print("7. 😊 Sentiment Analysis & Emotion Detection (analyze emotional tone)")
        print("8. 🏷️  Topic Modeling & Theme Detection (discover topics and themes)")
        print("9. 📂 Text Classification & Categorization (classify document type)")
        print("10. ❓ Question Answering (extract answers and key information)")
        print("11. 🚪 Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-11): ").strip()
                
                task_map = {
                    "1": TaskType.SUMMARIZATION,
                    "2": TaskType.KEYWORD_EXTRACTION,
                    "3": TaskType.REDUNDANCY_ANALYSIS,
                    "4": TaskType.GRAMMAR_CHECK,
                    "5": TaskType.COVERAGE_SCORING,
                    "6": TaskType.NAMED_ENTITY_RECOGNITION,
                    "7": TaskType.SENTIMENT_ANALYSIS,
                    "8": TaskType.TOPIC_MODELING,
                    "9": TaskType.TEXT_CLASSIFICATION,
                    "10": TaskType.QUESTION_ANSWERING,
                    "11": None
                }
                
                if choice in task_map:
                    return task_map[choice]
                else:
                    print("❌ Invalid choice. Please enter 1-11.")
                    
            except Exception:
                print("❌ Invalid input. Please try again.")

    def _display_results(self, task_type: TaskType, results: Dict[str, Any], num_agents: int):
        """Enhanced display method for all tasks including new ones."""
        print("\n" + "=" * 70)
        print(f"📊 RESULTS: {task_type.value.upper()} ({num_agents} agents)")
        print("=" * 70)
        
        # New task display methods
        if task_type == TaskType.NAMED_ENTITY_RECOGNITION:
            self._display_ner_results(results)
        elif task_type == TaskType.SENTIMENT_ANALYSIS:
            self._display_sentiment_results(results)
        elif task_type == TaskType.TOPIC_MODELING:
            self._display_topic_results(results)
        elif task_type == TaskType.TEXT_CLASSIFICATION:
            self._display_classification_results(results)
        elif task_type == TaskType.QUESTION_ANSWERING:
            self._display_qa_results(results)
        # [Keep existing display methods for original 5 tasks]

    def _display_ner_results(self, results: Dict[str, Any]):
        """Display Named Entity Recognition results."""
        entities = results.get("entities", {})
        total_entities = results.get("total_entities", 0)
        
        print("\n👤 NAMED ENTITIES DETECTED:")
        for category, entity_list in entities.items():
            if entity_list:
                print(f"\n  {category}:")
                for entity in entity_list[:10]:  # Show top 10 per category
                    print(f"    • {entity}")
        
        print(f"\n📈 NER METRICS:")
        print(f"  🎯 Total Entities: {total_entities}")
        print(f"  📂 Categories Found: {len([cat for cat, ents in entities.items() if ents])}")
        
        if total_entities > 10:
            print("  ✅ Rich entity content detected")
        elif total_entities > 5:
            print("  ⚠️  Moderate entity content")
        else:
            print("  ❌ Limited entity content - consider longer text")

    def _display_sentiment_results(self, results: Dict[str, Any]):
        """Display Sentiment Analysis results."""
        sentiment_label = results.get("sentiment_label", "neutral")
        sentiment_score = results.get("sentiment_score", 0.0)
        dominant_emotion = results.get("dominant_emotion", "neutral")
        confidence = results.get("confidence", 0.0)
        
        print("\n😊 SENTIMENT ANALYSIS:")
        
        # Sentiment visualization
        sentiment_emoji = {"positive": "😊", "negative": "😢", "neutral": "😐"}
        emotion_emoji = {"joy": "😄", "anger": "😠", "sadness": "😢", "fear": "😰", "surprise": "😲", "neutral": "😐"}
        
        print(f"  {sentiment_emoji.get(sentiment_label, '😐')} Overall Sentiment: {sentiment_label.upper()}")
        print(f"  📊 Sentiment Score: {sentiment_score:.3f} (range: -1 to +1)")
        print(f"  {emotion_emoji.get(dominant_emotion, '😐')} Dominant Emotion: {dominant_emotion}")
        print(f"  🎯 Confidence: {confidence:.1%}")
        
        # Assessment
        if abs(sentiment_score) > 0.5:
            print(f"  ✅ Strong {sentiment_label} sentiment detected")
        elif abs(sentiment_score) > 0.2:
            print(f"  ⚠️  Moderate {sentiment_label} sentiment")
        else:
            print("  ❌ Neutral or mixed sentiment")

    def _display_topic_results(self, results: Dict[str, Any]):
        """Display Topic Modeling results."""
        dominant_topics = results.get("dominant_topics", [])
        themes = results.get("themes", [])
        
        print("\n🏷️ TOPIC MODELING RESULTS:")
        
        if dominant_topics:
            print("  📊 Dominant Topics:")
            for topic, score in dominant_topics:
                print(f"    • {topic.title()}: {score} relevance points")
        else:
            print("    No specific topics identified")
        
        if themes:
            print("\n  🎨 Key Themes:")
            for theme in themes[:5]:
                print(f"    • {theme}")
        
        print(f"\n📈 TOPIC METRICS:")
        print(f"  🎯 Topics Identified: {len(dominant_topics)}")
        print(f"  🎨 Themes Extracted: {len(themes)}")
        
        if len(dominant_topics) > 2:
            print("  ✅ Rich topical content")
        elif len(dominant_topics) > 0:
            print("  ⚠️  Some topical content identified")
        else:
            print("  ❌ Limited topical content")

    def _display_classification_results(self, results: Dict[str, Any]):
        """Display Text Classification results."""
        primary_category = results.get("primary_category", "general")
        all_categories = results.get("all_categories", {})
        style = results.get("style", "neutral")
        
        print("\n📂 TEXT CLASSIFICATION:")
        print(f"  📝 Primary Category: {primary_category.title()}")
        print(f"  🎨 Writing Style: {style.title()}")
        
        if all_categories:
            print("\n  📊 All Categories Detected:")
            for category, score in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {category.title()}: {score} indicators")
        
        print(f"\n📈 CLASSIFICATION METRICS:")
        print(f"  🎯 Categories Detected: {len(all_categories)}")
        
        if len(all_categories) > 2:
            print("  ✅ Clear categorical indicators")
        elif len(all_categories) > 0:
            print("  ⚠️  Some categorical indicators")
        else:
            print("  ❌ Generic content - difficult to classify")

    def _display_qa_results(self, results: Dict[str, Any]):
        """Display Question Answering results."""
        questions = results.get("questions", [])
        key_facts = results.get("key_facts", [])
        qa_pairs = results.get("qa_pairs", [])
        
        print("\n❓ QUESTION ANSWERING ANALYSIS:")
        
        if questions:
            print(f"\n  ❓ Questions Identified ({len(questions)}):")
            for question in questions[:5]:
                print(f"    • {question}")
        
        if qa_pairs:
            print(f"\n  💬 Question-Answer Pairs ({len(qa_pairs)}):")
            for pair in qa_pairs[:3]:
                print(f"    Q: {pair['question']}")
                print(f"    A: {pair['answer'][:100]}{'...' if len(pair['answer']) > 100 else ''}")
                print()
        
        if key_facts:
            print(f"\n  📋 Key Facts Extracted ({len(key_facts)}):")
            for fact in key_facts[:5]:
                print(f"    • {fact[:80]}{'...' if len(fact) > 80 else ''}")
        
        print(f"\n📈 QA METRICS:")
        print(f"  ❓ Questions Found: {len(questions)}")
        print(f"  📋 Key Facts: {len(key_facts)}")
        print(f"  💬 QA Pairs: {len(qa_pairs)}")
        
        if len(questions) > 2 or len(key_facts) > 5:
            print("  ✅ Information-rich content")
        elif len(questions) > 0 or len(key_facts) > 2:
            print("  ⚠️  Some extractable information")
        else:
            print("  ❌ Limited extractable information")

    # [Keep all other existing methods: run, _select_agent_count, _get_input_text, _ask_for_adjustment, _ask_continue]

    def run(self):
        """Main interactive loop."""
        print("🐝 Welcome to SwarmMind - Multi-Agent NLP Framework")
        print("Enhanced with 10 specialized NLP tasks")
        
        while True:
            try:
                # Step 1: Task selection
                task_type = self._select_task()
                if not task_type:
                    break
                
                # Step 2: Agent count selection
                num_agents = self._select_agent_count()
                
                # Step 3: Input text
                input_text = self._get_input_text()
                if not input_text.strip():
                    print("❌ No text provided. Please try again.")
                    continue
                
                # Step 4: Configure and run swarm
                print(f"\n🔧 Configuring swarm with {num_agents} agents for {task_type.value}...")
                self.swarm.configure_for_task(task_type, num_agents)
                
                print(f"✅ Added {len(self.swarm.agents)} agents to swarm")
                print("🚀 Running swarm processing...")
                
                self.swarm.run_task_processing(input_text, rounds=2)
                
                # Step 5: Display results
                results = self.swarm.get_task_results()
                self._display_results(task_type, results, num_agents)
                
                # Step 6: Continue or exit
                if self._ask_continue():
                    continue
                else:
                    break
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for using SwarmMind.")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                if self._ask_continue():
                    continue
                else:
                    break

    def _select_agent_count(self) -> int:
        """Agent count selection."""
        print("\n🤖 SELECT NUMBER OF AGENTS (1-20):")
        print("💡 More agents = more thorough analysis but slower processing")
        print("💡 Recommended: 5-10 agents for balanced performance")
        
        while True:
            try:
                count = input("Enter number of agents (default: 5): ").strip()
                
                if not count:
                    return 5
                
                count = int(count)
                if 1 <= count <= 20:
                    return count
                else:
                    print("❌ Please enter a number between 1 and 20.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")

    def _get_input_text(self) -> str:
        """Get input text from user."""
        print("\n📝 ENTER TEXT TO ANALYZE:")
        print("💡 You can paste multiple lines. Press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done.")
        print("💡 Or enter a single line and press Enter twice.")
        print()
        
        lines = []
        empty_line_count = 0
        
        try:
            while True:
                line = input()
                if not line.strip():
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                else:
                    empty_line_count = 0
                    lines.append(line)
        except EOFError:
            pass
        
        return "\n".join(lines)

    def _ask_continue(self) -> bool:
        """Ask if user wants to continue with another task."""
        while True:
            try:
                choice = input("\n🔄 Would you like to analyze another text? (y/n, default: y): ").strip().lower()
                
                if not choice or choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no.")
                    
            except Exception:
                print("❌ Invalid input. Please try again.")


############################################################
# Enhanced Demo with All Tasks
############################################################

def demo_all_tasks():
    """Comprehensive demo showing all task types including new ones."""
    demo_text = (
        "Apple Inc. is a technology company based in Cupertino, California. "
        "The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
        "Apple is known for innovative products like the iPhone, iPad, and Mac computers. "
        "The company's success has made it one of the most valuable corporations in the world. "
        "Tim Cook currently serves as the CEO, leading the company to new heights. "
        "What makes Apple special is its focus on design and user experience. "
        "However, some critics argue that their products are overpriced. "
        "Despite this, Apple continues to dominate the premium technology market. "
        "The company's headquarters, Apple Park, opened in 2017 in California. "
        "Apple's commitment to innovation drives its research and development efforts."
    )
    
    print("🚀 SwarmMind Framework - Comprehensive Demo (All 10 Tasks)")
    print("=" * 70)
    
    for task_type in TaskType:
        print(f"\n{'='*70}")
        print(f"📋 TASK: {task_type.value.upper()}")
        print(f"{'='*70}")
        
        try:
            swarm = SwarmController()
            swarm.configure_for_task(task_type, 6)  # Use 6 agents for demo
            
            print(f"🤖 Configured {len(swarm.agents)} agents for {task_type.value}")
            
            swarm.run_task_processing(demo_text)
            results = swarm.get_task_results()
            
            interface = SwarmMindInterface()
            interface._display_results(task_type, results, len(swarm.agents))
            
        except Exception as e:
            print(f"❌ Error in {task_type.value}: {e}")
            continue


############################################################
# Main Entry Point
############################################################

def main():
    """Main entry point with user choice."""
    print("🐝 SwarmMind Framework - Enhanced with 10 NLP Tasks")
    print("Choose mode:")
    print("1. 🎮 Interactive Mode")
    print("2. 🧪 Demo Mode (all 10 tasks)")
    
    try:
        choice = input("Enter choice (1-2, default: 1): ").strip()
        
        if choice == "2":
            demo_all_tasks()
        else:
            interface = SwarmMindInterface()
            interface.run()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()