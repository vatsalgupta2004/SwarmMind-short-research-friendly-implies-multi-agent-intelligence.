from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from agents_swarm import (
    TaskType, Blackboard, Agent,
    KeywordAgent, SentenceRankAgent, NamedEntityRecognitionAgent, 
    SentimentAnalysisAgent, TopicModelingAgent, TextClassificationAgent,
    QuestionAnsweringAgent, RedundancyAgent, GrammarAgent, CoverageAgent,
    TaskResultsProcessor
)

class HiveRouter:
    """Enhanced router with intelligent agent selection"""
    
    def __init__(self):
        self.agent_templates = {
            TaskType.KEYWORD_EXTRACTION: self._build_keyword_agents,
            TaskType.SUMMARIZATION: self._build_summarization_agents,
            TaskType.SENTIMENT_ANALYSIS: self._build_sentiment_agents,
            TaskType.NAMED_ENTITY_RECOGNITION: self._build_ner_agents,
            TaskType.TOPIC_MODELING: self._build_topic_agents,
            TaskType.TEXT_CLASSIFICATION: self._build_classification_agents,
            TaskType.QUESTION_ANSWERING: self._build_qa_agents,
            TaskType.REDUNDANCY_ANALYSIS: self._build_redundancy_agents,
            TaskType.GRAMMAR_CHECK: self._build_grammar_agents,
            TaskType.COVERAGE_SCORING: self._build_coverage_agents,
        }
    
    def build(self, task: TaskType, num_agents: int) -> List[Agent]:
        """Build optimal agent configuration for task"""
        builder = self.agent_templates.get(task, self._build_default_agents)
        return builder(num_agents)
    
    def _build_keyword_agents(self, num_agents: int) -> List[Agent]:
        """Build keyword extraction focused agents"""
        agents = []
        primary_count = max(1, num_agents // 2)
        
        # Primary keyword agents
        for i in range(primary_count):
            agents.append(KeywordAgent(f"kw_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support agents
        remaining = num_agents - primary_count
        for i in range(remaining):
            agents.append(SentenceRankAgent(f"sent_support_{i+1}", trust=0.7))
        
        return agents[:num_agents]
    
    def _build_summarization_agents(self, num_agents: int) -> List[Agent]:
        """Build summarization focused agents"""
        agents = []
        sent_count = max(1, num_agents // 2)
        
        # Primary sentence ranking agents
        for i in range(sent_count):
            agents.append(SentenceRankAgent(f"sent_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support keyword agents
        remaining = num_agents - sent_count
        for i in range(remaining):
            agents.append(KeywordAgent(f"kw_support_{i+1}", trust=0.7))
        
        return agents[:num_agents]
    
    def _build_sentiment_agents(self, num_agents: int) -> List[Agent]:
        """Build sentiment analysis focused agents"""
        agents = []
        sentiment_count = max(1, num_agents // 2)
        
        # Primary sentiment agents
        for i in range(sentiment_count):
            agents.append(SentimentAnalysisAgent(f"sentiment_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support agents
        remaining = num_agents - sentiment_count
        for i in range(remaining):
            if i % 2 == 0:
                agents.append(TextClassificationAgent(f"class_support_{i//2 + 1}", trust=0.6))
            else:
                agents.append(KeywordAgent(f"kw_support_{i//2 + 1}", trust=0.6))
        
        return agents[:num_agents]
    
    def _build_ner_agents(self, num_agents: int) -> List[Agent]:
        """Build named entity recognition focused agents"""
        agents = []
        ner_count = max(1, num_agents // 2)
        
        # Primary NER agents
        for i in range(ner_count):
            agents.append(NamedEntityRecognitionAgent(f"ner_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support agents
        remaining = num_agents - ner_count
        for i in range(remaining):
            if i % 2 == 0:
                agents.append(KeywordAgent(f"kw_support_{i//2 + 1}", trust=0.7))
            else:
                agents.append(SentenceRankAgent(f"sent_support_{i//2 + 1}", trust=0.6))
        
        return agents[:num_agents]
    
    def _build_topic_agents(self, num_agents: int) -> List[Agent]:
        """Build topic modeling focused agents"""
        agents = []
        topic_count = max(1, num_agents // 2)
        
        # Primary topic agents
        for i in range(topic_count):
            agents.append(TopicModelingAgent(f"topic_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support keyword agents
        remaining = num_agents - topic_count
        for i in range(remaining):
            agents.append(KeywordAgent(f"kw_support_{i+1}", trust=0.7))
        
        return agents[:num_agents]
    
    def _build_classification_agents(self, num_agents: int) -> List[Agent]:
        """Build text classification focused agents"""
        agents = []
        class_count = max(1, num_agents // 2)
        
        # Primary classification agents
        for i in range(class_count):
            agents.append(TextClassificationAgent(f"class_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support agents
        remaining = num_agents - class_count
        for i in range(remaining):
            if i % 2 == 0:
                agents.append(KeywordAgent(f"kw_support_{i//2 + 1}", trust=0.7))
            else:
                agents.append(SentenceRankAgent(f"sent_support_{i//2 + 1}", trust=0.6))
        
        return agents[:num_agents]
    
    def _build_qa_agents(self, num_agents: int) -> List[Agent]:
        """Build question answering focused agents"""
        agents = []
        qa_count = max(1, min(num_agents // 2, 3))
        
        # Primary QA agents
        for i in range(qa_count):
            agents.append(QuestionAnsweringAgent(f"qa_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support agents
        remaining = num_agents - qa_count
        for i in range(remaining):
            if i % 3 == 0:
                agents.append(SentenceRankAgent(f"sent_support_{i//3 + 1}", trust=0.7))
            elif i % 3 == 1:
                agents.append(KeywordAgent(f"kw_support_{i//3 + 1}", trust=0.7))
            else:
                agents.append(NamedEntityRecognitionAgent(f"ner_support_{i//3 + 1}", trust=0.6))
        
        return agents[:num_agents]
    
    def _build_redundancy_agents(self, num_agents: int) -> List[Agent]:
        """Build redundancy analysis focused agents"""
        agents = []
        for i in range(num_agents):
            agents.append(RedundancyAgent(f"redundancy_{i+1}", trust=0.8 - i*0.05))
        return agents
    
    def _build_grammar_agents(self, num_agents: int) -> List[Agent]:
        """Build grammar check focused agents"""
        agents = []
        for i in range(num_agents):
            agents.append(GrammarAgent(f"grammar_{i+1}", trust=0.8 - i*0.05))
        return agents
    
    def _build_coverage_agents(self, num_agents: int) -> List[Agent]:
        """Build coverage scoring focused agents"""
        agents = []
        cov_count = max(1, num_agents // 2)
        
        # Primary coverage agents
        for i in range(cov_count):
            agents.append(CoverageAgent(f"coverage_primary_{i+1}", trust=0.8 - i*0.05))
        
        # Support keyword agents
        remaining = num_agents - cov_count
        for i in range(remaining):
            agents.append(KeywordAgent(f"kw_support_{i+1}", trust=0.7))
        
        return agents[:num_agents]
    
    def _build_default_agents(self, num_agents: int) -> List[Agent]:
        """Build default mixed agent configuration"""
        agents = []
        for i in range(num_agents):
            if i % 3 == 0:
                agents.append(KeywordAgent(f"default_kw_{i//3 + 1}", trust=0.7))
            elif i % 3 == 1:
                agents.append(SentenceRankAgent(f"default_sent_{i//3 + 1}", trust=0.7))
            else:
                agents.append(NamedEntityRecognitionAgent(f"default_ner_{i//3 + 1}", trust=0.6))
        return agents[:num_agents]
    
    def collect_results(self, task: TaskType, blackboard: Blackboard) -> Dict[str, Any]:
        """Collect and format task results - FIXED VERSION"""
        processor_map = {
            TaskType.SUMMARIZATION: TaskResultsProcessor.process_summarization_results,
            TaskType.KEYWORD_EXTRACTION: TaskResultsProcessor.process_keyword_extraction_results,
            TaskType.NAMED_ENTITY_RECOGNITION: TaskResultsProcessor.process_named_entity_recognition,
            TaskType.SENTIMENT_ANALYSIS: TaskResultsProcessor.process_sentiment_analysis,
            TaskType.TOPIC_MODELING: TaskResultsProcessor.process_topic_modeling,
            TaskType.TEXT_CLASSIFICATION: TaskResultsProcessor.process_text_classification,
            TaskType.QUESTION_ANSWERING: TaskResultsProcessor.process_question_answering,
            TaskType.REDUNDANCY_ANALYSIS: TaskResultsProcessor.process_redundancy_analysis_results,
            TaskType.GRAMMAR_CHECK: TaskResultsProcessor.process_grammar_check_results,
            TaskType.COVERAGE_SCORING: TaskResultsProcessor.process_coverage_scoring_results,
        }
        
        processor = processor_map.get(task, lambda bb: {"message": "Task completed"})
        # FIXED: Call with only blackboard argument (no second parameter)
        return processor(blackboard)

class TelemetryBus:
    """Enhanced telemetry with real-time metrics"""
    
    def __init__(self):
        self.completed = 0
        self.failed = 0
        self.agent_times: List[float] = []
        self.round_data: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.start_time = None
        self.active_agents = 0
    
    def record(self, success: bool, elapsed: float, agent_id: str = None):
        """Record agent execution"""
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            self.agent_times.append(elapsed)
    
    def start_round(self, round_num: int, agent_count: int):
        """Start tracking a round"""
        with self._lock:
            self.round_data.append({
                "round": round_num,
                "start_time": time.time(),
                "agent_count": agent_count,
                "completed": 0,
                "failed": 0
            })
            self.active_agents = agent_count
    
    def end_round(self, round_num: int):
        """End tracking a round"""
        with self._lock:
            if self.round_data and self.round_data[-1]["round"] == round_num:
                self.round_data[-1]["end_time"] = time.time()
                self.round_data[-1]["duration"] = (
                    self.round_data[-1]["end_time"] - self.round_data[-1]["start_time"]
                )
            self.active_agents = 0
    
    def snapshot(self, duration: float = None, agents: int = 0, rounds: int = 0) -> Dict[str, Any]:
        """Get comprehensive metrics snapshot"""
        with self._lock:
            total = max(1, self.completed + self.failed)
            avg_time = sum(self.agent_times) / len(self.agent_times) if self.agent_times else 0.0
            
            return {
                "duration": duration or 0.0,
                "throughput": self.completed / max(0.001, duration or 1.0),
                "success_rate": (self.completed / total) * 100.0,
                "avg_agent_time": avg_time,
                "total_agents": agents,
                "rounds_completed": rounds,
                "round_data": list(self.round_data),
                "active_agents": self.active_agents
            }

class ParallelRunner:
    """Enhanced parallel execution with real-time monitoring"""
    
    def __init__(self, blackboard: Blackboard, agents: List[Agent], task: TaskType, telemetry: TelemetryBus):
        self.blackboard = blackboard
        self.agents = agents
        self.task = task
        self.telemetry = telemetry
        self.execution_callbacks = []
    
    def add_callback(self, callback: callable):
        """Add execution callback for real-time updates"""
        self.execution_callbacks.append(callback)
    
    def run_round(self, text: str) -> Tuple[int, int]:
        """Execute one round of parallel processing"""
        # Clear channels
        channels = ["keywords", "sent_scores", "redundancy", "fluency", "coverage_pref",
                   "named_entities", "sentiment_analysis", "topic_modeling",
                   "text_classification", "question_answering"]
        
        for channel in channels:
            self.blackboard.clear_channel(channel)
        
        suitable_agents = [a for a in self.agents if a.enabled and a.is_suitable_for_task(self.task)]
        if not suitable_agents:
            return 0, 0
        
        self.telemetry.start_round(1, len(suitable_agents))
        
        completed, failed = 0, 0
        max_workers = min(len(suitable_agents), 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_agent = {
                executor.submit(self._safe_run, agent, text): agent
                for agent in suitable_agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                success, elapsed = future.result()
                
                self.telemetry.record(success, elapsed, agent.agent_id)
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                # Notify callbacks
                for callback in self.execution_callbacks:
                    try:
                        callback({
                            "agent_id": agent.agent_id,
                            "success": success,
                            "elapsed": elapsed,
                            "progress": (completed + failed) / len(suitable_agents)
                        })
                    except Exception:
                        pass
        
        self.telemetry.end_round(1)
        return completed, failed
    
    def _safe_run(self, agent: Agent, text: str) -> Tuple[bool, float]:
        """Safely execute agent with timing"""
        start_time = time.time()
        try:
            agent.run(text, self.blackboard, self.task)
            elapsed = time.time() - start_time
            return True, elapsed
        except Exception:
            elapsed = time.time() - start_time
            return False, elapsed

class HiveVisualizer:
    """Visualization helper for Streamlit integration"""
    
    @staticmethod
    def create_agent_network_data(agents: List[Agent]) -> Dict[str, Any]:
        """Create network visualization data"""
        nodes = []
        edges = []
        
        for i, agent in enumerate(agents):
            nodes.append({
                "id": agent.agent_id,
                "label": agent.role,
                "group": agent.role,
                "value": agent.trust * 10,
                "title": f"Trust: {agent.trust:.2f}\nRole: {agent.role}"
            })
        
        # Create edges based on task affinity
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                if len(agent1.task_affinity & agent2.task_affinity) > 0:
                    edges.append({
                        "from": agent1.agent_id,
                        "to": agent2.agent_id,
                        "value": len(agent1.task_affinity & agent2.task_affinity)
                    })
        
        return {"nodes": nodes, "edges": edges}
    
    @staticmethod
    def create_performance_timeline(telemetry: TelemetryBus) -> Dict[str, Any]:
        """Create performance timeline data"""
        timeline_data = []
        for round_data in telemetry.round_data:
            timeline_data.append({
                "round": round_data["round"],
                "duration": round_data.get("duration", 0),
                "completed": round_data.get("completed", 0),
                "failed": round_data.get("failed", 0)
            })
        
        return {"timeline": timeline_data}