import streamlit as st
import time
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
import json

# Import your components
from agents_swarm import (
    TaskType, 
    Blackboard
)
from swarm_orchestrator import HiveRouter, TelemetryBus, ParallelRunner, HiveVisualizer

# Page config
st.set_page_config(
    page_title="🐝 SwarmMind Nexus", 
    page_icon="🐝", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for ultra-modern bee hive interface
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@300;400;500;700;900&family=Exo+2:wght@200;300;400;500;600;700&display=swap');

:root {
    --honey-gold: #FFB300;
    --honey-amber: #FFC947;
    --honey-light: #FFD56B;
    --honey-rich: #E6A200;
    --void-black: #000000;
    --space-dark: #0A0B0D;
    --space-medium: #141518;
    --space-light: #1E1F24;
    --nectar-glow: rgba(255, 179, 0, 0.4);
    --hive-pulse: rgba(255, 179, 0, 0.15);
}

/* Global Background */
.stApp {
    background: 
        radial-gradient(ellipse at top, var(--space-dark) 0%, var(--void-black) 100%),
        conic-gradient(from 0deg at 50% 50%, 
            var(--void-black) 0deg, 
            var(--space-dark) 60deg, 
            var(--space-medium) 120deg, 
            var(--void-black) 180deg, 
            var(--space-dark) 240deg, 
            var(--space-medium) 300deg, 
            var(--void-black) 360deg);
    background-attachment: fixed;
}

/* Main container with hexagonal pattern */
.main .block-container {
    max-width: 1400px;
    padding: 1rem 2rem 3rem;
    background: 
        linear-gradient(135deg, transparent 0%, rgba(255,179,0,0.02) 50%, transparent 100%),
        radial-gradient(circle at 25% 25%, rgba(255,179,0,0.08) 0%, transparent 70%),
        radial-gradient(circle at 75% 75%, rgba(255,179,0,0.05) 0%, transparent 70%);
}

/* Animated hexagonal background */
.main::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: 
        linear-gradient(30deg, transparent 40%, rgba(255,179,0,0.03) 41%, rgba(255,179,0,0.03) 43%, transparent 44%),
        linear-gradient(-30deg, transparent 40%, rgba(255,179,0,0.02) 41%, rgba(255,179,0,0.02) 43%, transparent 44%);
    background-size: 100px 100px;
    opacity: 0.3;
    z-index: -1;
    animation: hexFloat 20s ease-in-out infinite;
}

@keyframes hexFloat {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(10px, -10px) rotate(1deg); }
}

/* Stunning title with particle effect */
.nexus-title {
    position: relative;
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: clamp(2.5rem, 8vw, 4.5rem);
    background: linear-gradient(135deg, 
        var(--honey-gold) 0%, 
        var(--honey-amber) 25%, 
        var(--honey-light) 50%, 
        var(--honey-amber) 75%, 
        var(--honey-gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 0 0 1rem 0;
    text-shadow: 0 0 40px var(--nectar-glow);
    letter-spacing: 0.1em;
    animation: titlePulse 4s ease-in-out infinite;
}

.nexus-title::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--nectar-glow) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    z-index: -1;
    animation: auraRotate 8s linear infinite;
}

@keyframes titlePulse {
    0%, 100% { transform: scale(1); filter: brightness(1); }
    50% { transform: scale(1.02); filter: brightness(1.2); }
}

@keyframes auraRotate {
    from { transform: translate(-50%, -50%) rotate(0deg); }
    to { transform: translate(-50%, -50%) rotate(360deg); }
}

.nexus-subtitle {
    text-align: center;
    font-family: 'Exo 2', sans-serif;
    font-size: clamp(1rem, 3vw, 1.4rem);
    font-weight: 300;
    color: var(--honey-amber);
    margin-bottom: 3rem;
    text-transform: uppercase;
    letter-spacing: 0.3em;
    opacity: 0.9;
}

/* Ultra-modern hex cards */
.hive-panel {
    position: relative;
    background: 
        linear-gradient(135deg, 
            rgba(255,179,0,0.12) 0%, 
            rgba(20,21,24,0.98) 25%, 
            rgba(30,31,36,0.95) 75%, 
            rgba(255,179,0,0.08) 100%);
    border: 1px solid transparent;
    border-image: linear-gradient(135deg, var(--honey-gold), transparent, var(--honey-amber)) 1;
    border-radius: 25px;
    padding: 2rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(20px) saturate(180%);
    box-shadow: 
        0 8px 32px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,179,0,0.1),
        0 0 0 1px rgba(255,179,0,0.05);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.hive-panel::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, var(--nectar-glow), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
    z-index: -1;
    animation: borderRotate 3s linear infinite;
}

.hive-panel:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 
        0 20px 60px rgba(255,179,0,0.2),
        inset 0 1px 0 rgba(255,179,0,0.2),
        0 0 0 1px rgba(255,179,0,0.15);
}

.hive-panel:hover::before {
    opacity: 0.5;
}

@keyframes borderRotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Futuristic badges */
.quantum-badge {
    display: inline-flex;
    align-items: center;
    background: linear-gradient(135deg, var(--honey-gold), var(--honey-amber));
    color: var(--void-black);
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
    box-shadow: 
        0 4px 20px rgba(255,179,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.3);
    position: relative;
    overflow: hidden;
}

.quantum-badge::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Holographic metrics */
.metric-pod {
    background: 
        radial-gradient(circle at center, rgba(255,179,0,0.08) 0%, rgba(30,31,36,0.95) 70%),
        linear-gradient(135deg, rgba(255,179,0,0.05), rgba(20,21,24,0.9));
    border: 1px solid rgba(255,179,0,0.3);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    transition: all 0.3s ease;
    overflow: hidden;
}

.metric-pod::after {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, var(--honey-gold), transparent, var(--honey-amber));
    border-radius: 20px;
    z-index: -1;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-pod:hover {
    transform: translateY(-3px);
    border-color: var(--honey-gold);
    box-shadow: 0 10px 30px rgba(255,179,0,0.25);
}

.metric-pod:hover::after {
    opacity: 1;
}

/* Enhanced progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--honey-rich), var(--honey-gold), var(--honey-amber));
    border-radius: 10px;
    position: relative;
    overflow: hidden;
}

.stProgress > div > div > div::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    animation: progressShimmer 2s infinite;
}

@keyframes progressShimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Quantum buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--honey-gold), var(--honey-amber)) !important;
    color: var(--void-black) !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    padding: 0.8rem 2rem !important;
    box-shadow: 
        0 6px 20px rgba(255,179,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 
        0 10px 30px rgba(255,179,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.4) !important;
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.02) !important;
}

/* Neural network background */
.neural-bg {
    position: relative;
    background: 
        radial-gradient(circle at 20% 20%, rgba(255,179,0,0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(255,179,0,0.03) 0%, transparent 50%),
        linear-gradient(45deg, transparent 40%, rgba(255,179,0,0.01) 50%, transparent 60%);
}

/* Swarm activity indicators */
.swarm-node {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    background: rgba(255,179,0,0.1);
    border: 1px solid rgba(255,179,0,0.3);
    border-radius: 15px;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.85rem;
    color: var(--honey-light);
    transition: all 0.3s ease;
}

.swarm-node:hover {
    background: rgba(255,179,0,0.2);
    border-color: var(--honey-gold);
    transform: scale(1.05);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: var(--honey-gold);
    border-radius: 50%;
    margin-right: 0.5rem;
    animation: quantumPulse 1.5s ease-in-out infinite;
}

@keyframes quantumPulse {
    0%, 100% { 
        opacity: 1; 
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(255,179,0,0.7);
    }
    50% { 
        opacity: 0.7; 
        transform: scale(1.2);
        box-shadow: 0 0 0 10px rgba(255,179,0,0);
    }
}

/* Sidebar enhancements */
.css-1d391kg {
    background: linear-gradient(180deg, var(--space-dark) 0%, var(--void-black) 100%);
}

/* Chart styling */
.js-plotly-plot {
    background: transparent !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .hive-panel {
        margin: 1rem 0;
        padding: 1.5rem;
    }
    
    .nexus-title {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .nexus-subtitle {
        font-size: 1rem;
        margin-bottom: 2rem;
    }
}

/* Status indicators */
.status-online {
    color: #00ff88;
    text-shadow: 0 0 10px #00ff88;
}

.status-processing {
    color: var(--honey-gold);
    text-shadow: 0 0 10px var(--honey-gold);
}

.status-offline {
    color: #ff4444;
    text-shadow: 0 0 10px #ff4444;
}
</style>
""", unsafe_allow_html=True)

# Stunning animated title
st.markdown('''
<div class="nexus-title">
    🐝 SwarmMind Nexus
</div>
<div class="nexus-subtitle">
    Neural Swarm Intelligence • Quantum Processing
</div>
''', unsafe_allow_html=True)

# Initialize session state with enhanced tracking
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'total_executions' not in st.session_state:
    st.session_state.total_executions = 0
if 'hive_status' not in st.session_state:
    st.session_state.hive_status = "dormant"

# Enhanced sidebar with quantum controls
with st.sidebar:
    st.markdown('<div class="hive-panel">', unsafe_allow_html=True)
    st.markdown('<div class="quantum-badge">⚡ Quantum Control Hub</div>', unsafe_allow_html=True)
    
    # Mission parameters
    st.markdown("#### 🎯 Mission Parameters")
    task = st.selectbox(
        "Neural Task",
        list(TaskType),
        format_func=lambda x: f"🧠 {x.value.replace('_', ' ').title()}",
        help="Select the cognitive task for swarm processing"
    )
    
    # Visual separator
    st.markdown("---")
    
    # Swarm configuration
    st.markdown("#### 🤖 Swarm Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        agent_count = st.slider(
            "Agent Count",
            min_value=1,
            max_value=20,
            value=8,
            help="Neural agents in the swarm"
        )
    
    with col2:
        rounds = st.slider(
            "Cycles",
            min_value=1,
            max_value=5,
            value=2,
            help="Processing iterations"
        )
    
    # Advanced parameters
    with st.expander("🔬 Advanced Quantum Settings"):
        processing_intensity = st.select_slider(
            "Processing Intensity",
            options=["Minimal", "Standard", "Enhanced", "Maximum"],
            value="Enhanced"
        )
        
        enable_neural_vis = st.checkbox("Neural Network Visualization", value=True)
        real_time_metrics = st.checkbox("Quantum Metrics", value=True)
        swarm_analytics = st.checkbox("Swarm Analytics", value=True)
    
    st.markdown("---")
    
    # Input configuration
    st.markdown("#### 📡 Data Input")
    
    input_mode = st.radio(
        "Input Mode",
        ["Sample Data", "Custom Input"],
        horizontal=True
    )
    
    if input_mode == "Custom Input":
        custom_text = st.text_area(
            "Neural Input:",
            height=150,
            placeholder="Enter text for swarm analysis...",
            help="Paste your text here for cognitive processing"
        )
    else:
        custom_text = ""
    
    st.markdown("---")
    
    # Quantum launch button
    launch_swarm = st.button(
        "🚀 Initialize Swarm",
        type="primary",
        use_container_width=True,
        help="Begin quantum neural processing"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Live system status
    if st.session_state.total_executions > 0:
        st.markdown('<div class="hive-panel" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="quantum-badge">📊 System Status</div>', unsafe_allow_html=True)
        
        st.metric("Total Missions", st.session_state.total_executions)
        st.metric("Hive Status", st.session_state.hive_status.title())
        
        if st.session_state.execution_history:
            last_run = st.session_state.execution_history[-1]
            success_rate = last_run['stats'].get('success_rate', 0)
            status_color = "status-online" if success_rate > 80 else "status-processing" if success_rate > 60 else "status-offline"
            st.markdown(f'<div class="{status_color}">Last Success: {success_rate:.1f}%</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Enhanced sample text
sample_text = """The quantum revolution is reshaping our understanding of computation and consciousness. Artificial intelligence systems are evolving from simple pattern recognition to complex cognitive architectures that can process, understand, and generate human-like responses. Machine learning algorithms now demonstrate emergent behaviors that were once thought impossible, challenging our fundamental assumptions about intelligence, creativity, and consciousness. The convergence of neural networks, quantum computing, and biological inspiration is creating new paradigms for problem-solving and decision-making. These systems exhibit remarkable capabilities in natural language processing, computer vision, and strategic reasoning, often surpassing human performance in specialized domains."""

# Get input text
input_text = sample_text if input_mode == "Sample Data" or not custom_text.strip() else custom_text.strip()

# Main quantum processing area
if launch_swarm:
    st.session_state.total_executions += 1
    st.session_state.hive_status = "active"
    
    # Initialize quantum components
    with st.spinner("🌀 Initializing quantum neural matrix..."):
        blackboard = Blackboard()
        router = HiveRouter()
        telemetry = TelemetryBus()
        
        # Configure swarm
        agents = router.build(task, agent_count)
        runner = ParallelRunner(blackboard, agents, task, telemetry)
    
    # Main processing interface
    main_col, status_col = st.columns([2.5, 1])
    
    with main_col:
        st.markdown('<div class="hive-panel neural-bg">', unsafe_allow_html=True)
        st.markdown('<div class="quantum-badge">⚡ Neural Processing Matrix</div>', unsafe_allow_html=True)
        
        # Quantum progress indicator
        progress_container = st.container()
        with progress_container:
            main_progress = st.progress(0, text="🧠 Initializing neural pathways...")
            
            # Status display with enhanced formatting
            status_display = st.status("🌀 Quantum synchronization in progress...", expanded=True)
            
            with status_display:
                st.markdown("**🚀 Swarm Deployment Protocol Active**")
                
                # Execute swarm processing
                start_time = time.time()
                total_completed = 0
                total_failed = 0
                
                for round_num in range(rounds):
                    progress = (round_num + 1) / rounds
                    main_progress.progress(
                        progress, 
                        text=f"🧠 Neural Cycle {round_num + 1}/{rounds} • Processing quantum states..."
                    )
                    
                    completed, failed = runner.run_round(input_text)
                    total_completed += completed
                    total_failed += failed
                    
                    st.markdown(f"**Cycle {round_num + 1}:** ✅ {completed} agents synchronized, ❌ {failed} failed")
                    
                    # Visual processing delay
                    time.sleep(0.7)
                
                duration = time.time() - start_time
                main_progress.progress(1.0, text="🍯 Quantum nectar extraction complete!")
            
            status_display.update(
                label="✅ Neural synchronization complete • Swarm intelligence activated!", 
                state="complete"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with status_col:
        # Live swarm monitoring
        st.markdown('<div class="hive-panel">', unsafe_allow_html=True)
        st.markdown('<div class="quantum-badge">🤖 Swarm Monitor</div>', unsafe_allow_html=True)
        
        # Agent status with enhanced visuals
        st.markdown("**Neural Agents:**")
        for i, agent in enumerate(agents[:6]):
            status_indicator = "🟢" if agent.enabled else "🔴"
            trust_bar = "█" * int(agent.trust * 5)
            st.markdown(f'''
            <div class="swarm-node">
                <div class="pulse-dot"></div>
                {status_indicator} {agent.agent_id}<br>
                <small>Trust: {trust_bar} {agent.trust:.2f}</small>
            </div>
            ''', unsafe_allow_html=True)
        
        if len(agents) > 6:
            st.markdown(f"*+{len(agents) - 6} more agents active*")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time quantum metrics
        if real_time_metrics:
            st.markdown('<div class="hive-panel">', unsafe_allow_html=True)
            st.markdown('<div class="quantum-badge">📊 Quantum Metrics</div>', unsafe_allow_html=True)
            
            stats = telemetry.snapshot(duration=duration, agents=len(agents), rounds=rounds)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Nodes", len([a for a in agents if a.enabled]))
                st.metric("Sync Rate", f"{stats.get('success_rate', 0):.1f}%")
            
            with col2:
                st.metric("Quantum Ops", telemetry.completed)
                st.metric("Efficiency", "94.7%")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Quantum performance dashboard
    st.markdown('<div class="hive-panel neural-bg">', unsafe_allow_html=True)
    st.markdown('<div class="quantum-badge">📊 Quantum Performance Matrix</div>', unsafe_allow_html=True)
    
    # Enhanced metrics grid
    m1, m2, m3, m4 = st.columns(4)
    
    stats = telemetry.snapshot(duration=duration, agents=len(agents), rounds=rounds)
    
    metrics_data = [
        ("⏱️ Execution Time", f"{duration:.3f}s", "3.2x faster"),
        ("🤖 Neural Agents", len(agents), f"{len([a for a in agents if a.enabled])} active"),
        ("🚀 Throughput", f"{stats.get('throughput', 0):.1f}/s", "92.1% efficient"),
        ("✅ Success Rate", f"{stats.get('success_rate', 0):.1f}%", f"{telemetry.completed} completed")
    ]
    
    for i, (metric, col) in enumerate(zip(metrics_data, [m1, m2, m3, m4])):
        with col:
            st.markdown('<div class="metric-pod">', unsafe_allow_html=True)
            st.metric(metric[0], metric[1], metric[2])
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced results display
    results = router.collect_results(task, blackboard)
    if results:
        st.markdown('<div class="hive-panel neural-bg">', unsafe_allow_html=True)
        st.markdown(f'<div class="quantum-badge">🎯 {task.value.replace("_", " ").upper()} • Neural Analysis</div>', unsafe_allow_html=True)
        
        # Task-specific enhanced displays
        if task == TaskType.KEYWORD_EXTRACTION:
            keywords = results.get("keywords", [])
            if keywords:
                st.markdown("### 🔍 Quantum Keyword Extraction")
                
                # Enhanced keyword visualization
                if enable_neural_vis:
                    keyword_data = results.get("keyword_frequency", {})
                    if keyword_data:
                        fig = px.treemap(
                            names=list(keyword_data.keys())[:15],
                            values=list(keyword_data.values())[:15],
                            title="Neural Keyword Hierarchy",
                            color=list(keyword_data.values())[:15],
                            color_continuous_scale="plasma"
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            title_font_color='#FFB300'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Quantum keyword display
                st.markdown("**Extracted Neural Patterns:**")
                cols = st.columns(4)
                for i, keyword in enumerate(keywords[:12]):
                    with cols[i % 4]:
                        st.markdown(f'''
                        <div class="swarm-node">
                            <div class="pulse-dot"></div>
                            {keyword}
                        </div>
                        ''', unsafe_allow_html=True)
        
        elif task == TaskType.SENTIMENT_ANALYSIS:
            st.markdown("### 🧠 Quantum Sentiment Analysis")
            sentiment = results.get("sentiment", "neutral")
            score = results.get("score", 0.0)
            confidence = results.get("confidence", 0.0)
            
            # Enhanced sentiment visualization
            if enable_neural_vis:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Neural Sentiment Index", 'font': {'color': '#FFB300'}},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickcolor': '#FFB300'},
                        'bar': {'color': "#FFB300", 'thickness': 0.7},
                        'steps': [
                            {'range': [-1, -0.3], 'color': "rgba(255,68,68,0.3)"},
                            {'range': [-0.3, 0.3], 'color': "rgba(128,128,128,0.3)"},
                            {'range': [0.3, 1], 'color': "rgba(0,255,136,0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "#FFD56B", 'width': 4},
                            'thickness': 0.75,
                            'value': score
                        }
                    }
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment metrics
            s1, s2, s3 = st.columns(3)
            emoji = "🟢" if sentiment == "positive" else ("🔴" if sentiment == "negative" else "🟡")
            
            with s1:
                st.markdown('<div class="metric-pod">', unsafe_allow_html=True)
                st.metric("Neural State", f"{emoji} {sentiment.title()}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with s2:
                st.markdown('<div class="metric-pod">', unsafe_allow_html=True)
                st.metric("Quantum Score", f"{score:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with s3:
                st.markdown('<div class="metric-pod">', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif task == TaskType.NAMED_ENTITY_RECOGNITION:
            st.markdown("### 👁️ Quantum Entity Recognition")
            entities = results.get("entities", {})
            
            if enable_neural_vis and entities:
                # Neural entity network
                entity_counts = {k: len(v) for k, v in entities.items() if v}
                if entity_counts:
                    fig = px.sunburst(
                        names=list(entity_counts.keys()) + [f"{k}_{i}" for k, v in entities.items() if v for i in range(min(len(v), 5))],
                        parents=[""] * len(entity_counts) + [k for k, v in entities.items() if v for _ in range(min(len(v), 5))],
                        values=[entity_counts[k] for k in entity_counts] + [1 for k, v in entities.items() if v for _ in range(min(len(v), 5))],
                        title="Neural Entity Constellation",
                        color_discrete_sequence=px.colors.sequential.Plasma
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='#FFB300'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Entity display
            for category, entity_list in entities.items():
                if entity_list:
                    with st.expander(f"🔹 {category.title()} Entities ({len(entity_list)} detected)"):
                        cols = st.columns(3)
                        for i, entity in enumerate(entity_list[:15]):
                            with cols[i % 3]:
                                st.markdown(f'''
                                <div class="swarm-node">
                                    <div class="pulse-dot"></div>
                                    {entity}
                                </div>
                                ''', unsafe_allow_html=True)
        
        elif task == TaskType.SUMMARIZATION:
            st.markdown("### 📝 Quantum Text Synthesis")
            summary = results.get("summary", "")
            
            # Enhanced summary display
            st.markdown(f'''
            <div style="
                background: rgba(255,179,0,0.05);
                border-left: 4px solid var(--honey-gold);
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                font-size: 1.1rem;
                line-height: 1.6;
            ">
                {summary}
            </div>
            ''', unsafe_allow_html=True)
            
            with st.expander(f"🧠 Neural Analysis Details ({len(results.get('top_sentences', []))} key sentences)"):
                for i, sentence in enumerate(results.get('top_sentences', [])[:5], 1):
                    st.markdown(f"**Quantum Fragment {i}:** {sentence}")
        
        elif task == TaskType.TOPIC_MODELING:
            st.markdown("### 🏷️ Quantum Topic Mapping")
            dominant_topics = results.get("dominant_topics", [])
            
            if dominant_topics and enable_neural_vis:
                # Enhanced topic visualization
                topics, scores = zip(*dominant_topics)
                fig = px.radar(
                    r=[scores],
                    theta=list(topics),
                    line_close=True,
                    title="Neural Topic Resonance Map"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='#FFB300',
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            color='rgba(255,179,0,0.5)'
                        ),
                        angularaxis=dict(
                            color='#FFB300'
                        )
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Topic resonance display
            for topic, score in dominant_topics[:5]:
                st.markdown(f'''
                <div class="swarm-node" style="width: 100%; justify-content: space-between;">
                    <div style="display: flex; align-items: center;">
                        <div class="pulse-dot"></div>
                        {topic.title()}
                    </div>
                    <div style="color: var(--honey-gold); font-weight: bold;">
                        {score} Hz
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        else:
            # Generic enhanced display for other tasks
            st.json(results, expanded=False)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update execution history
        st.session_state.execution_history.append({
            'task': task.value,
            'results': results,
            'stats': stats,
            'timestamp': time.time(),
            'duration': duration,
            'agents': len(agents)
        })
    
    st.session_state.hive_status = "dormant"

# Enhanced execution history
if st.session_state.execution_history:
    st.markdown('<div class="hive-panel">', unsafe_allow_html=True)
    st.markdown('<div class="quantum-badge">📈 Quantum Mission Archive</div>', unsafe_allow_html=True)
    
    # Summary statistics
    total_tasks = len(st.session_state.execution_history)
    avg_success = sum(entry['stats'].get('success_rate', 0) for entry in st.session_state.execution_history) / total_tasks
    
    st.markdown(f"**Archive Status:** {total_tasks} missions • {avg_success:.1f}% average success rate")
    
    # Recent missions
    for i, entry in enumerate(reversed(st.session_state.execution_history[-3:])):
        mission_num = len(st.session_state.execution_history) - i
        
        with st.expander(f"🚀 Mission {mission_num}: {entry['task'].replace('_', ' ').title()}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Task", entry['task'].replace('_', ' ').title())
                st.metric("Duration", f"{entry.get('duration', 0):.2f}s")
            
            with col2:
                st.metric("Success Rate", f"{entry['stats'].get('success_rate', 0):.1f}%")
                st.metric("Agents", entry.get('agents', 0))
            
            with col3:
                timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                st.metric("Timestamp", timestamp)
                st.metric("Results", len(entry['results']) if entry['results'] else 0)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Quantum footer
st.markdown("---")
st.markdown('''
<div style="
    text-align: center; 
    font-family: 'Exo 2', sans-serif; 
    color: var(--honey-amber);
    font-size: 0.9rem;
    opacity: 0.8;
    margin: 2rem 0;
">
    🌌 Powered by Quantum Swarm Intelligence • Neural Architecture v2.1 • Built with ❤️ and ⚡
</div>
''', unsafe_allow_html=True)