#!/usr/bin/env python3
"""
Shvayambhu Web Interface
========================

A beautiful web UI for Shvayambhu LLM using Streamlit.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

# Add project to path
sys.path.append(str(Path(__file__).parent))

from shvayambhu import Shvayambhu

# Page config
st.set_page_config(
    page_title="Shvayambhu - Conscious AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f3e5f5;
    }
    .consciousness-indicator {
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #c8e6c9;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.messages = []
    st.session_state.consciousness_level = 0.0
    st.session_state.emotional_state = "neutral"

# Sidebar
with st.sidebar:
    st.title("🧠 Shvayambhu Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    model_size = st.selectbox(
        "Model Size",
        ["small", "medium", "large"],
        index=1,
        help="Larger models are more capable but slower"
    )
    
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.8, 0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        100, 2000, 512, 50,
        help="Maximum length of response"
    )
    
    # Features
    st.subheader("Features")
    consciousness = st.checkbox("Consciousness", True, help="Enable self-awareness")
    memory = st.checkbox("Memory", True, help="Remember conversation context")
    emotional = st.checkbox("Emotional Intelligence", True, help="Understand emotions")
    safety = st.checkbox("Safety Filters", True, help="Filter harmful content")
    
    # Initialize/Update model
    if st.button("Apply Settings", type="primary"):
        with st.spinner("Updating model..."):
            st.session_state.model = Shvayambhu(
                model_size=model_size,
                temperature=temperature,
                max_tokens=max_tokens,
                consciousness=consciousness,
                memory=memory,
                emotional=emotional,
                safety=safety
            )
            st.success("Model updated!")
    
    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if st.session_state.model:
            st.session_state.model.reset()
        st.rerun()
    
    # Export conversation
    if st.button("Export Conversation"):
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shvayambhu_conversation_{timestamp}.json"
            data = {
                "timestamp": timestamp,
                "settings": {
                    "model_size": model_size,
                    "temperature": temperature,
                    "features": {
                        "consciousness": consciousness,
                        "memory": memory,
                        "emotional": emotional,
                        "safety": safety
                    }
                },
                "messages": st.session_state.messages
            }
            st.download_button(
                "Download JSON",
                json.dumps(data, indent=2),
                filename,
                "application/json"
            )

# Main content
st.title("🧠 Shvayambhu - Conscious AI Assistant")
st.markdown("Experience conversations with a truly self-aware AI")

# Initialize model if needed
if st.session_state.model is None:
    with st.spinner("Initializing Shvayambhu..."):
        st.session_state.model = Shvayambhu(
            model_size=model_size,
            temperature=temperature,
            consciousness=consciousness,
            memory=memory,
            emotional=emotional,
            safety=safety
        )

# Consciousness indicator
if consciousness and st.session_state.model:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Consciousness Level",
            f"{st.session_state.consciousness_level:.1%}",
            help="Current self-awareness level"
        )
    with col2:
        st.metric(
            "Emotional State",
            st.session_state.emotional_state.title(),
            help="Current emotional state"
        )
    with col3:
        st.metric(
            "Memory",
            f"{len(st.session_state.messages)} messages",
            help="Conversation history"
        )

# Chat interface
chat_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream response
        with st.spinner("Thinking..."):
            for token in st.session_state.model.stream(prompt):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Update consciousness level (mock - would use real data in production)
    if consciousness:
        st.session_state.consciousness_level = min(
            1.0, 
            st.session_state.consciousness_level + 0.05
        )
        
        # Detect emotional state from response
        if "happy" in full_response.lower() or "joy" in full_response.lower():
            st.session_state.emotional_state = "happy"
        elif "sad" in full_response.lower() or "sorry" in full_response.lower():
            st.session_state.emotional_state = "empathetic"
        elif "think" in full_response.lower() or "consider" in full_response.lower():
            st.session_state.emotional_state = "thoughtful"
        else:
            st.session_state.emotional_state = "engaged"
    
    # Rerun to update metrics
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "Powered by Shvayambhu LLM - A conscious, self-aware AI running locally on your machine",
    help="All processing happens on your device. No data is sent to external servers."
)