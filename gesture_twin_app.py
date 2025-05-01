import streamlit as st
import cv2
import numpy as np
import os
import time
from utils.gesture_tracker import GestureTracker
from utils.gesture_analyzer import GestureAnalyzer
from utils.user_profile import UserProfile, load_user_profiles, save_user_profile

# Page configuration
st.set_page_config(
    page_title="Gestural Digital Twin System",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'registration_mode' not in st.session_state:
    st.session_state.registration_mode = False
if 'verification_mode' not in st.session_state:
    st.session_state.verification_mode = False
if 'tracking_enabled' not in st.session_state:
    st.session_state.tracking_enabled = False
if 'collected_gestures' not in st.session_state:
    st.session_state.collected_gestures = []
if 'gesture_analyzer' not in st.session_state:
    st.session_state.gesture_analyzer = GestureAnalyzer()
if 'users_loaded' not in st.session_state:
    # Create profiles directory if it doesn't exist
    os.makedirs('profiles', exist_ok=True)
    st.session_state.users = load_user_profiles()
    st.session_state.users_loaded = True

# Initialize the gesture tracker
tracker = GestureTracker()

# Main title
st.title("Gestural Digital Twin System")
st.markdown("""
This application creates a digital twin of your unique gesture patterns and uses them to recognize your identity.
It analyzes your personal gesture "signature" and adapts over time to better understand your natural movements.
""")

# Sidebar for user management
st.sidebar.title("User Management")

# User selection or creation
user_names = list(st.session_state.users.keys())
user_names.append("Create New User")

selected_user = st.sidebar.selectbox(
    "Select User Profile", 
    options=user_names,
    index=0 if user_names else 0
)

if selected_user == "Create New User":
    new_user_name = st.sidebar.text_input("Enter new user name")
    if st.sidebar.button("Create User") and new_user_name:
        if new_user_name not in st.session_state.users:
            st.session_state.users[new_user_name] = UserProfile(new_user_name)
            save_user_profile(st.session_state.users[new_user_name])
            st.session_state.current_user = new_user_name
            st.sidebar.success(f"User {new_user_name} created!")
            st.experimental_rerun()
        else:
            st.sidebar.error("User already exists!")
else:
    st.session_state.current_user = selected_user

# Mode selection
st.sidebar.title("Operation Mode")
mode = st.sidebar.radio(
    "Select Mode",
    options=["Dashboard", "Register Gestures", "Verify Identity"]
)

# Update session state based on mode
st.session_state.registration_mode = (mode == "Register Gestures")
st.session_state.verification_mode = (mode == "Verify Identity")

# Main content based on selected mode
if mode == "Dashboard" and st.session_state.current_user:
    st.header(f"Dashboard for {st.session_state.current_user}")
    
    # User statistics and visualizations
    user = st.session_state.users[st.session_state.current_user]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Profile")
        st.write(f"Username: {user.name}")
        st.write(f"Created: {user.created_date}")
        st.write(f"Last Updated: {user.last_updated}")
        st.write(f"Number of Registered Gestures: {len(user.gesture_patterns)}")
        st.write(f"Verification Attempts: {user.verification_attempts}")
        st.write(f"Successful Verifications: {user.successful_verifications}")
        
        if user.verification_attempts > 0:
            success_rate = (user.successful_verifications / user.verification_attempts) * 100
            st.write(f"Verification Success Rate: {success_rate:.2f}%")
    
    with col2:
        st.subheader("Gesture Confidence Levels")
        if user.gesture_patterns:
            # Create a bar chart for gesture confidence levels
            gesture_names = list(user.gesture_patterns.keys())
            confidence_values = [user.gesture_confidence.get(name, 50) for name in gesture_names]
            
            chart_data = {
                'Gesture': gesture_names,
                'Confidence': confidence_values
            }
            
            st.bar_chart(chart_data, x='Gesture', y='Confidence', height=300)
        else:
            st.info("No gestures registered yet.")
    
    # Gesture adaptation over time
    st.subheader("Gesture Adaptation Over Time")
    if user.adaptation_history:
        adaptation_data = {
            'Date': list(user.adaptation_history.keys()),
            'Adaptation Score': list(user.adaptation_history.values())
        }
        st.line_chart(adaptation_data, x='Date', y='Adaptation Score', height=300)
    else:
        st.info("No adaptation history available yet.")
    
    # Actions
    st.subheader("Profile Actions")
    if st.button("Reset User Profile"):
        st.session_state.users[st.session_state.current_user] = UserProfile(st.session_state.current_user)
        save_user_profile(st.session_state.users[st.session_state.current_user])
        st.success("User profile has been reset.")
        st.experimental_rerun()

elif mode == "Register Gestures" and st.session_state.current_user:
    st.header(f"Register Gestures for {st.session_state.current_user}")
    
    available_gestures = [
        "Wave", "Pinch", "Open Hand", "Point", "Fist",
        "Thumbs Up", "Victory", "Swipe Left", "Swipe Right"
    ]
    
    # Which gestures are already registered
    user = st.session_state.users[st.session_state.current_user]
    registered_gestures = list(user.gesture_patterns.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Register New Gesture")
        
        # Filter out already registered gestures
        unregistered_gestures = [g for g in available_gestures if g not in registered_gestures]
        
        if unregistered_gestures:
            gesture_to_register = st.selectbox(
                "Select a gesture to register",
                options=unregistered_gestures
            )
            
            st.write("When ready, click 'Start Recording' and perform the gesture.")
            
            if st.button("Start Recording"):
                st.session_state.tracking_enabled = True
                st.session_state.collected_gestures = []
                st.info(f"Perform the '{gesture_to_register}' gesture. Recording for 3 seconds...")
                
                # This is placeholder - actual implementation would use OpenCV stream
                placeholder = st.empty()
                with placeholder.container():
                    for i in range(3, 0, -1):
                        st.write(f"Recording in {i}...")
                        time.sleep(1)
                    
                    # This would be replaced by actual gesture collection from webcam
                    # For demo purposes, we'll just simulate collecting gesture data
                    st.session_state.collected_gestures = tracker.simulate_gesture_data(gesture_to_register)
                    
                    if st.session_state.collected_gestures:
                        # Register the gesture in the user profile
                        user.add_gesture_pattern(gesture_to_register, st.session_state.collected_gestures)
                        save_user_profile(user)
                        st.success(f"Successfully registered '{gesture_to_register}' gesture!")
                    else:
                        st.error("Failed to collect gesture data. Please try again.")
                    
                st.session_state.tracking_enabled = False
        else:
            st.info("All available gestures have been registered.")
    
    with col2:
        st.subheader("Registered Gestures")
        if registered_gestures:
            for gesture in registered_gestures:
                st.write(f"- {gesture} (Confidence: {user.gesture_confidence.get(gesture, 50)}%)")
                
            if st.button("Re-record All Gestures"):
                user.gesture_patterns = {}
                user.gesture_confidence = {}
                save_user_profile(user)
                st.success("All gestures have been cleared. You can re-record them now.")
                st.experimental_rerun()
        else:
            st.info("No gestures registered yet.")

elif mode == "Verify Identity" and st.session_state.current_user:
    st.header(f"Verify Identity: {st.session_state.current_user}")
    
    user = st.session_state.users[st.session_state.current_user]
    registered_gestures = list(user.gesture_patterns.keys())
    
    if not registered_gestures:
        st.warning("No gestures are registered for this user. Please register gestures first.")
    else:
        st.write("To verify your identity, perform one of your registered gestures.")
        
        selected_gesture = st.selectbox(
            "Select gesture to perform",
            options=registered_gestures
        )
        
        st.write(f"When ready, click 'Verify' and perform the '{selected_gesture}' gesture.")
        
        if st.button("Verify"):
            st.session_state.tracking_enabled = True
            
            # This is placeholder - actual implementation would use OpenCV stream
            placeholder = st.empty()
            with placeholder.container():
                st.info(f"Perform the '{selected_gesture}' gesture. Verifying for 3 seconds...")
                
                for i in range(3, 0, -1):
                    st.write(f"Verifying in {i}...")
                    time.sleep(1)
                
                # This would be replaced by actual gesture verification
                # For demo purposes, we'll just simulate verification (80% chance of success)
                verification_result = st.session_state.gesture_analyzer.verify_gesture(
                    tracker.simulate_gesture_data(selected_gesture),
                    user.gesture_patterns.get(selected_gesture, []),
                    threshold=0.7
                )
                
                user.verification_attempts += 1
                
                if verification_result['match']:
                    user.successful_verifications += 1
                    st.success(f"Identity verified! Confidence: {verification_result['confidence']:.2f}%")
                    
                    # Update user's gesture pattern with this successful verification
                    new_pattern = tracker.simulate_gesture_data(selected_gesture)
                    user.update_gesture_pattern(selected_gesture, new_pattern)
                    user.update_gesture_confidence(selected_gesture, verification_result['confidence'])
                    
                    # Record adaptation
                    user.record_adaptation(verification_result['adaptation_score'])
                else:
                    st.error(f"Verification failed. Confidence: {verification_result['confidence']:.2f}%")
                
                save_user_profile(user)
            
            st.session_state.tracking_enabled = False
else:
    st.info("Please select or create a user to continue.")

# Information section
st.sidebar.markdown("---")
st.sidebar.info("""
This application demonstrates a Gestural Digital Twin system that:
1. Creates a unique profile based on your personal gesture patterns
2. Learns and adapts to your gestures over time
3. Can verify your identity based on how you perform gestures
4. Provides analytics on gesture confidence and system adaptation
""")