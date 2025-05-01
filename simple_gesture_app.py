# simple_gesture_app.py
import streamlit as st
import numpy as np
import os
import time
import random
import cv2

# Import the local modules
from utils.gesture_tracker import GestureTracker
from utils.gesture_analyzer import GestureAnalyzer
from utils.user_profile import UserProfile, load_user_profiles, save_user_profile

st.set_page_config(
    page_title="Simple Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# Initialize session state variables
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
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

# Main title
st.title("Simple Gesture Recognition System")
st.markdown("""
This application demonstrates basic hand gesture recognition using OpenCV.
It detects skin color to identify hand contours and track movements.
""")

# Sidebar - User Management
st.sidebar.title("User Management")
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
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        else:
            st.sidebar.error("User already exists!")
else:
    st.session_state.current_user = selected_user

# Mode selection
st.sidebar.title("Mode")
mode = st.sidebar.radio(
    "Select Mode",
    options=["Demonstration", "Register Gesture", "Test Gesture"]
)

# Main content
if st.session_state.current_user and st.session_state.current_user != "Create New User":
    user = st.session_state.users[st.session_state.current_user]
    
    if mode == "Demonstration":
        st.header("Hand Gesture Demonstration")
        st.write("This mode shows how the hand detection works with sample data.")
        
        # Display simulated hand detection
        tracker = GestureTracker()
        
        # Create a placeholder for showing images
        image_placeholder = st.empty()
        
        # Show hand tracking with simulated data
        if st.button("Run Demonstration"):
            for i in range(30):
                # Generate a blank image
                img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                
                # Draw a simulated hand position
                cx = 300 + int(100 * np.sin(i * 0.2))
                cy = 200 + int(50 * np.cos(i * 0.2))
                
                # Draw contour
                cv2.circle(img, (cx, cy), 50, (0, 255, 0), 2)
                
                # Draw trajectory
                for j in range(max(0, i-10), i):
                    x1 = 300 + int(100 * np.sin(j * 0.2))
                    y1 = 200 + int(50 * np.cos(j * 0.2))
                    x2 = 300 + int(100 * np.sin((j+1) * 0.2))
                    y2 = 200 + int(50 * np.cos((j+1) * 0.2))
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Display the image
                image_placeholder.image(img, channels="BGR", caption=f"Frame {i+1}/30")
                
                # Short delay to simulate real-time processing
                time.sleep(0.1)
        
        st.write("""
        ### How It Works
        
        1. The system detects skin color in the webcam image
        2. It identifies the hand contour and tracks its movement
        3. Feature points are extracted from the hand shape
        4. These features are used to recognize gesture patterns
        """)
        
    elif mode == "Register Gesture":
        st.header("Register New Gesture")
        
        available_gestures = [
            "Wave", "Pinch", "Open Hand", "Point", "Fist",
            "Thumbs Up", "Victory", "Swipe Left", "Swipe Right"
        ]
        
        # Which gestures are already registered
        registered_gestures = list(user.gesture_patterns.keys())
        
        # Filter out already registered gestures
        unregistered_gestures = [g for g in available_gestures if g not in registered_gestures]
        
        if unregistered_gestures:
            gesture_to_register = st.selectbox(
                "Select a gesture to register",
                options=unregistered_gestures
            )
            
            st.write("For demonstration purposes, this will use simulated data.")
            
            if st.button("Register Gesture"):
                # Create a tracker
                tracker = GestureTracker()
                
                # For demo purposes, simulate collecting gesture data
                st.write(f"Collecting data for '{gesture_to_register}' gesture...")
                progress_bar = st.progress(0)
                
                for i in range(10):
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.2)
                
                # Simulate gesture data
                collected_gestures = tracker.simulate_gesture_data(gesture_to_register)
                
                # Register the gesture
                user.add_gesture_pattern(gesture_to_register, collected_gestures)
                save_user_profile(user)
                
                st.success(f"Successfully registered '{gesture_to_register}' gesture!")
        else:
            st.info("All available gestures have been registered.")
        
        # Show registered gestures
        if registered_gestures:
            st.subheader("Registered Gestures")
            for gesture in registered_gestures:
                st.write(f"- {gesture} (Confidence: {user.gesture_confidence.get(gesture, 50)}%)")
                
            if st.button("Clear All Gestures"):
                user.gesture_patterns = {}
                user.gesture_confidence = {}
                save_user_profile(user)
                st.success("All gestures have been cleared.")
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                
    elif mode == "Test Gesture":
        st.header("Test Gesture Recognition")
        
        registered_gestures = list(user.gesture_patterns.keys())
        
        if not registered_gestures:
            st.warning("No gestures are registered for this user. Please register gestures first.")
        else:
            st.write("For demonstration, select a gesture to test:")
            
            selected_gesture = st.selectbox(
                "Select gesture",
                options=registered_gestures
            )
            
            if st.button("Test Recognition"):
                # Create tracker and analyzer
                tracker = GestureTracker()
                analyzer = GestureAnalyzer()
                
                # Simulate performing the selected gesture
                st.write(f"Performing the '{selected_gesture}' gesture...")
                progress_bar = st.progress(0)
                
                for i in range(10):
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.2)
                
                # Simulate recognition with 80% accuracy
                actual_gesture = selected_gesture
                if random.random() < 0.2:  # 20% chance of error
                    other_gestures = [g for g in registered_gestures if g != selected_gesture]
                    if other_gestures:
                        actual_gesture = random.choice(other_gestures)
                
                # Simulate test data
                test_data = tracker.simulate_gesture_data(actual_gesture)
                
                # Verify against all registered gestures
                results = {}
                for gesture in registered_gestures:
                    reference_data = user.gesture_patterns.get(gesture, [])
                    verification = analyzer.verify_gesture(test_data, reference_data)
                    results[gesture] = verification['confidence']
                
                # Find the best match
                best_match = max(results, key=results.get)
                confidence = results[best_match]
                
                # Show results
                st.subheader("Recognition Results")
                
                if best_match == selected_gesture:
                    st.success(f"✓ Correctly identified as '{best_match}' with {confidence:.2f}% confidence")
                else:
                    st.error(f"✗ Misidentified as '{best_match}' with {confidence:.2f}% confidence")
                    st.info(f"You were performing: '{selected_gesture}'")
                
                # Show all confidence scores
                st.write("Confidence scores for all gestures:")
                for gesture, score in results.items():
                    st.write(f"- {gesture}: {score:.2f}%")
                
                # Update user profile with this verification
                user.verification_attempts += 1
                if best_match == selected_gesture:
                    user.successful_verifications += 1
                    user.update_gesture_confidence(selected_gesture, confidence)
                save_user_profile(user)
else:
    st.info("Please select or create a user to continue.")

# Information section
st.sidebar.markdown("---")
st.sidebar.info("""
### How to use this in a real application:

In a real application with webcam access:
1. Connect a webcam to your computer
2. The system would detect your hand using skin color detection
3. Your hand gestures would be tracked in real-time
4. The system would learn and recognize your unique gesture patterns

This demonstration uses simulated data.
""")