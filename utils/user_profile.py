import os
import json
import datetime
import pickle
import numpy as np
from pathlib import Path

class UserProfile:
    """
    Class to store and manage user profiles with their gesture patterns.
    """
    def __init__(self, name):
        self.name = name
        self.created_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated = self.created_date
        self.gesture_patterns = {}  # {gesture_type: [gesture_data]}
        self.gesture_confidence = {}  # {gesture_type: confidence_level}
        self.verification_attempts = 0
        self.successful_verifications = 0
        self.adaptation_history = {}  # {date: adaptation_score}
    
    def add_gesture_pattern(self, gesture_type, gesture_data):
        """
        Add a new gesture pattern for a user.
        
        Args:
            gesture_type: Type of gesture (e.g., "Wave", "Pinch")
            gesture_data: Data for the gesture pattern
        """
        self.gesture_patterns[gesture_type] = gesture_data
        self.gesture_confidence[gesture_type] = 50.0  # Default confidence of 50%
        self.update_timestamp()
    
    def update_gesture_pattern(self, gesture_type, new_data, learning_rate=0.2):
        """
        Update an existing gesture pattern with new data.
        
        Args:
            gesture_type: Type of gesture to update
            new_data: New gesture data
            learning_rate: Rate at which to update (0-1)
        """
        if gesture_type not in self.gesture_patterns:
            self.add_gesture_pattern(gesture_type, new_data)
            return
        
        # In a real implementation, this would be a more sophisticated update
        # that integrates the new pattern into the existing one
        # For simplicity, we'll just replace a random frame with the new data
        
        # Get current patterns
        current_patterns = self.gesture_patterns[gesture_type]
        
        # For demo purposes, we'll just replace the pattern completely
        # with a learning rate factor. In a real system, you'd combine them.
        self.gesture_patterns[gesture_type] = new_data
        
        self.update_timestamp()
    
    def update_gesture_confidence(self, gesture_type, confidence):
        """
        Update the confidence level for a gesture.
        
        Args:
            gesture_type: Type of gesture
            confidence: New confidence level (0-100)
        """
        if gesture_type in self.gesture_confidence:
            # Gradually adjust confidence
            old_confidence = self.gesture_confidence[gesture_type]
            self.gesture_confidence[gesture_type] = old_confidence * 0.7 + confidence * 0.3
        else:
            self.gesture_confidence[gesture_type] = confidence
        
        self.update_timestamp()
    
    def record_adaptation(self, adaptation_score):
        """
        Record adaptation score in history.
        
        Args:
            adaptation_score: Score indicating how much the system adapted (0-1)
        """
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.adaptation_history[date] = adaptation_score
        self.update_timestamp()
    
    def update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self):
        """
        Convert user profile to dictionary for serialization.
        
        Returns:
            dict: User profile as dictionary
        """
        # Convert numpy arrays or complex objects to lists for JSON serialization
        return {
            'name': self.name,
            'created_date': self.created_date,
            'last_updated': self.last_updated,
            'gesture_patterns': self.gesture_patterns,
            'gesture_confidence': self.gesture_confidence,
            'verification_attempts': self.verification_attempts,
            'successful_verifications': self.successful_verifications,
            'adaptation_history': self.adaptation_history
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create user profile from dictionary.
        
        Args:
            data: Dictionary representation of user profile
            
        Returns:
            UserProfile: User profile object
        """
        profile = cls(data['name'])
        profile.created_date = data['created_date']
        profile.last_updated = data['last_updated']
        profile.gesture_patterns = data['gesture_patterns']
        profile.gesture_confidence = data['gesture_confidence']
        profile.verification_attempts = data['verification_attempts']
        profile.successful_verifications = data['successful_verifications']
        profile.adaptation_history = data['adaptation_history']
        return profile


def save_user_profile(user_profile):
    """
    Save user profile to file.
    
    Args:
        user_profile: UserProfile object to save
    """
    # Create profiles directory if it doesn't exist
    os.makedirs('profiles', exist_ok=True)
    
    # Save profile to json file
    profile_path = os.path.join('profiles', f"{user_profile.name}.json")
    
    with open(profile_path, 'w') as f:
        json.dump(user_profile.to_dict(), f, indent=2)


def load_user_profile(username):
    """
    Load user profile from file.
    
    Args:
        username: Name of user to load
        
    Returns:
        UserProfile: Loaded user profile or None if not found
    """
    profile_path = os.path.join('profiles', f"{username}.json")
    
    if not os.path.exists(profile_path):
        return None
    
    try:
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        return UserProfile.from_dict(data)
    except Exception as e:
        print(f"Error loading user profile: {str(e)}")
        return None


def load_user_profiles():
    """
    Load all user profiles from profiles directory.
    
    Returns:
        dict: Dictionary of {username: UserProfile}
    """
    profiles = {}
    
    if not os.path.exists('profiles'):
        return profiles
    
    for profile_file in os.listdir('profiles'):
        if profile_file.endswith('.json'):
            username = os.path.splitext(profile_file)[0]
            profile = load_user_profile(username)
            if profile:
                profiles[username] = profile
    
    return profiles


def delete_user_profile(username):
    """
    Delete a user profile.
    
    Args:
        username: Name of user to delete
        
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    profile_path = os.path.join('profiles', f"{username}.json")
    
    if os.path.exists(profile_path):
        try:
            os.remove(profile_path)
            return True
        except:
            return False
    
    return False