import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dtw import dtw
import random

class GestureAnalyzer:
    """
    Class to analyze and compare gesture patterns.
    """
    def __init__(self):
        self.feature_weights = {
            'landmarks': 0.5,
            'velocities': 0.1,
            'relative_positions': 0.2,
            'angles': 0.1,
            'trajectory': 0.1
        }
    
    def compare_gesture_features(self, features1, features2):
        """
        Compare two sets of gesture features and return a similarity score.
        
        Args:
            features1: First set of gesture features
            features2: Second set of gesture features
            
        Returns:
            float: Similarity score between 0 (different) and 1 (identical)
        """
        # Calculate similarity scores for each feature type
        similarity_scores = {}
        
        # Compare landmarks using cosine similarity
        if 'landmarks' in features1 and 'landmarks' in features2 and features1['landmarks'] and features2['landmarks']:
            landmarks1 = np.array(features1['landmarks'])
            landmarks2 = np.array(features2['landmarks'])
            
            # Ensure same length by padding shorter array if needed
            if len(landmarks1) < len(landmarks2):
                landmarks1 = np.pad(landmarks1, (0, len(landmarks2) - len(landmarks1)), 'constant')
            elif len(landmarks2) < len(landmarks1):
                landmarks2 = np.pad(landmarks2, (0, len(landmarks1) - len(landmarks2)), 'constant')
            
            landmarks1 = landmarks1.reshape(1, -1)
            landmarks2 = landmarks2.reshape(1, -1)
            
            landmark_similarity = float(cosine_similarity(landmarks1, landmarks2)[0][0])
            similarity_scores['landmarks'] = max(0, landmark_similarity)  # Ensure non-negative
        else:
            similarity_scores['landmarks'] = 0.0
        
        # Compare velocities using cosine similarity
        if 'velocities' in features1 and 'velocities' in features2 and features1['velocities'] and features2['velocities']:
            velocities1 = np.array(features1['velocities'])
            velocities2 = np.array(features2['velocities'])
            
            # Ensure same length
            if len(velocities1) < len(velocities2):
                velocities1 = np.pad(velocities1, (0, len(velocities2) - len(velocities1)), 'constant')
            elif len(velocities2) < len(velocities1):
                velocities2 = np.pad(velocities2, (0, len(velocities1) - len(velocities2)), 'constant')
            
            velocities1 = velocities1.reshape(1, -1)
            velocities2 = velocities2.reshape(1, -1)
            
            velocity_similarity = float(cosine_similarity(velocities1, velocities2)[0][0])
            similarity_scores['velocities'] = max(0, velocity_similarity)
        else:
            similarity_scores['velocities'] = 0.0
        
        # Compare relative positions
        if ('relative_positions' in features1 and 'relative_positions' in features2 and 
            features1['relative_positions'] and features2['relative_positions']):
            rel_pos1 = np.array(features1['relative_positions'])
            rel_pos2 = np.array(features2['relative_positions'])
            
            # Ensure same length
            if len(rel_pos1) < len(rel_pos2):
                rel_pos1 = np.pad(rel_pos1, (0, len(rel_pos2) - len(rel_pos1)), 'constant')
            elif len(rel_pos2) < len(rel_pos1):
                rel_pos2 = np.pad(rel_pos2, (0, len(rel_pos1) - len(rel_pos2)), 'constant')
            
            rel_pos1 = rel_pos1.reshape(1, -1)
            rel_pos2 = rel_pos2.reshape(1, -1)
            
            rel_pos_similarity = float(cosine_similarity(rel_pos1, rel_pos2)[0][0])
            similarity_scores['relative_positions'] = max(0, rel_pos_similarity)
        else:
            similarity_scores['relative_positions'] = 0.0
        
        # Compare angles
        if 'angles' in features1 and 'angles' in features2 and features1['angles'] and features2['angles']:
            angles1 = np.array(features1['angles'])
            angles2 = np.array(features2['angles'])
            
            # Ensure same length
            if len(angles1) < len(angles2):
                angles1 = np.pad(angles1, (0, len(angles2) - len(angles1)), 'constant')
            elif len(angles2) < len(angles1):
                angles2 = np.pad(angles2, (0, len(angles1) - len(angles2)), 'constant')
            
            # Calculate angle similarity as 1 - normalized_difference
            angle_diff = np.mean(np.abs(angles1 - angles2) / 180.0)
            angle_similarity = 1.0 - angle_diff
            similarity_scores['angles'] = max(0, angle_similarity)
        else:
            similarity_scores['angles'] = 0.0
        
        # Compare trajectories using Dynamic Time Warping
        if 'trajectory' in features1 and 'trajectory' in features2 and features1['trajectory'] and features2['trajectory']:
            try:
                traj1 = np.array(features1['trajectory'])
                traj2 = np.array(features2['trajectory'])
                
                # Ensure trajectories have at least 2 points
                if len(traj1) >= 2 and len(traj2) >= 2:
                    # Calculate DTW distance
                    alignment = dtw(traj1, traj2, dist=lambda x, y: np.linalg.norm(x - y))
                    # Convert distance to similarity score (higher is better)
                    max_distance = max(len(traj1), len(traj2)) * np.sqrt(2 * (500**2))  # Max possible distance in a 500x500 space
                    trajectory_similarity = 1.0 - min(1.0, alignment.distance / max_distance)
                else:
                    trajectory_similarity = 0.5  # Default value if not enough points
                
                similarity_scores['trajectory'] = max(0, trajectory_similarity)
            except Exception as e:
                print(f"Error comparing trajectories: {str(e)}")
                similarity_scores['trajectory'] = 0.0
        else:
            similarity_scores['trajectory'] = 0.0
        
        # Calculate weighted average similarity
        weighted_similarity = 0.0
        total_weight = 0.0
        
        for feature_type, similarity in similarity_scores.items():
            if feature_type in self.feature_weights:
                weighted_similarity += similarity * self.feature_weights[feature_type]
                total_weight += self.feature_weights[feature_type]
        
        # Normalize
        if total_weight > 0:
            weighted_similarity /= total_weight
        
        return weighted_similarity
    
    def compare_gesture_sequences(self, sequence1, sequence2):
        """
        Compare two sequences of gesture features.
        
        Args:
            sequence1: First sequence of gesture features
            sequence2: Second sequence of gesture features
            
        Returns:
            float: Similarity score between 0 (different) and 1 (identical)
        """
        if not sequence1 or not sequence2:
            return 0.0
        
        # Calculate pairwise similarities between frames
        similarities = []
        
        # Use a sliding window approach to find the best alignment
        # This is a simplified version - a full implementation would use sequence alignment algorithms
        min_len = min(len(sequence1), len(sequence2))
        
        # Sample frames at regular intervals if sequences are long
        if min_len > 10:
            stride = min_len // 10
            samples1 = [sequence1[i] for i in range(0, len(sequence1), stride)]
            samples2 = [sequence2[i] for i in range(0, len(sequence2), stride)]
        else:
            samples1 = sequence1
            samples2 = sequence2
        
        # Compare corresponding frames
        for i in range(min(len(samples1), len(samples2))):
            similarity = self.compare_gesture_features(samples1[i], samples2[i])
            similarities.append(similarity)
        
        # Average similarity across frames
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    
    def verify_gesture(self, test_sequence, reference_sequence, threshold=0.7):
        """
        Verify if a test gesture matches a reference gesture.
        
        Args:
            test_sequence: Sequence of gesture features to verify
            reference_sequence: Reference sequence to compare against
            threshold: Minimum similarity threshold for verification
            
        Returns:
            dict: Verification result with match status, confidence and adaptation score
        """
        # Compare the sequences
        similarity = self.compare_gesture_sequences(test_sequence, reference_sequence)
        
        # Convert similarity to confidence percentage
        confidence = similarity * 100.0
        
        # Calculate adaptation score (how much the system learned from this verification)
        # Higher for borderline matches, lower for very high or very low similarity
        adaptation_distance = abs(similarity - threshold)
        adaptation_score = 1.0 - min(1.0, adaptation_distance * 2)
        
        # Determine if it's a match
        match = similarity >= threshold
        
        return {
            'match': match,
            'confidence': confidence,
            'similarity': similarity,
            'adaptation_score': adaptation_score
        }
    
    def analyze_user_consistency(self, gesture_history):
        """
        Analyze the consistency of a user's gestures over time.
        
        Args:
            gesture_history: List of gesture sequences for the same gesture type
            
        Returns:
            dict: Analysis results including consistency score and variations
        """
        if not gesture_history or len(gesture_history) < 2:
            return {
                'consistency': 0.0,
                'variations': [],
                'improvement_suggestions': []
            }
        
        # Calculate similarities between consecutive performances
        consistency_scores = []
        for i in range(1, len(gesture_history)):
            similarity = self.compare_gesture_sequences(gesture_history[i], gesture_history[i-1])
            consistency_scores.append(similarity)
        
        # Average consistency
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        # Find variations (points of low consistency)
        variations = []
        for i, score in enumerate(consistency_scores):
            if score < avg_consistency * 0.8:  # 20% below average is considered a variation
                variations.append(i+1)  # Index in the gesture history
        
        # Generate improvement suggestions based on consistency
        suggestions = []
        if avg_consistency < 0.6:
            suggestions.append("Try to perform the gesture more consistently")
        if avg_consistency < 0.4:
            suggestions.append("Practice the gesture several times to build muscle memory")
        
        return {
            'consistency': avg_consistency,
            'variations': variations,
            'improvement_suggestions': suggestions
        }
    
    def optimize_feature_weights(self, training_data):
        """
        Optimize feature weights based on training data.
        
        Args:
            training_data: List of (user_id, gesture_type, gesture_sequence) tuples
            
        Returns:
            dict: Optimized feature weights
        """
        # In a real implementation, this would use a more sophisticated approach
        # to optimize weights based on which features are most discriminative
        
        # For now, we'll just return a predefined set of weights
        return {
            'landmarks': 0.5,
            'velocities': 0.1,
            'relative_positions': 0.2,
            'angles': 0.1,
            'trajectory': 0.1
        }