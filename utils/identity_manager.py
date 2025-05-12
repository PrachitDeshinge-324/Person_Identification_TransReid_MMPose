"""
Global Identity Manager for Person Identification.
Handles constraint enforcement across multiple tracked individuals:
1. Ensures one person per name (no duplicates)
2. Ensures one name per person (no dual labels)
3. Resolves conflicts when competing tracks have similar identifications
4. Performs global optimization across multiple tracks
"""

import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

class IdentityManager:
    def __init__(self, conflict_threshold=0.15):
        """
        Initialize the global identity manager.
        
        Args:
            conflict_threshold: Maximum allowed confidence difference to consider a conflict.
                               Lower values make the system more likely to detect conflicts.
        """
        # Dictionary mapping identity names to track IDs
        self.name_to_track = {}
        
        # Dictionary mapping track IDs to identity names
        self.track_to_name = {}
        
        # Tracks the confidence scores for each assignment
        self.assignment_confidence = {}
        
        # History of identity assignments for stability
        self.assignment_history = defaultdict(list)
        
        # Top candidate matches for each track (for global optimization)
        self.top_candidates = {}
        
        # Configuration
        self.conflict_threshold = conflict_threshold
        self.history_max_size = 10
        self.history_weight_decay = 0.9  # Weight decay for older assignments
        
        # Confidence boost for established assignments (to promote stability)
        self.established_assignment_boost = 0.05
        
        # Minimum confidence improvement needed to override an existing assignment
        self.override_threshold = 0.1
        
        # Global optimization parameters
        self.top_candidates_per_track = 3
        self.optimization_interval = 60  # frames
        self.frames_since_optimization = 0
    
    def update_track_identity(self, track_id, identity_name, confidence, candidates=None, match_id=None):
        """
        Update the identity for a track, enforcing uniqueness constraints.
        
        Args:
            track_id: ID of the track
            identity_name: Detected identity name
            confidence: Confidence score of the detection
            candidates: Optional list of (name, confidence, match_id) tuples for global optimization
            match_id: Optional database ID of the matched identity
            
        Returns:
            tuple: (assigned_name, assigned_confidence, was_modified)
                - assigned_name: The final identity after constraint enforcement
                - assigned_confidence: The confidence of the assignment
                - was_modified: Whether the original identity was modified
        """
        # If candidates are provided, store them for global optimization
        if candidates:
            self.store_top_candidates(track_id, candidates)
            self.frames_since_optimization += 1
            
            # Periodically perform global optimization
            if self.frames_since_optimization >= self.optimization_interval:
                changes = self.apply_global_optimization()
                self.frames_since_optimization = 0
                
                # If changes were made by global optimization, return the new assignment
                if changes > 0 and track_id in self.track_to_name:
                    name = self.track_to_name[track_id]
                    conf = self.assignment_confidence.get(track_id, confidence)
                    return name, conf, True
        
        # Skip constraint enforcement for "Unknown" and "Pending..." identities
        if identity_name in ["Unknown", "Pending..."]:
            # Just update the history for this track
            self._update_history(track_id, identity_name, confidence)
            return identity_name, confidence, False
        
        # Apply historical confidence stabilization
        historical_confidence = self._get_historical_confidence(track_id, identity_name)
        if historical_confidence > 0:
            # Blend current with historical confidence (80% current, 20% historical)
            confidence = 0.8 * confidence + 0.2 * historical_confidence
        
        # Check if this track already has an assigned identity
        current_name = self.track_to_name.get(track_id)
        if current_name == identity_name:
            # Same identity, just update confidence
            self.assignment_confidence[track_id] = max(confidence, self.assignment_confidence.get(track_id, 0))
            self._update_history(track_id, identity_name, confidence)
            return identity_name, confidence, False
            
        # Check if the identity is already assigned to another track
        existing_track = self.name_to_track.get(identity_name)
        
        # CASE 1: Identity already assigned to another track
        if existing_track is not None and existing_track != track_id:
            existing_confidence = self.assignment_confidence.get(existing_track, 0)
            
            # Compare confidences to resolve the conflict
            if confidence > existing_confidence + self.override_threshold:
                # New assignment has significantly higher confidence, reassign identity
                
                # Remove existing assignment
                old_name = self.track_to_name.get(track_id)
                if old_name and old_name != "Unknown" and old_name != "Pending...":
                    if self.name_to_track.get(old_name) == track_id:
                        del self.name_to_track[old_name]
                
                # Update existing track to Unknown
                self.track_to_name[existing_track] = "Unknown"
                self.assignment_confidence[existing_track] = existing_confidence * 0.5  # Reduce confidence
                
                # Assign new identity to current track
                self.track_to_name[track_id] = identity_name
                self.name_to_track[identity_name] = track_id
                self.assignment_confidence[track_id] = confidence
                self._update_history(track_id, identity_name, confidence)
                
                return identity_name, confidence, False
            else:
                # Not confident enough to reassign, keep current assignment but return Unknown
                
                # Check if the current track already had a different identity
                if track_id in self.track_to_name and self.track_to_name[track_id] != "Unknown" and self.track_to_name[track_id] != "Pending...":
                    # Keep existing assignment
                    name = self.track_to_name[track_id]
                    conf = self.assignment_confidence.get(track_id, 0.5)
                    self._update_history(track_id, name, conf)  # Reinforce existing assignment
                    return name, conf, True
                else:
                    # Return Unknown for this track
                    self._update_history(track_id, "Unknown", confidence * 0.8)  # Reduced confidence for Unknown
                    return "Unknown", confidence * 0.8, True
        
        # CASE 2: Track already has a different identity
        elif track_id in self.track_to_name and self.track_to_name[track_id] != "Unknown" and self.track_to_name[track_id] != "Pending...":
            current_name = self.track_to_name[track_id]
            current_conf = self.assignment_confidence.get(track_id, 0)
            
            # Apply stability boost to the current assignment
            boosted_current_conf = current_conf + self.established_assignment_boost
            
            if confidence > boosted_current_conf + self.override_threshold:
                # New identity has significantly higher confidence
                
                # Remove current name assignment
                if current_name in self.name_to_track and self.name_to_track[current_name] == track_id:
                    del self.name_to_track[current_name]
                
                # Assign new identity
                self.track_to_name[track_id] = identity_name
                self.name_to_track[identity_name] = track_id
                self.assignment_confidence[track_id] = confidence
                self._update_history(track_id, identity_name, confidence)
                
                return identity_name, confidence, True
            else:
                # Keep the current identity due to stability
                self._update_history(track_id, current_name, current_conf)
                return current_name, current_conf, True
        
        # CASE 3: New valid assignment, no conflicts
        else:
            # Remove any previous assignment for this track
            old_name = self.track_to_name.get(track_id)
            if old_name and old_name != "Unknown" and old_name != "Pending..." and self.name_to_track.get(old_name) == track_id:
                del self.name_to_track[old_name]
            
            # Make new assignment
            self.track_to_name[track_id] = identity_name
            self.name_to_track[identity_name] = track_id
            self.assignment_confidence[track_id] = confidence
            self._update_history(track_id, identity_name, confidence)
            
            return identity_name, confidence, False
    
    def _update_history(self, track_id, identity_name, confidence):
        """
        Update history of identity assignments for a track.
        
        Args:
            track_id: ID of the track
            identity_name: Assigned identity name
            confidence: Confidence of the assignment
        """
        # Initialize if needed
        if track_id not in self.assignment_history:
            self.assignment_history[track_id] = []
        
        # Add new assignment to history
        self.assignment_history[track_id].append((identity_name, confidence))
        
        # Trim history if needed
        if len(self.assignment_history[track_id]) > self.history_max_size:
            self.assignment_history[track_id] = self.assignment_history[track_id][-self.history_max_size:]
    
    def _get_historical_confidence(self, track_id, identity_name):
        """
        Calculate confidence boost from historical assignments.
        
        Args:
            track_id: ID of the track
            identity_name: Identity name to check
            
        Returns:
            float: Historical confidence score (0 if no history)
        """
        if track_id not in self.assignment_history:
            return 0
        
        # Get all historical assignments of this identity to this track
        history = self.assignment_history[track_id]
        matching_history = [(i, (name, conf)) for i, (name, conf) in enumerate(history) if name == identity_name]
        
        if not matching_history:
            return 0
        
        # Calculate weighted average of historical confidences
        # More recent assignments get higher weight
        total_weighted_conf = 0
        total_weight = 0
        
        for i, (_, conf) in matching_history:
            # Calculate recency-based weight
            # More recent = higher index = higher weight
            recency_weight = self.history_weight_decay ** (len(history) - 1 - i)
            total_weighted_conf += conf * recency_weight
            total_weight += recency_weight
        
        if total_weight == 0:
            return 0
            
        return total_weighted_conf / total_weight
    
    def get_all_identities(self):
        """
        Get all currently assigned identities.
        
        Returns:
            dict: Mapping of track_id -> (identity_name, confidence)
        """
        all_identities = {}
        for track_id, name in self.track_to_name.items():
            conf = self.assignment_confidence.get(track_id, 0.0)
            all_identities[track_id] = (name, conf)
        return all_identities
    
    def get_identity_conflicts(self):
        """
        Find all identities that are assigned to multiple tracks.
        
        Returns:
            dict: Mapping of identity_name -> list of track_ids
        """
        conflicts = {}
        
        # Group tracks by identity name
        identity_to_tracks = defaultdict(list)
        for track_id, name in self.track_to_name.items():
            if name != "Unknown" and name != "Pending...":
                identity_to_tracks[name].append(track_id)
        
        # Find identities with multiple tracks
        for name, tracks in identity_to_tracks.items():
            if len(tracks) > 1:
                conflicts[name] = tracks
        
        return conflicts
    
    def force_unique_identity(self, identity_name):
        """
        Force uniqueness for an identity by keeping only the highest confidence assignment.
        
        Args:
            identity_name: Name of the identity to enforce uniqueness for
            
        Returns:
            bool: True if changes were made, False otherwise
        """
        # Find all tracks assigned to this identity
        tracks_with_this_identity = []
        
        for track_id, name in self.track_to_name.items():
            if name == identity_name:
                confidence = self.assignment_confidence.get(track_id, 0.0)
                tracks_with_this_identity.append((track_id, confidence))
        
        if len(tracks_with_this_identity) <= 1:
            # Already unique, no changes needed
            return False
        
        # Sort by decreasing confidence
        tracks_with_this_identity.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the highest confidence assignment
        best_track_id = tracks_with_this_identity[0][0]
        
        # Update all other tracks to "Unknown"
        changes_made = False
        for track_id, _ in tracks_with_this_identity[1:]:
            self.track_to_name[track_id] = "Unknown"
            self.assignment_confidence[track_id] = 0.3  # Low confidence for Unknown
            changes_made = True
        
        # Ensure name_to_track mapping is correct
        self.name_to_track[identity_name] = best_track_id
        
        return changes_made
    
    def resolve_low_confidence_conflicts(self, confidence_threshold=0.5):
        """
        Resolve conflicts for low confidence assignments by setting them to "Unknown"
        
        Args:
            confidence_threshold: Threshold below which to resolve conflicts
            
        Returns:
            int: Number of conflicts resolved
        """
        conflicts = self.get_identity_conflicts()
        changes_made = 0
        
        for name, tracks in conflicts.items():
            # For each conflicting identity, check confidences
            low_conf_tracks = []
            for track_id in tracks:
                conf = self.assignment_confidence.get(track_id, 0.0)
                if conf < confidence_threshold:
                    low_conf_tracks.append(track_id)
            
            # If we found low confidence conflicts, resolve them
            for track_id in low_conf_tracks:
                self.track_to_name[track_id] = "Unknown"
                self.assignment_confidence[track_id] = 0.3  # Low confidence for Unknown
                changes_made += 1
            
            # Make sure name_to_track mapping is correct
            # If all were low confidence, this identity has no track now
            if len(low_conf_tracks) == len(tracks) and name in self.name_to_track:
                del self.name_to_track[name]
        
        return changes_made
    
    def clear_inactive_tracks(self, active_track_ids):
        """
        Clean up references to tracks that are no longer active.
        
        Args:
            active_track_ids: Set of track IDs that are currently active
            
        Returns:
            int: Number of tracks cleaned up
        """
        tracks_to_remove = []
        for track_id in self.track_to_name.keys():
            if track_id not in active_track_ids:
                tracks_to_remove.append(track_id)
        
        # Remove inactive tracks
        changes_made = 0
        for track_id in tracks_to_remove:
            name = self.track_to_name[track_id]
            # If this track was the one assigned to its identity, remove the mapping
            if name in self.name_to_track and self.name_to_track[name] == track_id:
                del self.name_to_track[name]
            
            # Remove track-related data
            del self.track_to_name[track_id]
            if track_id in self.assignment_confidence:
                del self.assignment_confidence[track_id]
            if track_id in self.assignment_history:
                del self.assignment_history[track_id]
            if hasattr(self, 'top_candidates') and track_id in self.top_candidates:
                del self.top_candidates[track_id]
            
            changes_made += 1
        
        return changes_made
    
    def store_top_candidates(self, track_id, candidates):
        """
        Store top candidate matches for a track for later global optimization.
        
        Args:
            track_id: ID of the track
            candidates: List of (name, confidence, match_id) tuples sorted by descending confidence
        """
        # Keep only the top N candidates
        self.top_candidates[track_id] = candidates[:self.top_candidates_per_track]
    
    def global_identity_optimization(self):
        """
        Perform global optimization to find the best identity assignment 
        across all tracks that maximizes overall confidence.
        
        Returns:
            dict: Mapping of track_id -> (identity_name, confidence)
        """
        # If we don't have enough tracks with candidates, don't perform optimization
        if len(self.top_candidates) < 2:
            return {}
            
        # Extract all unique identities from candidates
        all_identities = set()
        for track_id, candidates in self.top_candidates.items():
            for name, _, _ in candidates:
                if name != "Unknown" and name != "Pending...":
                    all_identities.add(name)
        
        # If no valid identities, skip optimization
        if not all_identities:
            return {}
        
        # Create a cost matrix for the assignment problem
        # Rows = tracks, Columns = identities
        track_ids = list(self.top_candidates.keys())
        identity_names = list(all_identities)
        
        # Initialize cost matrix with high values (we'll minimize cost)
        cost_matrix = np.full((len(track_ids), len(identity_names)), 1000.0)
        
        # Fill in costs (1 - confidence) for each track-identity pair
        for i, track_id in enumerate(track_ids):
            candidates = self.top_candidates[track_id]
            for name, conf, _ in candidates:
                if name in identity_names:
                    j = identity_names.index(name)
                    # Convert confidence to cost (lower cost = better match)
                    cost_matrix[i, j] = 1.0 - conf
        
        # Apply the Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create the optimal assignments
        optimal_assignments = {}
        for i, track_idx in enumerate(row_ind):
            track_id = track_ids[track_idx]
            identity_idx = col_ind[i]
            
            # Only proceed if the cost is reasonable (corresponds to confidence > 0.2)
            if cost_matrix[track_idx, identity_idx] < 0.8:
                identity_name = identity_names[identity_idx]
                confidence = 1.0 - cost_matrix[track_idx, identity_idx]
                optimal_assignments[track_id] = (identity_name, confidence)
        
        return optimal_assignments
    
    def apply_global_optimization(self):
        """
        Apply the results of global optimization to update track identities.
        
        Returns:
            int: Number of assignments that were changed
        """
        optimal_assignments = self.global_identity_optimization()
        if not optimal_assignments:
            return 0
            
        # Count how many assignments will change
        changes_made = 0
        
        # First, remove all assignments that will be changed
        identities_to_remove = set()
        for track_id, (identity_name, _) in optimal_assignments.items():
            # If this track already has this identity, no change needed
            current_identity = self.track_to_name.get(track_id)
            if current_identity == identity_name:
                continue
                
            # Otherwise, we'll make changes
            changes_made += 1
            
            # If this track has a different identity, remove it
            if track_id in self.track_to_name:
                old_identity = self.track_to_name[track_id]
                if old_identity != "Unknown" and old_identity != "Pending...":
                    identities_to_remove.add(old_identity)
            
            # If this identity is assigned to another track, remove that assignment
            if identity_name in self.name_to_track and self.name_to_track[identity_name] != track_id:
                old_track = self.name_to_track[identity_name]
                if old_track in self.track_to_name:
                    self.track_to_name[old_track] = "Unknown"
                    self.assignment_confidence[old_track] = 0.3  # Low confidence for Unknown
        
        # Now apply the new optimal assignments
        for track_id, (identity_name, confidence) in optimal_assignments.items():
            self.track_to_name[track_id] = identity_name
            self.name_to_track[identity_name] = track_id
            self.assignment_confidence[track_id] = confidence
            self._update_history(track_id, identity_name, confidence)
            
        return changes_made
