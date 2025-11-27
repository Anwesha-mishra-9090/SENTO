import hashlib
import hmac
import base64
import json
from datetime import datetime
import os


class GraphDatabase:
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.sensitive_fields = [
            'user_input', 'raw_features', 'context_tags',
            'risk_factors', 'intervention_provided'
        ]
        # Graph database storage (simulated with dictionaries)
        self.emotion_nodes = {}
        self.user_nodes = {}
        self.relationship_edges = {}
        self.node_counter = 0
        self.edge_counter = 0

    def _generate_encryption_key(self):
        """Generate a secure encryption key"""
        return base64.b64encode(os.urandom(32)).decode('utf-8')

    def create_emotion_node(self, emotion_data):
        """Create an emotion node in the graph"""
        node_id = f"emotion_{self.node_counter}"
        self.node_counter += 1

        # Encrypt sensitive data
        protected_data = self.encrypt_sensitive_data(emotion_data)

        self.emotion_nodes[node_id] = {
            'id': node_id,
            'type': 'emotion',
            'data': protected_data,
            'created_at': datetime.now().isoformat(),
            'relationships': []
        }
        return node_id

    def create_user_node(self, user_data):
        """Create a user node in the graph"""
        node_id = f"user_{self.node_counter}"
        self.node_counter += 1

        self.user_nodes[node_id] = {
            'id': node_id,
            'type': 'user',
            'data': user_data,
            'created_at': datetime.now().isoformat(),
            'relationships': []
        }
        return node_id

    def create_relationship(self, from_node, to_node, relationship_type, properties=None):
        """Create a relationship between nodes"""
        edge_id = f"edge_{self.edge_counter}"
        self.edge_counter += 1

        relationship = {
            'id': edge_id,
            'from_node': from_node,
            'to_node': to_node,
            'type': relationship_type,
            'properties': properties or {},
            'created_at': datetime.now().isoformat()
        }

        self.relationship_edges[edge_id] = relationship

        # Add relationship references to nodes
        if from_node in self.emotion_nodes:
            self.emotion_nodes[from_node]['relationships'].append(edge_id)
        elif from_node in self.user_nodes:
            self.user_nodes[from_node]['relationships'].append(edge_id)

        if to_node in self.emotion_nodes:
            self.emotion_nodes[to_node]['relationships'].append(edge_id)
        elif to_node in self.user_nodes:
            self.user_nodes[to_node]['relationships'].append(edge_id)

        return edge_id

    def get_node_relationships(self, node_id):
        """Get all relationships for a node"""
        relationships = []

        if node_id in self.emotion_nodes:
            for edge_id in self.emotion_nodes[node_id]['relationships']:
                if edge_id in self.relationship_edges:
                    relationships.append(self.relationship_edges[edge_id])
        elif node_id in self.user_nodes:
            for edge_id in self.user_nodes[node_id]['relationships']:
                if edge_id in self.relationship_edges:
                    relationships.append(self.relationship_edges[edge_id])

        return relationships

    def find_emotional_patterns(self, user_id, time_period="30d"):
        """Find emotional patterns for a user"""
        user_emotions = []

        # Find all emotion nodes related to this user
        for edge_id, edge in self.relationship_edges.items():
            if (edge['type'] == 'experienced' and
                    (edge['from_node'] == user_id or edge['to_node'] == user_id)):

                emotion_node_id = edge['to_node'] if edge['from_node'] == user_id else edge['from_node']
                if emotion_node_id in self.emotion_nodes:
                    user_emotions.append(self.emotion_nodes[emotion_node_id])

        # Analyze patterns (simplified)
        patterns = {
            'frequent_emotions': self._analyze_frequent_emotions(user_emotions),
            'emotional_transitions': self._analyze_emotional_transitions(user_emotions),
            'time_based_patterns': self._analyze_time_patterns(user_emotions)
        }

        return patterns

    def _analyze_frequent_emotions(self, emotion_nodes):
        """Analyze most frequent emotions"""
        emotion_counts = {}
        for node in emotion_nodes:
            emotion = node['data'].get('emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        return sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

    def _analyze_emotional_transitions(self, emotion_nodes):
        """Analyze emotional transitions"""
        transitions = []
        sorted_nodes = sorted(emotion_nodes, key=lambda x: x['created_at'])

        for i in range(1, len(sorted_nodes)):
            prev_emotion = sorted_nodes[i - 1]['data'].get('emotion', 'unknown')
            curr_emotion = sorted_nodes[i]['data'].get('emotion', 'unknown')

            if prev_emotion != curr_emotion:
                transitions.append(f"{prev_emotion} -> {curr_emotion}")

        return transitions

    def _analyze_time_patterns(self, emotion_nodes):
        """Analyze time-based emotional patterns"""
        # Group by time of day
        time_patterns = {'morning': [], 'afternoon': [], 'evening': [], 'night': []}

        for node in emotion_nodes:
            created_time = datetime.fromisoformat(node['created_at'])
            hour = created_time.hour

            if 5 <= hour < 12:
                time_patterns['morning'].append(node)
            elif 12 <= hour < 17:
                time_patterns['afternoon'].append(node)
            elif 17 <= hour < 22:
                time_patterns['evening'].append(node)
            else:
                time_patterns['night'].append(node)

        return time_patterns

    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data fields"""
        if not isinstance(data, dict):
            return data

        encrypted_data = data.copy()

        for field in self.sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                if isinstance(encrypted_data[field], (dict, list)):
                    # Convert to string for encryption
                    data_str = json.dumps(encrypted_data[field])
                    encrypted_data[f"{field}_encrypted"] = self._simple_encrypt(data_str)
                    del encrypted_data[field]
                else:
                    encrypted_data[f"{field}_encrypted"] = self._simple_encrypt(str(encrypted_data[field]))
                    del encrypted_data[field]

        # Add encryption metadata
        encrypted_data['encryption_metadata'] = {
            'encrypted_at': datetime.now().isoformat(),
            'encrypted_fields': self.sensitive_fields,
            'version': '1.0'
        }

        return encrypted_data

    def _simple_encrypt(self, text):
        """Simple encryption for demonstration"""
        text_bytes = text.encode('utf-8')
        key_bytes = self.encryption_key.encode('utf-8')

        # Simple XOR encryption (not secure for production)
        encrypted = bytearray()
        for i, byte in enumerate(text_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            encrypted.append(byte ^ key_byte)

        return base64.b64encode(bytes(encrypted)).decode('utf-8')

    def get_graph_statistics(self):
        """Get graph database statistics"""
        return {
            'total_nodes': len(self.emotion_nodes) + len(self.user_nodes),
            'emotion_nodes': len(self.emotion_nodes),
            'user_nodes': len(self.user_nodes),
            'relationships': len(self.relationship_edges),
            'node_counter': self.node_counter,
            'edge_counter': self.edge_counter
        }

    def export_graph_data(self, format='json'):
        """Export graph data"""
        graph_data = {
            'emotion_nodes': self.emotion_nodes,
            'user_nodes': self.user_nodes,
            'relationships': self.relationship_edges
        }

        if format == 'json':
            return json.dumps(graph_data, indent=2, default=str)
        else:
            return graph_data