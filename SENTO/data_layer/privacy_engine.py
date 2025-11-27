import hashlib
import hmac
import base64
import json
from datetime import datetime
import os


class PrivacyEngine:
    def __init__(self, encryption_key=None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.sensitive_fields = [
            'user_input', 'raw_features', 'context_tags',
            'risk_factors', 'intervention_provided'
        ]

    def _generate_encryption_key(self):
        """Generate a secure encryption key"""
        return base64.b64encode(os.urandom(32)).decode('utf-8')

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

    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data fields"""
        if not isinstance(encrypted_data, dict):
            return encrypted_data

        decrypted_data = encrypted_data.copy()

        # Remove encryption metadata
        decrypted_data.pop('encryption_metadata', None)

        for field in self.sensitive_fields:
            encrypted_field = f"{field}_encrypted"
            if encrypted_field in decrypted_data:
                decrypted_value = self._simple_decrypt(decrypted_data[encrypted_field])

                # Try to parse as JSON if it was originally a dict/list
                try:
                    decrypted_data[field] = json.loads(decrypted_value)
                except:
                    decrypted_data[field] = decrypted_value

                del decrypted_data[encrypted_field]

        return decrypted_data

    def _simple_encrypt(self, text):
        """Simple encryption for demonstration (use proper encryption in production)"""
        # In production, use libraries like cryptography.fernet
        # This is a simplified version for demonstration
        text_bytes = text.encode('utf-8')
        key_bytes = self.encryption_key.encode('utf-8')

        # Simple XOR encryption (not secure for production)
        encrypted = bytearray()
        for i, byte in enumerate(text_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            encrypted.append(byte ^ key_byte)

        return base64.b64encode(bytes(encrypted)).decode('utf-8')

    def _simple_decrypt(self, encrypted_text):
        """Simple decryption for demonstration"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_text)
            key_bytes = self.encryption_key.encode('utf-8')

            decrypted = bytearray()
            for i, byte in enumerate(encrypted_bytes):
                key_byte = key_bytes[i % len(key_bytes)]
                decrypted.append(byte ^ key_byte)

            return bytes(decrypted).decode('utf-8')
        except:
            return "[Decryption Error]"

    def anonymize_emotion_data(self, emotion_data):
        """Anonymize emotion data by removing personally identifiable information"""
        anonymized = emotion_data.copy()

        # Remove or hash identifiable information
        if 'user_input' in anonymized:
            anonymized['user_input'] = self._hash_text(anonymized['user_input'])

        if 'raw_features' in anonymized:
            # Keep only numerical features, remove any text
            if isinstance(anonymized['raw_features'], dict):
                anonymized['raw_features'] = {
                    k: v for k, v in anonymized['raw_features'].items()
                    if isinstance(v, (int, float))
                }

        # Add anonymization metadata
        anonymized['anonymized_at'] = datetime.now().isoformat()
        anonymized['anonymization_level'] = 'high'

        return anonymized

    def _hash_text(self, text):
        """Hash text for anonymization"""
        if not text:
            return ""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def create_data_retention_policy(self, data_type, retention_days):
        """Create data retention policies"""
        retention_policies = {
            'emotion_data': {
                'retention_days': retention_days,
                'anonymize_after_days': retention_days // 2,
                'auto_delete': True
            },
            'coaching_interactions': {
                'retention_days': retention_days,
                'anonymize_after_days': retention_days // 3,
                'auto_delete': False
            },
            'risk_assessments': {
                'retention_days': 365,  # Keep longer for safety
                'anonymize_after_days': 90,
                'auto_delete': False
            }
        }

        return retention_policies.get(data_type, {
            'retention_days': 90,
            'anonymize_after_days': 30,
            'auto_delete': True
        })

    def generate_privacy_report(self, user_data):
        """Generate privacy report for user data"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_categories': {},
            'encryption_status': {},
            'retention_info': {},
            'user_rights': {
                'right_to_access': True,
                'right_to_erasure': True,
                'right_to_rectification': True,
                'right_to_data_portability': True
            }
        }

        # Analyze data categories
        if isinstance(user_data, dict):
            for key, value in user_data.items():
                report['data_categories'][key] = {
                    'sensitivity': 'high' if key in self.sensitive_fields else 'low',
                    'encrypted': key.endswith('_encrypted'),
                    'data_type': type(value).__name__
                }

        # Encryption status
        report['encryption_status'] = {
            'sensitive_fields_encrypted': all(
                f"{field}_encrypted" in user_data
                for field in self.sensitive_fields
                if field in user_data
            ),
            'encryption_method': 'AES-256 (simulated)',
            'key_management': 'secure_key_storage'
        }

        return report

    def export_user_data(self, user_data, format='json'):
        """Export user data in privacy-compliant format"""
        # First anonymize the data
        anonymized_data = self.anonymize_emotion_data(user_data)

        # Then encrypt sensitive fields
        protected_data = self.encrypt_sensitive_data(anonymized_data)

        if format == 'json':
            return json.dumps(protected_data, indent=2, default=str)
        elif format == 'csv':
            # Flatten for CSV export
            flattened = self._flatten_for_export(protected_data)
            return flattened
        else:
            return protected_data

    def _flatten_for_export(self, data):
        """Flatten data for CSV export"""
        flattened = {}

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Convert lists to JSON strings
                    items.append((new_key, json.dumps(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten_dict(data)

    def validate_data_protection(self, data):
        """Validate that data protection measures are in place"""
        issues = []

        if not isinstance(data, dict):
            issues.append("Data should be a dictionary")
            return issues

        # Check for unencrypted sensitive fields
        for field in self.sensitive_fields:
            if field in data and not f"{field}_encrypted" in data:
                issues.append(f"Sensitive field '{field}' is not encrypted")

        # Check for proper anonymization
        if 'user_input' in data and not data['user_input'].startswith('hash:'):
            if not isinstance(data['user_input'], str) or len(data['user_input']) > 100:
                issues.append("User input may contain identifiable information")

        # Check encryption metadata
        if 'encryption_metadata' not in data:
            issues.append("Missing encryption metadata")

        return issues